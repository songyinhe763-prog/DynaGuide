"""Probability paths for (conditional) flow matching.

We follow the convention used by FlowPolicy / the ``consistency_flow_matching`` repo,
which is also Lipman et al.'s *target* CFM parameterisation with ``sigma = 0``:

.. math::

    x_t = (1 - t)\\, \\varepsilon + t\\, x_1 , \\qquad t \\in [0, 1]

where :math:`\\varepsilon \\sim \\mathcal N(0, I)` is the source sample (``t = 0``) and
:math:`x_1` is the data sample (``t = 1``).  The conditional vector field is constant
along the path,

.. math::

    u_t(x_1 \\mid \\varepsilon) = x_1 - \\varepsilon ,

so the ODE :math:`\\mathrm d x / \\mathrm d t = v_\\theta(x, t)` is integrated from
``t = 0`` (noise) to ``t = 1`` (action chunk).

Note that this is the *same* path as the usual rectified-flow ``(1-t) x_0 + t x_1``
formulation with ``x_0 = eps``; only the sign of the time axis differs from the
"diffusion" convention used by DDIM in DynaGuide.

Because the map from a velocity to the implied endpoints is linear and known in closed
form, guidance in the endpoint (clean action chunk) space and guidance in the velocity
space can be converted into each other exactly::

    x1_hat = x_t + (1 - t) * v        (project_onto_data)
    x0_hat = x_t - t * v              (project_onto_noise)
    v      = (x1_hat - x_t) / (1 - t)
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch


def pad_t_like_x(t, x: torch.Tensor):
    """Reshape a ``(B,)`` time vector so that it broadcasts against ``x`` (B, ...)."""
    if not torch.is_tensor(t):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def time_embedding_scale(t: torch.Tensor, multiplier: float = 99.0) -> torch.Tensor:
    """FlowPolicy feeds ``t * 99`` to the sinusoidal time embedding.

    The multiplier is a pure embedding rescaling (the sinusoid has a fixed frequency
    range); it lets a diffusion-era UNet see times spread over the range it was tuned
    for.  It does **not** change the ODE, only the network input.
    """
    return t * multiplier


def flow_noise_fraction(t: torch.Tensor, sigma_min: float = 0.0) -> torch.Tensor:
    """Equivalent *noise fraction* of the linear path, i.e. ``sigma / sqrt(1 + sigma^2)``.

    For the linear interpolant the signal-to-noise ratio is

    .. math:: \\mathrm{SNR}(t) = \\frac{t^2}{(1-t)^2} ,\\quad \\sigma(t) = \\frac{1-t}{t}

    so ``sigma / sqrt(1 + sigma^2)`` maps the flow time onto the
    ``sqrt(1 - alpha_bar)`` quantity that appears in the DynaGuide/DDIM update.  This is
    what one should use when porting a guidance weight schedule from a VP diffusion
    sampler to a linear flow sampler (see ``core.flow_dynaguide.GuidanceWeightSpec``).
    """
    t = torch.as_tensor(t, dtype=torch.float32) if not torch.is_tensor(t) else t
    t = t.clamp(min=max(sigma_min, 1e-6), max=1.0 - 1e-6)
    sigma = (1.0 - t) / t
    return sigma / torch.sqrt(1.0 + sigma ** 2)


def flow_sigma(t: torch.Tensor) -> torch.Tensor:
    """Noise-to-signal ratio ``sigma(t) = (1 - t) / t`` of the linear path."""
    t = t.clamp(min=1e-6, max=1.0 - 1e-6)
    return (1.0 - t) / t


class LinearFlowPath:
    """Linear (rectified flow) probability path with an optional gaussian smoothing.

    Parameters
    ----------
    sigma:
        Standard deviation of the extra gaussian smoothing on the interpolant.  The
        default ``0.0`` reproduces FlowPolicy's ``ConsistencyFM`` (deterministic
        straight lines, ``sigma_t = 1 - t``).
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        self.sigma = float(sigma)

    # ------------------------------------------------------------------ path ----
    def mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mean of the path: ``(1 - t) x0 + t x1``."""
        t = pad_t_like_x(t, x0)
        return (1.0 - t) * x0 + t * x1

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Path standard deviation, ``1 - (1 - sigma) t`` (deterministic: ``1 - t``)."""
        return 1.0 - (1.0 - self.sigma) * t

    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.mu_t(x0, x1, t)
        sigma_t = pad_t_like_x(self.sigma_t(t), x0)
        return mu_t + sigma_t * epsilon

    def sample(self, x1: torch.Tensor, t: Optional[torch.Tensor] = None):
        """Draw ``(t, x_t, u_t, eps)`` for a batch of data samples ``x1``."""
        if t is None:
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        eps = torch.randn_like(x1)
        x0 = eps
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.target_velocity(x0, x1)
        return t, xt, ut, eps

    def target_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Conditional vector field ``u_t = x1 - x0`` (constant along the path)."""
        return x1 - x0

    # ------------------------------------------------------- projections ----
    # 这两个投影是整条链路的枢纽。
    #   project_onto_data 给出“网络认为最终要执行的动作块”，
    #   引导能量、动力学模型、物理约束全部作用在它上面；
    #   而不是作用在带噪的 x_t 上（那样是分布外查询，梯度会有偏）。
    def project_onto_data(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``x1_hat = x_t + (1 - t) v`` -- the clean action chunk implied by ``v``."""
        t = pad_t_like_x(t, x_t)
        return x_t + (1.0 - t) * v

    def project_onto_noise(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``x0_hat = x_t - t v`` -- the implied source sample."""
        t = pad_t_like_x(t, x_t)
        return x_t - t * v

    def velocity_from_data(self, x_t: torch.Tensor, t: torch.Tensor, x1_hat: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`project_onto_data`."""
        t = pad_t_like_x(t, x_t).clamp(max=1.0 - 1e-6)
        return (x1_hat - x_t) / (1.0 - t)

    # ------------------------------------------------------------ extras ----
    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal to noise ratio of the interpolant, ``t^2 / (1-t)^2``."""
        t = t.clamp(min=1e-6, max=1.0 - 1e-6)
        return (t / (1.0 - t)) ** 2

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LinearFlowPath(sigma={self.sigma})"


def interpolate_endpoint(
    x_t: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    segment_end: torch.Tensor,
    method: str = "euler",
) -> torch.Tensor:
    """Advance ``x_t`` to ``segment_end`` with a single (forward) euler step.

    This is FlowPolicy's ``f`` / ``f_euler`` used inside the consistency loss, and is
    also the projection used by the guidance energy (``segment_end = 1`` gives the
    predicted action chunk).
    """
    seg = segment_end if torch.is_tensor(segment_end) else torch.as_tensor(segment_end, dtype=x_t.dtype, device=x_t.device)
    seg = pad_t_like_x(seg, x_t) if seg.dim() == 1 else seg
    t_p = pad_t_like_x(t, x_t)
    return x_t + (seg - t_p) * v


def ddim_equivalence_table(num_train_timesteps: int = 100, beta_schedule: str = "squaredcos_cap_v2"):
    """Utility: ``(t_flow, sqrt(1 - alpha_bar), (1 - alpha_bar)/sqrt(alpha_bar))``.

    Used by the documentation / tests to show which guidance weight a DynaGuide weight
    on the DDIM noise prediction corresponds to on a linear flow path.  We compute the
    VP betas without ``diffusers`` so this module stays dependency free.
    """
    if beta_schedule != "squaredcos_cap_v2":
        raise NotImplementedError(beta_schedule)
    steps = num_train_timesteps + 1
    s = torch.linspace(0, 1, steps, dtype=torch.float64)
    f = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = (f / f[0]).clamp(max=1.0)[:-1]
    rows = []
    for k in range(num_train_timesteps):
        ab = float(alphas_cumprod[num_train_timesteps - 1 - k])
        # flow time whose SNR matches the VP SNR of level k
        sigma = math.sqrt(max(1e-8, (1.0 - ab) / ab))
        t_flow = 1.0 / (1.0 + sigma)
        rows.append(
            dict(
                k=k,
                alpha_bar=ab,
                sqrt_one_minus_alpha_bar=math.sqrt(max(0.0, 1.0 - ab)),
                clean_correction=(1.0 - ab) / math.sqrt(ab),
                t_flow=t_flow,
            )
        )
    return rows
