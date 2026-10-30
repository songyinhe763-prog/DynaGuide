"""DynaGuide for velocity-field (flow-matching) policies.

The original DynaGuide steers a *DDIM noise prediction*:

.. code-block:: python

    noise_pred = noise_pred - (1 - alphas_cumprod[k]).sqrt() * scaled_grad

This module defines the mathematically consistent flow-matching counterpart and the
weight schedules that make the migration defensible.  The derivation is shipped in
``docs/FLOW_DYNAGUIDE.md``; the short version:

1. ``sqrt(1 - alpha_bar_k)`` in DDIM is **the Jacobian of the epsilon <-> score
   parameterisation** (``d eps / d score = -sqrt(1 - alpha_bar)``).  It is not a free
   knob: it makes a fixed guidance strength ``s`` on the *sample-space* score become a
   correction of size ``s * sqrt(1 - alpha_bar)`` in epsilon space, and it anneals the
   guidance to zero as the sample becomes clean.
2. For the linear interpolant ``x_t = (1 - t) eps + t a1`` the corresponding marginal
   noise standard deviation is ``sigma_t = 1 - t``, and the score/velocity relation is
   ``eps_eq = x_t - t v = -(1 - t) grad log p_t``.  A guidance term added to the score
   therefore becomes a velocity correction of size

   .. math:: \\Delta v = -s\\, \\lambda(t)\\, \\nabla_{x_t} J

   with ``lambda(t)`` the flow analogue of ``sqrt(1 - alpha_bar)``
   (``(1 - t)**2 / 2`` from the Fokker-Planck drift ``f + (g^2/2) grad log p`` with
   ``g = sigma_t``, recovers both the anneal and the bounded ``t -> 0`` limit).
3. The cost ``J`` must be evaluated on the *Tweedie / one-euler clean action estimate*
   ``x1_hat = x_t + (1 - t) v`` -- which is exactly FlowPolicy's own ``f_theta`` -- not on
   the noisy sample: the critic was trained on clean action chunks, so a gradient taken
   at ``x_t`` is out of distribution and biased.
4. The gradient is transported with the identity map (``torch.autograd`` on the detached
   velocity) by default -- the "stop-grad / QGF" variant -- and can optionally include the
   sampler Jacobian ``I + (1 - t) d v / d x_t`` (RectifID's anchored guidance).

Everything else of DynaGuide is reused unchanged: soft-min (log-sum-exp) over the
positive guidance conditions, subtracted soft-min over the negative ones, the temperature
``sigma`` (``alpha`` in the original code), the guidance scale ``s`` and the ``M``-fold
inner "stochastic sampling" loop (``ss`` in the original code).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch

from core.flows.paths import flow_noise_fraction


# ------------------------------------------------------------------ schedules ----
@dataclass
class GuidanceWeightSpec:
    """``lambda(t)``: how much guidance to apply at flow time ``t`` (0 = noise, 1 = data).

    Modes
    -----
    ``fp``      ``(1 - t)**2 / 2``  -- recommended default: bounded at ``t -> 0`` and
                self-annealing at ``t -> 1`` (Fokker-Planck drift of the linear path).
    ``linear``  ``1 - t`` -- the marginal noise std, a first-order analogue.
    ``ddim``    ``(1 - t) / t`` -- the *strict* DDIM-epsilon equivalent; diverges at
                ``t -> 0`` so it is clamped at ``clamp_max`` (default 50) and should only
                be used with ``t_start >= 0.05``.
    ``noise_fraction``  ``sigma / sqrt(1 + sigma**2)`` with ``sigma = (1-t)/t``: exactly
                the ``sqrt(1 - alpha_bar)`` of the equivalent VP schedule.
    ``const``   ``1`` -- pure ``s`` knob (QGF style).
    ``power``   ``(1 - t)**power``.
    """

    mode: str = "fp"
    power: float = 1.0
    clamp_max: Optional[float] = 50.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t) if not torch.is_tensor(t) else t
        t = t.clamp(min=1e-4, max=1.0)
        if self.mode == "const":
            w = torch.ones_like(t)
        elif self.mode == "linear":
            w = 1.0 - t
        elif self.mode == "fp":
            w = 0.5 * (1.0 - t) ** 2
        elif self.mode == "power":
            w = (1.0 - t) ** self.power
        elif self.mode == "ddim":
            w = (1.0 - t) / t
        elif self.mode == "noise_fraction":
            w = flow_noise_fraction(t)
        else:
            raise ValueError(f"unknown weight schedule {self.mode!r}")
        if self.clamp_max is not None:
            w = w.clamp(max=self.clamp_max)
        return w


# ------------------------------------------------------------------- guidance ----
@dataclass
class FlowDynaGuide:
    """Turns an energy (cost) into a velocity-space guidance correction.

    Parameters
    ----------
    energy:
        Callable ``energy(actions, proprio, ctx) -> (cost, info)``; lower is better.
    scale:
        DynaGuide's ``s``.  NOTE: because ``lambda(t)`` already carries the ``(1-t)**2/2``
        anneal, ``s`` lives on a different scale than the DDIM one -- sweep it.
    weight:
        The ``lambda(t)`` schedule.
    mode:
        ``"endpoint"`` (default, Tweedie clean-action estimate -- recommended) or
        ``"velocity"`` (DynaGuide-literal ablation: evaluate the cost on the noisy chunk).
    jacobian:
        ``False`` (default) transports the gradient with the identity map (stop-grad /
        QGF).  ``True`` includes ``I + (1 - t) dv/dx`` (RectifID anchored), which needs a
        differentiable velocity and costs an extra backward through the policy net.
    clip_norm:
        Clip the gradient to this L2 norm (OmniGuide's ``alpha`` guard).
    clip_value:
        Elementwise clip of the gradient.
    ema:
        Momentum for the guidance direction across sampler steps; ``0`` disables.
    """

    energy: Callable
    scale: float = 1.0
    weight: GuidanceWeightSpec = field(default_factory=GuidanceWeightSpec)
    mode: str = "endpoint"
    jacobian: bool = False
    clip_norm: Optional[float] = 5.0
    clip_value: Optional[float] = None
    ema: float = 0.0
    name: str = "flow_dynaguide"

    _ema_buffer: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------- helpers ----
    def clean_estimate(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``x1_hat = x_t + (1 - t) v`` (the linear-path Tweedie estimate)."""
        t_e = t.reshape(-1, *([1] * (x_t.dim() - 1)))
        return x_t + (1.0 - t_e) * v

    def _postprocess(self, g: torch.Tensor) -> torch.Tensor:
        if self.clip_value is not None:
            g = g.clamp(-self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = g.norm()
            if float(norm) > self.clip_norm:
                g = g * (self.clip_norm / (norm + 1e-9))
        if self.ema > 0:
            if self._ema_buffer is None or self._ema_buffer.shape != g.shape:
                self._ema_buffer = g.detach().clone()
            else:
                self._ema_buffer = self.ema * self._ema_buffer + (1.0 - self.ema) * g.detach()
            g = self._ema_buffer
        return g

    def reset(self):
        self._ema_buffer = None

    # ------------------------------------------------------------- forward ----
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
        ctx: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        ctx = dict(ctx or {})
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        if self.mode == "endpoint":
            v_ref = v if self.jacobian else v.detach()
            target = self.clean_estimate(x, t, v_ref)
        elif self.mode == "velocity":
            target = x
        else:
            raise ValueError(f"unknown guidance mode {self.mode!r}")

        cost, info = self.energy(target, ctx.get("proprio"), ctx)
        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]
        grad = torch.nan_to_num(grad)
        grad = self._postprocess(grad)

        w = self.weight(t).reshape(-1, *([1] * (grad.dim() - 1))).to(grad.dtype)
        delta_v = -float(self.scale) * w * grad  # descend the energy

        out = {
            "delta_v": delta_v,
            "energy": cost.detach(),
            "grad_norm": grad.norm().detach(),
            "weight": w.detach().mean(),
            "target": target.detach(),
        }
        for key, value in (info or {}).items():
            out[f"info/{key}"] = value
        return out

    # ------------------------------------------------- compatibility shim ----
    def guidance_fn(self) -> Callable:
        """Callable with ``(state, actions, t, velocity, ctx)`` for the robomimic policy."""

        def _fn(state, actions, t=None, velocity=None, ctx=None):
            ctx = dict(ctx or {})
            ctx.setdefault("state", state)
            if velocity is None:
                raise ValueError("flow guidance needs the current velocity estimate")
            t_vec = torch.as_tensor(t, dtype=actions.dtype, device=actions.device) if t is not None else torch.zeros(
                1, dtype=actions.dtype, device=actions.device
            )
            return self(actions, t_vec, velocity, ctx)

        return _fn


# ------------------------------------------------------------------- factory ----
def make_flow_guidance(
    energy: Callable,
    scale: float = 1.0,
    weight_mode: str = "fp",
    mode: str = "endpoint",
    jacobian: bool = False,
    clip_norm: Optional[float] = 5.0,
    clip_value: Optional[float] = None,
    ema: float = 0.0,
    **weight_kwargs,
) -> FlowDynaGuide:
    """Convenience constructor mirroring the ``--scale/--alpha/--ss`` CLI of DynaGuide."""
    return FlowDynaGuide(
        energy=energy,
        scale=scale,
        weight=GuidanceWeightSpec(mode=weight_mode, **weight_kwargs),
        mode=mode,
        jacobian=jacobian,
        clip_norm=clip_norm,
        clip_value=clip_value,
        ema=ema,
    )


# ------------------------------------------------- drop-in for run_dynaguide.py ----
def calculate_flow_classifier_guidance(
    model,
    good_dataset=None,
    scale: float = 1.0,
    main_camera: str = "third_person",
    bad_dataset=None,
    alpha: float = 30.0,
    weight_mode: str = "fp",
    max_examples: Optional[int] = None,
    mode: str = "endpoint",
    jacobian: bool = False,
    clip_norm: Optional[float] = 5.0,
    **kwargs,
):
    """Flow counterpart of :func:`core.dynaguide.calculate_classifier_guidance`.

    Returns ``(guidance, good_embeddings, bad_embeddings)`` exactly like the DDIM version,
    so ``run_dynaguide.py`` can swap the two by changing one flag.  ``guidance`` has the
    signature ``(state, actions, t, velocity) -> {"delta_v": ..., "energy": ...}``.

    The only difference is that the guidance is a *velocity* correction and needs the
    current flow time; see the module docstring for the derivation.
    """
    from core.flow_energies import LatentMatchEnergy
    from core.dynaguide import prepare_np

    good_embeddings = None
    bad_embeddings = None

    def _bank(dataset):
        if dataset is None:
            return None
        embeddings = []
        idx = 0
        for length in dataset.lengths_list:
            idx += length
            sample = dataset.get_labeled_item(idx - 1)
            state, action, _label = prepare_np(sample[0]), prepare_np(sample[1]), sample[2]
            state = {k: torch.unsqueeze(v, dim=0) for k, v in state.items()}
            action = torch.unsqueeze(action, dim=0)
            with torch.no_grad():
                emb = model.state_action_embedding(state, action).flatten(start_dim=1)
            embeddings.append(emb.clone())
        out = torch.cat(embeddings, dim=0)
        if max_examples is not None:
            out = out[:max_examples]
        return out

    good_embeddings = _bank(good_dataset)
    bad_embeddings = _bank(bad_dataset)

    def latent_predictor(state, actions):
        relevant = {main_camera: state[main_camera][:, -1] * 255}
        relevant["proprio"] = state["proprio"][:, -1]
        return model.state_action_embedding(relevant, actions).flatten(start_dim=1)

    energy = LatentMatchEnergy(
        latent_predictor=latent_predictor,
        positives=good_embeddings,
        negatives=bad_embeddings,
        alpha=alpha,
    )
    guide = make_flow_guidance(
        energy,
        scale=scale,
        weight_mode=weight_mode,
        mode=mode,
        jacobian=jacobian,
        clip_norm=clip_norm,
        **kwargs,
    )
    return guide.guidance_fn(), good_embeddings, bad_embeddings

