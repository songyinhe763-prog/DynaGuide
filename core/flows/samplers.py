"""ODE samplers for flow policies, with a first-class guidance hook.

The samplers integrate the probability-flow ODE

.. math:: \\frac{\\mathrm d x}{\\mathrm d t} = v_\\theta(x, t)

from ``t = t_start ~ 0`` (noise) to ``t = t_end = 1`` (action chunk).

Guidance
--------
A guidance callable is injected at *every* velocity evaluation, i.e.

.. math:: \\frac{\\mathrm d x}{\\mathrm d t} = v_\\theta(x, t) + \\Delta v(x, t, v_\\theta)

with ``Delta v = -scale * w(t) * grad_x J(x)`` produced by
:class:`core.flow_dynaguide.FlowDynaGuide`.  This is the exact analogue of DynaGuide's

.. code-block:: python

    noise_pred = noise_pred - (1 - alphas_cumprod[k]).sqrt() * scaled_grad

for a velocity-field model: the guidance is a *drift* correction on the ODE, not a
correction on a noise prediction.  See ``docs/FLOW_DYNAGUIDE.md`` for the derivation of
the ``w(t)`` that makes the two equivalent.

Stochastic sampling (``ss``)
----------------------------
DynaGuide's experiments repeat the guided update ``ss`` times per macro step while the
DDIM timestep does not decrease (``run_dynaguide.py: for ss_iter in range(ss)``).  For a
continuous-time ODE we expose two faithful interpretations:

``ss_mode="split"`` (default)
    ``ss`` inner sub-steps of size ``dt / ss`` that *do* advance time.  Guidance is
    applied ``ss`` times per macro step -> less discretisation error and a smoother
    guidance signal (this is the recommended mode).
``ss_mode="refine"``
    ``ss`` inner sub-steps that keep the *same* time level and each advance by
    ``dt / ss``: a fixed-noise-level refinement, closest to the literal DynaGuide loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

VelocityFn = Callable[..., torch.Tensor]
GuidanceFn = Callable[..., Dict[str, torch.Tensor]]


@dataclass
class FlowSamplerConfig:
    """Configuration of :class:`FlowSampler`."""

    n_steps: int = 4
    method: str = "euler"  # euler | midpoint | heun
    t_start: float = 1e-2
    t_end: float = 1.0
    ss: int = 1
    ss_mode: str = "split"  # split | refine
    clip: Optional[float] = None
    time_scale: float = 1.0  # multiply t before feeding the network (FlowPolicy uses 99)

    def validate(self) -> "FlowSamplerConfig":
        if self.method not in ("euler", "midpoint", "heun"):
            raise ValueError(f"unknown sampler method {self.method!r}")
        if self.ss_mode not in ("split", "refine"):
            raise ValueError(f"unknown ss_mode {self.ss_mode!r}")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.ss < 1:
            raise ValueError("ss must be >= 1")
        if not 0.0 <= self.t_start < self.t_end <= 1.0:
            raise ValueError("require 0 <= t_start < t_end <= 1")
        return self


@dataclass
class SamplerTrace:
    """Everything the guided sampler saw, for analysis / figures."""

    states: List[torch.Tensor] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    velocities: List[torch.Tensor] = field(default_factory=list)
    guidance: List[Optional[torch.Tensor]] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    nfe: int = 0

    def as_dict(self) -> dict:
        def _cpu(x):
            return None if x is None else x.detach().cpu()

        return {
            "states": [_cpu(s) for s in self.states],
            "times": list(self.times),
            "guidance": [_cpu(g) for g in self.guidance],
            "energies": list(self.energies),
            "guidance_norms": [None if g is None else float(g.detach().norm()) for g in self.guidance],
            "nfe": self.nfe,
        }


class FlowSampler:
    """Guided ODE sampler for flow-matching action-chunk policies."""

    def __init__(self, config: Optional[FlowSamplerConfig] = None):
        self.config = (config or FlowSamplerConfig()).validate()

    # ------------------------------------------------------------------ api ----
    def sample(
        self,
        velocity_fn: VelocityFn,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        guidance: Optional[GuidanceFn] = None,
        ctx: Optional[dict] = None,
        record: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Integrate the (optionally guided) ODE.

        Parameters
        ----------
        velocity_fn:
            ``velocity_fn(x, t, cond) -> v``.  ``t`` is a ``(B,)`` tensor.
        x0:
            Initial noise, shape ``(B, T, D)``.
        cond:
            Global conditioning (observation features) passed straight through.
        guidance:
            ``guidance(x, t, v, ctx) -> {"delta_v": ..., "energy": ...}``.  Receives the
            *differentiable* current sample (``requires_grad=True``) and must return a
            tensor with the same shape as ``x``.
        ctx:
            Arbitrary payload (state dict, task id, ...) forwarded to the guidance.
        record:
            Keep the trajectory / guidance norms for logging.

        Returns
        -------
        dict with ``actions`` (B, T, D), ``trace`` (optional) and ``nfe``.
        """
        cfg = self.config
        trace = SamplerTrace() if record else None
        x = x0
        times = torch.linspace(cfg.t_start, cfg.t_end, cfg.n_steps + 1)
        nfe = 0
        last_delta = None
        last_energy = None

        for i in range(cfg.n_steps):
            t0 = float(times[i])
            t1 = float(times[i + 1])
            macro_dt = t1 - t0
            inner = cfg.ss
            if inner > 1 and cfg.ss_mode == "split":
                dt = macro_dt / inner
                t_points = [t0 + dt * j for j in range(inner)]
            elif inner > 1 and cfg.ss_mode == "refine":
                dt = macro_dt / inner
                t_points = [t0 for _ in range(inner)]
            else:
                dt = macro_dt
                t_points = [t0]

            for t_cur in t_points:
                t_cur = min(t_cur, cfg.t_end - 1e-6)
                if guidance is None:
                    x = x.detach()
                x, v, delta, energy, n = self._guided_velocity(
                    velocity_fn, x, t_cur, cond, guidance, ctx
                )
                nfe += n
                last_delta, last_energy = delta, energy

                if cfg.method == "euler":
                    x = x + v * dt
                elif cfg.method == "midpoint":
                    x_mid = x + v * (dt / 2.0)
                    x_mid, v_mid, _, _, n2 = self._guided_velocity(
                        velocity_fn, x_mid, t_cur + dt / 2.0, cond, guidance, ctx
                    )
                    nfe += n2
                    x = x + v_mid * dt
                elif cfg.method == "heun":
                    x_euler = x + v * dt
                    x_euler, v_end, _, _, n2 = self._guided_velocity(
                        velocity_fn, x_euler, t_cur + dt, cond, guidance, ctx
                    )
                    nfe += n2
                    x = x + 0.5 * (v + v_end) * dt
                else:  # pragma: no cover - validated in __init__
                    raise ValueError(cfg.method)

                if cfg.clip is not None:
                    x = x.clamp(-cfg.clip, cfg.clip)

                if record:
                    trace.states.append(x.detach().clone())
                    trace.times.append(float(t_cur))
                    trace.velocities.append(v.detach().clone())
                    trace.guidance.append(None if delta is None else delta.detach().clone())
                    trace.energies.append(float("nan") if energy is None else float(energy))

        if record:
            trace.nfe = nfe

        return {
            "actions": x,
            "trace": trace,
            "nfe": nfe,
            "last_guidance": last_delta,
            "last_energy": last_energy,
        }

    # -------------------------------------------------------------- helpers ----
    @torch.no_grad()
    def _batch_time(self, t_cur: float, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), float(t_cur), device=x.device, dtype=x.dtype)

    def _guided_velocity(self, velocity_fn, x, t_cur, cond, guidance, ctx):
        """Evaluate ``v`` and the guidance correction at time ``t_cur``."""
        t_vec = self._batch_time(t_cur, x)
        x_in = x if guidance is None else x.detach().requires_grad_(True)
        v = velocity_fn(x_in, t_vec * self.config.time_scale, cond)
        if guidance is None:
            return x, v, None, None, 1
        out = guidance(x_in, torch.full_like(t_vec, t_cur), v, ctx)
        # accept both a dict ({"delta_v": ..., "energy": ...}) and a bare correction tensor,
        # so that DynaGuide-style guidance callables plug in unchanged
        if isinstance(out, dict):
            delta = out["delta_v"]
            energy = out.get("energy", None)
        else:
            delta = out
            energy = None
        return x_in, v + delta, delta, energy, 1

    # ------------------------------------------------------------ short-cut ----
    def sample_actions(self, velocity_fn, x0, cond=None, guidance=None, ctx=None):
        return self.sample(velocity_fn, x0, cond=cond, guidance=guidance, ctx=ctx)["actions"]


def euler_flow_sample(velocity_fn, x0, n_steps=4, t_start=1e-2, t_end=1.0) -> torch.Tensor:
    """Minimal unguided euler integration (kept for quick checks / FlowPolicy parity)."""
    cfg = FlowSamplerConfig(n_steps=n_steps, method="euler", t_start=t_start, t_end=t_end)
    return FlowSampler(cfg).sample_actions(velocity_fn, x0)
