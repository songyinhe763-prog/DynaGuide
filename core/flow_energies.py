"""Guidance energies (differentiable cost functions) used by the flow DynaGuide.

Two families live here:

**Learned (DynaGuide-faithful)** -- :class:`LatentMatchEnergy`
    The energy is the (soft-min) distance in a *latent* space between the predicted
    end-of-chunk state and a bank of goal embeddings, optionally minus the distance to a
    bank of negative embeddings.  ``latent_predictor(state, actions)`` is any model with
    DynaGuide's ``state_action_embedding`` semantics (the DINO dynamics model shipped in
    ``core/dynamics_models.py``, or the light EM-bedded model in
    ``core/satellite/latent.py``).

**Analytic / physics-informed** -- :class:`SatellitePhysicsEnergy`
    A closed-form, differentiable surrogate of the disassembly dynamics
    (:class:`DifferentiableDisassemblyModel`) that scores an action chunk by

    * *protocol*: did the chunk actually progress the fastener (twist -> pull)?
    * *capture*: does it end near the staging pose once the module is free?
    * *force*: does it exceed the contact-force budget?
    * *momentum*: how much base angular momentum does the reaction inject?
    * *axis*: how much tangential (blocked) motion does it command?
    * *approach*: is the gripper even at the right module?

    The momentum term is the microgravity-specific addition: a "successful" but violent
    extraction that spins the free-floating satellite is not a success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from core.satellite.env import (
    BASE_ANGVEL,
    BASE_QUAT,
    EE_POS,
    EE_ROTVEC,
    GRIP,
    module_block,
)
from core.satellite.spec import SatelliteSpec, make_sc1_spec


# ------------------------------------------------------------------- torch 3D ----
def quat_normalize_t(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def quat_multiply_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_matrix_t(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize_t(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    zero = torch.zeros_like(w)
    one = torch.ones_like(w)
    rows = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return rows.reshape(*q.shape[:-1], 3, 3)


def rotvec_from_quat_t(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize_t(q)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign
    w = q[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.arccos(w)
    s = torch.sqrt((1.0 - w * w).clamp(min=1e-12))
    v = q[..., 1:] / s.unsqueeze(-1)
    return v * angle.unsqueeze(-1)


# ------------------------------------------------- differentiable surrogate ----
@dataclass
class DifferentiableDisassemblyModel:
    """Closed form, differentiable approximation of :class:`SatelliteEnv` for a chunk.

    It mirrors the environment's rules (twist -> unlock -> breakaway pull, blocked motion
    turned into a reaction wrench on the free-floating base) but keeps every operation
    smooth enough to differentiate w.r.t. the action chunk.  It is used as the *energy*
    for the physics-informed part of the guidance, and as the surrogate for ablations.
    """

    spec: SatelliteSpec = field(default_factory=make_sc1_spec)
    dt_scale: float = 1.0

    def rollout(
        self,
        actions: torch.Tensor,
        proprio: torch.Tensor,
        module_index: int,
        grasped: bool = True,
    ) -> Dict[str, torch.Tensor]:
        spec = self.spec
        mod = spec.modules[module_index]
        j = mod.joint
        device = actions.device
        if actions.dim() == 3:
            if actions.shape[0] != 1:
                raise ValueError("the differentiable disassembly model scores one chunk at a time")
            actions = actions[0]
        H = actions.shape[0]
        proprio = proprio.reshape(-1)  # accept (D,) or (1, D)

        dp = actions[:, 0:3] * spec.dp_scale
        dth = actions[:, 3:6] * spec.drot_scale
        axis = torch.as_tensor(j.axis, dtype=actions.dtype, device=device)
        attach = torch.as_tensor(mod.attach_np, dtype=actions.dtype, device=device)
        capture = torch.as_tensor(mod.capture_np, dtype=actions.dtype, device=device)
        grasp = torch.as_tensor(mod.grasp_pose, dtype=actions.dtype, device=device)

        ee = proprio[EE_POS].to(actions.dtype)
        R = quat_matrix_t(proprio[BASE_QUAT].to(actions.dtype))
        omega = proprio[BASE_ANGVEL].to(actions.dtype).clone()
        theta = torch.zeros(3, dtype=actions.dtype, device=device)

        blk = proprio[module_block(module_index)].to(actions.dtype)
        twist = blk[1] * j.required_twist
        press = blk[2] * max(j.press_depth, 1e-6)
        pull = blk[3] * j.travel

        inertia = torch.as_tensor(spec.base_inertia, dtype=actions.dtype, device=device)
        force_max = torch.zeros((), dtype=actions.dtype, device=device)
        impulse = torch.zeros((), dtype=actions.dtype, device=device)
        axis_err = torch.zeros((), dtype=actions.dtype, device=device)
        blocked_pull = torch.zeros((), dtype=actions.dtype, device=device)

        for t in range(H):
            axial = torch.dot(dp[t], axis)
            dp_ax = axial * axis
            dp_tan = dp[t] - dp_ax
            tangent = dp_tan.norm()

            # -------------------------------------------------- joint state ----
            if j.kind == "bolt":
                if float(twist) < j.required_twist - spec.twist_tol:
                    twist = twist + torch.relu(torch.dot(dth[t], axis))
                unlocked = float(twist) >= j.required_twist - spec.twist_tol
            elif j.kind == "latch":
                if float(press) < j.press_depth - 1e-9:
                    press = press + torch.relu(-axial)
                unlocked = float(press) >= j.press_depth - 1e-9
            else:
                unlocked = True

            pressing = (
                j.kind == "latch"
                and float(press) < j.press_depth - 1e-9
                and float(axial.detach()) < 0.0
            )

            # ------------------------------------------------ blocked motion ----
            blocked = tangent
            if not unlocked:
                if float(axial.detach()) > 0.0:
                    blocked = blocked + torch.relu(axial)  # cannot pull yet
                    blocked_pull = blocked_pull + torch.relu(axial)
                elif not pressing:
                    blocked = blocked + torch.relu(-axial)  # cannot push into the body
            else:
                if float(axial.detach()) > 0.0:
                    enough = float(spec.k_imp * float(axial.detach())) >= j.breakaway_force
                    if enough:
                        pull = torch.clamp(pull + axial, max=j.travel)
                    else:
                        blocked = blocked + torch.relu(axial)  # not enough force
                elif not pressing:
                    blocked = blocked + torch.relu(-axial)

            force = blocked * spec.k_imp
            force_max = torch.maximum(force_max, force)
            impulse = impulse + force * spec.dt

            # reaction wrench on the free-floating base (Newton's third law)
            # 这里就是“微重力”与地面任务的根本区别——
            #   拧/拔产生的力矩不会消失，而是以反作用力偶加回自由漂浮基座，
            #   累积成姿态漂移；漂移超预算就判“拆下来了但不安全”。
            direction = (dp[t] / dp[t].norm().clamp(min=1e-6)) * force
            f_world = R @ direction
            torque = torch.cross(R @ attach, -f_world, dim=0)
            omega = omega + torque * spec.dt / inertia
            theta = theta + omega * spec.dt

            ee = ee + dp[t]
            axis_err = axis_err + tangent / spec.dp_scale

        detached = float(pull >= j.travel * 0.999)
        capture_dist = (ee - capture).norm() if detached else torch.zeros((), dtype=actions.dtype, device=device)
        return {
            "force_max": force_max,
            "impulse": impulse,
            "attitude_drift": theta.norm(),
            "pull_ratio": pull / j.travel,
            "twist_ratio": twist / max(j.required_twist, 1e-6),
            "press_ratio": press / max(j.press_depth, 1e-6),
            "axis_err": axis_err / max(H, 1),
            "blocked_pull": blocked_pull,
            "approach_dist": (ee - grasp).norm() * 0.0 + (proprio[EE_POS].to(actions.dtype) - grasp).norm(),
            "capture_dist": capture_dist,
            "ee_end": ee,
            "detached": torch.tensor(float(detached), dtype=actions.dtype, device=device),
            "grasped": torch.tensor(float(grasped), dtype=actions.dtype, device=device),
        }


# ----------------------------------------------------------------- energy API ----
class EnergyTerm:
    """Base class: ``__call__(actions, proprio, ctx) -> (cost, info)``."""

    name: str = "term"
    weight: float = 1.0

    def __call__(self, actions: torch.Tensor, proprio: torch.Tensor, ctx: dict):  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.name}, w={self.weight})"


@dataclass
class LatentMatchEnergy(EnergyTerm):
    """DynaGuide's guidance objective, re-expressed as an energy to minimise.

    ``J(a) = logsumexp_i(-||z_hat(a) - z_i^+|| / alpha) - logsumexp_j(-||z_hat(a) - z_j^-|| / alpha)``

    which is exactly ``-log p(g^+ | a) + log p(g^- | a)`` under DynaGuide's Gaussian
    latent assumption, i.e. *minimising* ``J`` moves the predicted end state towards the
    positive set and away from the negative set.
    """

    latent_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
    positives: Optional[torch.Tensor] = None  # (K+, D)
    negatives: Optional[torch.Tensor] = None  # (K-, D)
    alpha: float = 20.0
    mode: str = "logsumexp"  # logsumexp | min | mean
    name: str = "latent"
    weight: float = 1.0

    def __call__(self, actions: torch.Tensor, proprio: torch.Tensor, ctx: dict):
        z_hat = self.latent_predictor(ctx.get("state"), actions).reshape(1, -1)
        info: Dict[str, torch.Tensor] = {}
        cost = torch.zeros((), dtype=actions.dtype, device=actions.device)
        if self.positives is not None:
            d = torch.cdist(self.positives.to(actions.device, actions.dtype), z_hat).squeeze(-1)
            j_pos = self._aggregate(d)
            cost = cost + j_pos
            info["latent_pos"] = j_pos.detach()
        if self.negatives is not None:
            d = torch.cdist(self.negatives.to(actions.device, actions.dtype), z_hat).squeeze(-1)
            j_neg = self._aggregate(d)
            cost = cost - j_neg
            info["latent_neg"] = j_neg.detach()
        return cost, info

    def _aggregate(self, d: torch.Tensor) -> torch.Tensor:
        if self.mode == "min":
            return d.min()
        if self.mode == "mean":
            return d.mean()
        return torch.logsumexp(-d / self.alpha, dim=0)


@dataclass
class SatellitePhysicsEnergy(EnergyTerm):
    """Composite physics-informed energy for the satellite disassembly chunk."""

    spec: SatelliteSpec = field(default_factory=make_sc1_spec)
    weights: Dict[str, float] = field(
        default_factory=lambda: dict(
            protocol=2.0,
            premature_pull=1.5,
            capture=1.0,
            force=1.0,
            momentum=2.0,
            axis=0.25,
            approach=0.5,
        )
    )
    model: Optional[DifferentiableDisassemblyModel] = None
    name: str = "physics"
    weight: float = 1.0

    def __post_init__(self):
        if self.model is None:
            self.model = DifferentiableDisassemblyModel(self.spec)

    def rollout(self, actions, proprio, ctx):
        idx = ctx.get("module_index", 0)
        return self.model.rollout(actions, proprio, idx)

    def __call__(self, actions: torch.Tensor, proprio: torch.Tensor, ctx: dict):
        spec = self.spec
        r = self.rollout(actions, proprio, ctx)
        n_steps = actions.shape[-2]
        terms: Dict[str, torch.Tensor] = {
            # protocol progress: maximise pull_ratio (energy is its negative)
            "protocol": -r["pull_ratio"],
            # pulling before the fastener is released: wasted force, pure momentum injection
            "premature_pull": r["blocked_pull"] / max(spec.dp_scale * n_steps, 1e-6),
            # reaching the staging pose once the module is free
            "capture": r["capture_dist"] / max(spec.capture_tol, 1e-6) * r["detached"],
            # contact force beyond the operational budget
            "force": (r["force_max"] / spec.force_budget) ** 2,
            # momentum: base attitude drift + transferred impulse
            "momentum": (r["attitude_drift"] / spec.attitude_budget) ** 2
            + r["impulse"] / max(spec.force_budget * 0.25, 1e-6),
            # tangential (blocked) motion while the module is fastened
            "axis": r["axis_err"],
            # is the gripper at the requested module at all?
            "approach": r["approach_dist"] / 0.10,
        }
        total = torch.zeros((), dtype=actions.dtype, device=actions.device)
        info: Dict[str, torch.Tensor] = {}
        for key, value in terms.items():
            w = float(self.weights.get(key, 0.0))
            total = total + w * value
            info[key] = value.detach()
        return total, info


@dataclass
class CompositeEnergy(EnergyTerm):
    """Weighted sum of several energy terms."""

    terms: List[EnergyTerm] = field(default_factory=list)
    name: str = "composite"

    def __call__(self, actions, proprio, ctx):
        total = torch.zeros((), dtype=actions.dtype, device=actions.device)
        info: Dict[str, torch.Tensor] = {}
        for term in self.terms:
            cost, sub = term(actions, proprio, ctx)
            total = total + float(term.weight) * cost
            info.update({f"{term.name}/{k}": v for k, v in sub.items()})
            info[f"{term.name}/total"] = (float(term.weight) * cost).detach()
        return total, info

    def add(self, term: EnergyTerm) -> "CompositeEnergy":
        self.terms.append(term)
        return self

