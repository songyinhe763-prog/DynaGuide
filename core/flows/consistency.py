"""Training objectives for flow-based action-chunk policies.

Two objectives are provided:

``rectified_flow_loss``
    Plain (conditional) flow matching objective of the linear path::

        L = E_{t ~ U[eps, 1], eps ~ N(0, I)} || v_theta(x_t, t) - (x_1 - eps) ||^2

    This is the objective used by rectified-flow / Flow-Matching action-chunking
    policies and is the recommended default: it is stable, cheap, and its velocity field
    is exactly the object the guided ODE sampler integrates.

``ConsistencyFlowMatcher`` / ``consistency_flow_loss``
    FlowPolicy's *consistency* flow matching objective (Zhang et al., AAAI 2025 oral).
    The [0, 1] time interval is split into ``num_segments`` segments; the network is
    asked to (i) map any point of a segment to the end of that segment by a forward
    euler step and (ii) have a self-consistent velocity inside the segment.  This makes
    few-step (down to 1-step) sampling possible.  The implementation follows
    ``flow_policy_3d/policy/flowpolicy.py::compute_loss`` of the reference repository,
    with the boolean masks converted to float masks and the model call factored out.

Both losses act on **normalised** actions in ``[-1, 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch

from core.flows.paths import LinearFlowPath, pad_t_like_x

VelocityFn = Callable[..., torch.Tensor]


def _reduce(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x.reshape(x.shape[0], -1), dim=-1)


def _expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """(B,) or (B,1,1) time -> full ``x`` shape, without autograd surprises."""
    t = pad_t_like_x(t, x)
    return t.expand_as(x)


def rectified_flow_loss(
    velocity_fn: VelocityFn,
    actions: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    path: Optional[LinearFlowPath] = None,
    t_min: float = 1e-2,
    condition_mask: Optional[torch.Tensor] = None,
    condition_data: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Plain flow-matching loss. ``actions`` are the normalised target chunks ``x1``."""
    path = path or LinearFlowPath()
    B = actions.shape[0]
    device = actions.device
    t = torch.rand(B, device=device, dtype=actions.dtype) * (1.0 - t_min) + t_min
    x1 = actions
    x0 = torch.randn_like(x1)
    x_t = path.mu_t(x0, x1, t)
    u_t = path.target_velocity(x0, x1)
    if condition_mask is not None and condition_data is not None:
        x_t = x_t.clone()
        x_t[condition_mask] = condition_data[condition_mask]
    v = velocity_fn(x_t, t, cond)
    if condition_mask is not None and condition_data is not None:
        v = v.clone()
        v[condition_mask] = condition_data[condition_mask]
    per_sample = torch.square(v - u_t).mean(dim=tuple(range(1, u_t.dim())))
    loss = per_sample.mean()
    info = {
        "bc_loss": loss.item(),
        "fm_loss": loss.item(),
        "t_mean": t.mean().item(),
    }
    return loss, info


@dataclass
class ConsistencyFlowMatcher:
    """FlowPolicy's consistency flow matching objective.

    Parameters
    ----------
    eps:
        Lower bound of the sampled time (the ODE starts at ``eps``; FlowPolicy uses
        ``1e-2``).
    num_segments:
        Number of consistency segments over ``[0, 1]`` (FlowPolicy default ``2``).
    boundary:
        Threshold below which the "map to segment end" operator is used instead of the
        single-step euler update; ``1`` means "always euler" (FlowPolicy's value).
    delta:
        Extra time step used for the velocity-consistency term, and the margin that
        disables the velocity loss close to a segment boundary.
    alpha:
        Weight of the velocity consistency term relative to the position term.
    """

    eps: float = 1e-2
    num_segments: int = 2
    boundary: float = 1.0
    delta: float = 1e-2
    alpha: float = 1e-5
    path: LinearFlowPath = field(default_factory=LinearFlowPath)

    # ------------------------------------------------------------------ ops ----
    # 这个“段终点外推”就是 FlowPolicy 论文里的 f_θ^i，
    #   也是我们引导时用的“干净动作估计”的同一个算子（段终点取 1 时两者完全一致）。
    @staticmethod
    def f_euler(
        t_expand: torch.Tensor,
        segment_ends_expand: torch.Tensor,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """One forward euler step from ``t`` to the end of its segment."""
        return x_t + (segment_ends_expand - t_expand) * v_t

    @classmethod
    def threshold_based_f_euler(
        cls,
        t_expand: torch.Tensor,
        segment_ends_expand: torch.Tensor,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        threshold: float,
        x_at_segment_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Euler step far from the boundary, exact segment end close to it."""
        if isinstance(threshold, (int, float)) and threshold == 0:
            return x_at_segment_ends
        below = (t_expand < threshold).float()
        euler = cls.f_euler(t_expand, segment_ends_expand, x_t, v_t)
        return below * euler + (1.0 - below) * x_at_segment_ends

    def segment_boundaries(self, t: torch.Tensor) -> torch.Tensor:
        """Right end of the segment each sample in ``t`` belongs to."""
        segments = torch.linspace(0.0, 1.0, self.num_segments + 1, device=t.device, dtype=t.dtype)
        idx = torch.searchsorted(segments, t.contiguous(), side="left").clamp(min=1)
        return segments[idx]

    # --------------------------------------------------------------- loss ----
    def loss(
        self,
        velocity_fn: VelocityFn,
        actions: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        condition_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Consistency flow matching loss (position + velocity consistency)."""
        x1 = actions
        B = x1.shape[0]
        device, dtype = x1.device, x1.dtype

        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device, dtype=dtype) * (1.0 - self.eps) + self.eps
        r = torch.clamp(t + self.delta, max=1.0)
        t_expand = _expand_time(t, x1)
        r_expand = _expand_time(r, x1)

        x_t = t_expand * x1 + (1.0 - t_expand) * x0
        x_r = r_expand * x1 + (1.0 - r_expand) * x0
        if condition_mask is not None and condition_data is not None:
            x_t = x_t.clone()
            x_r = x_r.clone()
            x_t[condition_mask] = condition_data[condition_mask]
            x_r[condition_mask] = condition_data[condition_mask]

        segment_ends = self.segment_boundaries(t)
        segment_ends_expand = _expand_time(segment_ends, x1)
        x_at_segment_ends = segment_ends_expand * x1 + (1.0 - segment_ends_expand) * x0

        v_t = velocity_fn(x_t, t, cond)
        v_r = velocity_fn(x_r, r, cond)
        if condition_mask is not None and condition_data is not None:
            v_t = v_t.clone()
            v_r = v_r.clone()
            v_t[condition_mask] = condition_data[condition_mask]
            v_r[condition_mask] = condition_data[condition_mask]
        v_r = torch.nan_to_num(v_r)

        f_t = self.f_euler(t_expand, segment_ends_expand, x_t, v_t)
        f_r = self.threshold_based_f_euler(
            r_expand, segment_ends_expand, x_r, v_r, self.boundary, x_at_segment_ends
        )

        losses_f = _reduce(torch.square(f_t - f_r))

        # velocity self-consistency, disabled near segment boundaries and above threshold
        if self.boundary == 0:
            losses_v = torch.zeros_like(losses_f)
        else:
            below = (t_expand < self.boundary).float()
            far = ((segment_ends - t) > 1.01 * self.delta).float()
            far = far.reshape(B, 1, 1).expand_as(x1)
            losses_v = _reduce(below * far * torch.square(v_t - v_r))

        loss = torch.mean(losses_f + self.alpha * losses_v)
        info = {
            "bc_loss": loss.item(),
            "consistency_pos_loss": losses_f.mean().item(),
            "consistency_vel_loss": losses_v.mean().item(),
        }
        return loss, info


def consistency_flow_loss(
    velocity_fn: VelocityFn,
    actions: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, dict]:
    """Convenience wrapper around :class:`ConsistencyFlowMatcher` with FlowPolicy defaults."""
    cfg = ConsistencyFlowMatcher(
        eps=kwargs.pop("eps", 1e-2),
        num_segments=kwargs.pop("num_segments", 2),
        boundary=kwargs.pop("boundary", 1),
        delta=kwargs.pop("delta", 1e-2),
        alpha=kwargs.pop("alpha", 1e-5),
    )
    return cfg.loss(velocity_fn, actions, cond, **kwargs)
