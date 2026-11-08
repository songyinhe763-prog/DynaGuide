# ═══════════════════════════════════════════════════════════════════════════
# ★ 讲解要点（本文件 = 被引导的“基础策略”本体）
#   结构很简单，讲的时候按三块走：
#     ① encode(观测窗口)  → 观测嵌入（相当于 Diffusion Policy 的条件向量）
#     ② velocity(带噪动作, t, 条件) → 速度场 v_θ，这是网络唯一的输出
#     ③ sample() 调 FlowSampler 积分出动作块，引导以参数形式传进去
#   关键设计：
#     · 策略本身对“要拆哪个模块、用什么手法”一无所知（无条件），
#       所以它会学成三种示范的平均 —— 这正是需要推理期引导的原因；
#     · 动作在训练前归一化到 [-1,1]（LinearNormalizer），
#       采样后再反归一化，仿真里看到的是真实尺度。
# ═══════════════════════════════════════════════════════════════════════════
"""A standalone flow-matching action-chunk policy (the "base policy" of the pipeline).

The policy maps an observation window to a *velocity field* over action chunks and is
trained with either

* ``objective="fm"``   rectified flow matching (default, cheap and stable), or
* ``objective="cfm"``  FlowPolicy's consistency flow matching (few-step / 1-step).

It is the CPU-friendly sibling of ``robomimic/robomimic/algo/flow_policy.py`` and shares
its runtime objects (``LinearFlowPath``, ``FlowSampler``, ``FlowDynaGuide``), so results
transfer 1:1 to the robomimic integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from core.flows.consistency import ConsistencyFlowMatcher, rectified_flow_loss
from core.flows.nets import ConditionalUnet1D, ObsEncoder
from core.flows.normalizer import LinearNormalizer
from core.flows.paths import LinearFlowPath
from core.flows.samplers import FlowSampler, FlowSamplerConfig


@dataclass
class FlowPolicyConfig:
    obs_dim: int = 36
    action_dim: int = 7
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    encoder_hidden: int = 192
    encoder_out: int = 128
    diffusion_step_embed_dim: int = 128
    down_dims: tuple = (128, 256)
    kernel_size: int = 5
    n_groups: int = 8
    objective: str = "fm"  # fm | cfm
    t_min: float = 1e-2
    time_scale: float = 1.0
    cfm_kwargs: dict = field(
        default_factory=lambda: dict(eps=1e-2, num_segments=2, boundary=1.0, delta=1e-2, alpha=1e-5)
    )


class FlowActionChunkPolicy(nn.Module):
    """Velocity-field policy with a DynaGuide-compatible sampling interface."""

    def __init__(self, config: FlowPolicyConfig, normalizer: Optional[LinearNormalizer] = None):
        super().__init__()
        self.config = config
        self.path = LinearFlowPath()
        self.normalizer = normalizer or LinearNormalizer()
        self.obs_encoder = ObsEncoder(config.obs_dim, config.n_obs_steps, config.encoder_out, config.encoder_hidden)
        self.velocity_net = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.encoder_out,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
        )
        self.cfm = ConsistencyFlowMatcher(**config.cfm_kwargs)

    # ------------------------------------------------------------- forward ----
    def encode(self, obs_window: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(obs_window)

    def velocity(self, x: torch.Tensor, t: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(x, t * self.config.time_scale, obs_cond)

    def velocity_fn(self, obs_cond: torch.Tensor) -> Callable:
        def fn(x, t):
            return self.velocity(x, t, obs_cond)

        return fn

    # --------------------------------------------------------------- loss ----
    def loss(self, batch: Dict[str, torch.Tensor]) -> tuple:
        obs_window = batch["obs"]  # (B, To, Do)
        actions = batch["actions"]  # (B, T, Da) normalised
        obs_cond = self.encode(obs_window)

        def velocity_fn(x_t, t, cond):
            return self.velocity(x_t, t, cond)

        if self.config.objective == "cfm":
            return self.cfm.loss(velocity_fn, actions, obs_cond)
        return rectified_flow_loss(velocity_fn, actions, obs_cond, t_min=self.config.t_min)

    # ------------------------------------------------------------- sampling ----
    def sample(
        self,
        obs_window: torch.Tensor,
        sampler: Optional[FlowSampler] = None,
        guidance: Optional[Callable] = None,
        ctx: Optional[dict] = None,
        record: bool = False,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        cfg = sampler.config if sampler is not None else FlowSamplerConfig(
            n_steps=4, method="euler", t_start=1e-2, t_end=1.0, time_scale=self.config.time_scale
        )
        sampler = sampler or FlowSampler(cfg)
        obs_cond = self.encode(obs_window)
        B = obs_window.shape[0] * n_samples
        if n_samples > 1:
            obs_cond = obs_cond.repeat_interleave(n_samples, dim=0)
        x0 = torch.randn(B, self.config.horizon, self.config.action_dim, device=obs_window.device)
        out = sampler.sample(
            lambda x, t, cond: self.velocity(x, t, cond), x0, cond=obs_cond, guidance=guidance, ctx=ctx, record=record
        )
        naction = out["actions"]
        action = self.normalizer.unnormalize("action", naction)
        return {"actions": action, "n_action": naction, "trace": out.get("trace"), "nfe": out["nfe"]}

    # ------------------------------------------------------------- helpers ----
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "normalizer": self.normalizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "FlowActionChunkPolicy":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        policy = cls(blob["config"], normalizer=LinearNormalizer(blob.get("normalizer", {})))
        policy.load_state_dict(blob["state_dict"])
        policy.eval()
        return policy
