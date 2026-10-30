"""Latent dynamics model for guidance on the satellite task.

DynaGuide's dynamics model ``h_theta`` maps ``(phi(o_t), a) -> phi(o_{t+H})`` where ``phi``
is a **frozen DINOv2 patch encoder** of the camera image.  The satellite task is
low-dimensional and has no camera in this simulator, so ``phi`` is replaced by a frozen
*semantic* encoder over the proprioception:

.. code-block:: text

    phi(s) = [ base attitude (3), base angular rate (3), ee position (3), gripper (1),
               per module: attached, twist ratio, press ratio, pull ratio (4 x M) ]

This keeps DynaGuide's structure exactly (a frozen encoder + a transformer/MLP that
regresses the future latent + a Gaussian latent assumption for the guidance metric) while
remaining trainable in seconds on a CPU.  The encoder is standardised per dimension so
that the ``log-sum-exp(-d / alpha)`` metric of Eq. 2 has a well defined temperature, just
like the DINOv2 embeddings do.

The critical DynaGuide detail that *must* be reproduced is the **noise augmentation**: the
dynamics model is queried with not-yet-denosed action chunks during guidance, so it has to
be trained on noisy actions drawn from the same interpolant as the sampler.

.. code-block:: text

    a_t = (1 - t) * eps + t * a1 ,  t ~ U[0, 1]

applied to 50% of the batch (DynaGuide: "50% of the actions are noised, sampled from a
geometric distribution over the noise scheduler").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from core.satellite.env import BASE_ANGVEL, EE_POS, GRIP, module_block


@dataclass
class SemanticLatentEncoder:
    """Frozen ``phi``: proprioception -> standardised semantic latent.

    Standardisation uses a **fixed nominal scale per feature** (one unit = one task
    tolerance) rather than the empirical standard deviation: several features are binary
    (``attached``) or constant in the observation distribution (``press`` for non-latch
    modules), and a near-zero empirical std would blow the goal latent up by orders of
    magnitude.  The metric ``||z - z_goal||`` therefore stays interpretable and the
    temperature ``alpha`` of the log-sum-exp has a stable meaning.
    """

    num_modules: int = 3
    mean: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None

    @property
    def dim(self) -> int:
        return 10 + 4 * self.num_modules

    def default_scale(self, device=None) -> torch.Tensor:
        return torch.tensor(
            [0.15] * 3  # base attitude [rad]
            + [0.15] * 3  # base angular rate [rad/s]
            + [0.20] * 3  # end effector position [m]
            + [1.0]  # gripper
            + [1.0, 1.0, 1.0, 1.0] * self.num_modules,
            dtype=torch.float32,
            device=device,
        )

    def raw(self, proprio: torch.Tensor) -> torch.Tensor:
        """(..., D) proprio -> (..., latent_dim) unnormalised features.

        The base attitude enters as ``2 * quat_vector`` which equals the rotation vector to
        first order for the small drifts this task cares about; using one single code path
        for single samples and batches keeps the guidance latent **identical** whether the
        state comes from the environment or from a `(B, D)` training batch.
        """
        if isinstance(proprio, dict):
            proprio = proprio["proprio"]
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        attitude = proprio[..., 4:7] * 2.0
        feats = [attitude, proprio[..., BASE_ANGVEL], proprio[..., EE_POS], proprio[..., GRIP : GRIP + 1]]
        for i in range(self.num_modules):
            blk = proprio[..., module_block(i)]
            feats.append(blk[..., 0:4])
        return torch.cat(feats, dim=-1)

    def fit(self, proprio: torch.Tensor) -> "SemanticLatentEncoder":
        raw = self.raw(proprio.reshape(-1, proprio.shape[-1]))
        self.mean = raw.mean(dim=0, keepdim=True)
        self.scale = self.default_scale(raw.device)
        return self

    def __call__(self, proprio: torch.Tensor) -> torch.Tensor:
        raw = self.raw(proprio)
        if self.mean is None:
            return raw
        mean = self.mean.to(raw.device, raw.dtype)
        scale = (self.scale if self.scale is not None else self.default_scale(raw.device)).to(raw.device, raw.dtype)
        return (raw - mean) / scale

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "scale": self.scale}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.mean = state["mean"]
        self.scale = state.get("scale", None)
        return self


class SemanticDynamicsModel(nn.Module):
    """``h_theta``: (latent_t, action chunk) -> latent_{t+H}."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 7,
        horizon: int = 16,
        hidden: int = 256,
        n_layers: int = 3,
        num_modules: int = 3,
        noise_augment: bool = True,
        noise_prob: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.noise_augment = noise_augment
        self.noise_prob = noise_prob

        self.encoder = SemanticLatentEncoder(num_modules=num_modules)
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim * horizon, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
        )
        self.z_embed = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Mish())
        trunk = []
        for _ in range(n_layers):
            trunk += [nn.Linear(hidden * 2 if not trunk else hidden, hidden), nn.Mish()]
        self.trunk = nn.Sequential(*trunk)
        self.head = nn.Linear(hidden, latent_dim)

    # ------------------------------------------------------------- encoder ----
    def state_embedding(self, proprio: torch.Tensor) -> torch.Tensor:
        """``phi(s)`` -- the frozen latent used as the guidance target."""
        return self.encoder(proprio)

    def state_action_embedding(self, state, actions: torch.Tensor) -> torch.Tensor:
        """``h_theta(phi(s), a)`` -- the DynaGuide API, so the energies are drop-in."""
        z = self.state_embedding(state)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        a = actions.reshape(actions.shape[0], -1)
        h = self.trunk(torch.cat([self.z_embed(z), self.action_embed(a)], dim=-1))
        return self.head(h)

    def forward(self, proprio: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.state_action_embedding(proprio, actions)

    # ---------------------------------------------------------- augmentation ----
    def augment_actions(self, actions: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add interpolant noise ``a_t = (1-t) eps + t a`` to a fraction of the batch."""
        if not self.noise_augment:
            return actions
        B = actions.shape[0]
        mask = torch.rand(B, device=actions.device, generator=generator) < self.noise_prob
        if not bool(mask.any()):
            return actions
        t = torch.rand(B, 1, 1, device=actions.device, generator=generator)
        eps = torch.randn_like(actions)
        noisy = (1.0 - t) * eps + t * actions
        out = actions.clone()
        out[mask] = noisy[mask]
        return out

    def loss(
        self,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        proprio_end: torch.Tensor,
        augment: bool = True,
    ) -> Dict[str, torch.Tensor]:
        noisy_actions = self.augment_actions(actions) if augment else actions
        z_hat = self.state_action_embedding(proprio, noisy_actions)
        z_end = self.state_embedding(proprio_end)
        mse = torch.mean((z_hat - z_end) ** 2)
        return {"loss": mse, "mse": mse.detach(), "z_hat": z_hat, "z_end": z_end}


# ------------------------------------------------------------------ goal bank ----
def build_goal_bank(
    model: SemanticDynamicsModel,
    end_states: torch.Tensor,
    max_examples: Optional[int] = None,
) -> torch.Tensor:
    """Embed a bank of successful end states into the guidance latent space."""
    with torch.no_grad():
        z = model.state_embedding(end_states)
    if max_examples is not None:
        z = z[:max_examples]
    return z.detach()
