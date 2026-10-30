"""Small 1D conditional UNet used as the velocity network of the flow policies.

This is a dependency-free re-implementation of the ``ConditionalUnet1D`` used by
Diffusion Policy / robomimic (FiLM conditioning on the flow time + global observation
feature, GroupNorm + Mish residual blocks, down/up sampling), kept deliberately small so
that the whole satellite pipeline trains on a CPU.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out), out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """FiLM-conditioned residual block (scale/bias predicted from the condition)."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0] * out + embed[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """(B, T, C) -> (B, T, C) velocity / noise network."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims=(128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ]
        )
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups), nn.Conv1d(start_dim, input_dim, 1)
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: Optional[torch.Tensor] = None):
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([float(t)], dtype=torch.float32, device=sample.device)
        if t.dim() == 0:
            t = t[None]
        t = t.expand(sample.shape[0])
        cond = self.step_encoder(t)
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)

        x = sample.moveaxis(-1, -2)
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)
        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class ObsEncoder(nn.Module):
    """MLP encoder over a stacked observation window -> global conditioning vector."""

    def __init__(self, obs_dim: int, n_obs_steps: int, out_dim: int = 128, hidden: int = 192):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_obs_steps, hidden),
            nn.Mish(),
            nn.Linear(hidden, out_dim),
            nn.Mish(),
        )

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        # obs_window: (B, To, Do)
        return self.net(obs_window.reshape(obs_window.shape[0], -1))
