"""A tiny dependency-free linear normalizer.

Diffusion/flow policies are trained on actions in ``[-1, 1]``; this mirrors the
``LinearNormalizer`` of FlowPolicy and the ``hdf5_normalize_action`` behaviour of
robomimic, but without pulling in their dependencies, so that the standalone satellite
stack stays importable with only ``torch``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


class LinearNormalizer:
    """Per-key min-max scaling to ``[-1, 1]``."""

    def __init__(self, ranges: Optional[Dict[str, torch.Tensor]] = None, eps: float = 1e-6):
        # ranges[k] = tensor of shape (2, *dim) holding (min, max)
        self.ranges: Dict[str, torch.Tensor] = dict(ranges or {})
        self.eps = eps

    # ------------------------------------------------------------------ fit ----
    def fit(self, data: Dict[str, torch.Tensor]) -> "LinearNormalizer":
        for key, value in data.items():
            v = value.reshape(-1, *value.shape[1:])
            lo = v.min(dim=0).values
            hi = v.max(dim=0).values
            self.ranges[key] = torch.stack([lo, hi], dim=0).detach()
        return self

    def fit_one(self, key: str, value: torch.Tensor) -> "LinearNormalizer":
        v = value.reshape(-1, *value.shape[1:])
        lo = v.min(dim=0).values
        hi = v.max(dim=0).values
        self.ranges[key] = torch.stack([lo, hi], dim=0).detach()
        return self

    # -------------------------------------------------------------- scaling ----
    def _unsqueeze_like(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # r: (2, *dim) ; x: (..., *dim)
        while r.dim() - 1 < x.dim() - 1:
            r = r.unsqueeze(1)
        return r

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if key not in self.ranges:
            return x
        r = self._unsqueeze_like(self.ranges[key].to(x.device, x.dtype), x)
        lo, hi = r[0], r[1]
        span = (hi - lo).clamp(min=self.eps)
        return (2.0 * (x - lo) / span - 1.0).clamp(-1.0, 1.0)

    def unnormalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if key not in self.ranges:
            return x
        r = self._unsqueeze_like(self.ranges[key].to(x.device, x.dtype), x)
        lo, hi = r[0], r[1]
        span = (hi - lo).clamp(min=self.eps)
        return (x + 1.0) * 0.5 * span + lo

    # --------------------------------------------------------------- io ----
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.ranges.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> "LinearNormalizer":
        self.ranges = {k: v.clone() for k, v in state.items()}
        return self

    def keys(self) -> Iterable[str]:
        return self.ranges.keys()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LinearNormalizer(keys={list(self.ranges)})"
