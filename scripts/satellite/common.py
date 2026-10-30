"""Shared paths / helpers for the satellite experiment scripts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "satellite"
CKPT_DIR = REPO_ROOT / "checkpoints" / "satellite"
RESULTS_DIR = REPO_ROOT / "results" / "satellite"

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fmt(value: float, nd: int = 3) -> str:
    return f"{value:.{nd}f}"
