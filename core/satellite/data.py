"""Episode / chunk dataset utilities for the satellite experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------- io ----
def save_demos(path: str, data: Dict) -> None:
    """Save a demonstration dict to ``.npz`` (arrays only, metadata as attributes)."""
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)


def load_demos(path: str) -> Dict:
    blob = np.load(path, allow_pickle=False)
    return {k: blob[k] for k in blob.files}


# ---------------------------------------------------------------- chunking ----
def build_chunk_dataset(
    data: Dict,
    horizon: int = 16,
    n_obs_steps: int = 2,
    stride: int = 1,
    task_filter: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Slice episodes into (observation window, action chunk, end observation) tuples."""
    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    task_id = np.asarray(data.get("task_id", np.zeros(len(actions))), dtype=np.int64)

    obs_windows, act_chunks, obs_ends, tasks, eps = [], [], [], [], []
    for ep in np.unique(episode_id):
        idx = np.nonzero(episode_id == ep)[0]
        if len(idx) < horizon:
            continue
        e_obs = obs[idx]
        e_act = actions[idx]
        e_task = task_id[idx]
        L = len(idx)
        for t in range(0, L - horizon, stride):
            if task_filter is not None and int(e_task[t]) not in task_filter:
                continue
            start = max(0, t - n_obs_steps + 1)
            window = e_obs[start : t + 1]
            if window.shape[0] < n_obs_steps:
                pad = np.repeat(window[:1], n_obs_steps - window.shape[0], axis=0)
                window = np.concatenate([pad, window], axis=0)
            obs_windows.append(window)
            act_chunks.append(e_act[t : t + horizon])
            obs_ends.append(e_obs[t + horizon])
            tasks.append(e_task[t])
            eps.append(ep)

    if not obs_windows:
        raise ValueError("no chunks produced - check horizon / dataset length")

    return {
        "obs": torch.from_numpy(np.stack(obs_windows)),
        "actions": torch.from_numpy(np.stack(act_chunks)),
        "obs_end": torch.from_numpy(np.stack(obs_ends)),
        "task_id": torch.from_numpy(np.asarray(tasks, dtype=np.int64)),
        "episode_id": torch.from_numpy(np.asarray(eps, dtype=np.int64)),
    }


@dataclass
class ChunkDataset(Dataset):
    """Thin torch ``Dataset`` over the dict produced by :func:`build_chunk_dataset`."""

    tensors: Dict[str, torch.Tensor]

    def __post_init__(self):
        n = len(self.tensors["obs"])
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.tensors.items()}

    def subset(self, mask: torch.Tensor) -> "ChunkDataset":
        return ChunkDataset({k: v[mask] for k, v in self.tensors.items()})


def split_dataset(ds: ChunkDataset, val_fraction: float = 0.15, seed: int = 0) -> tuple:
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = ds.subset(torch.zeros(n, dtype=torch.bool).scatter_(0, train_idx, True))
    val = ds.subset(torch.zeros(n, dtype=torch.bool).scatter_(0, val_idx, True))
    return train, val


def end_state_bank(
    data: Dict,
    task_id: Optional[int] = None,
    require_ok: bool = True,
    behavior: Optional[int] = None,
) -> np.ndarray:
    """End-of-episode observations used as guidance conditions.

    Parameters
    ----------
    task_id:
        keep only episodes that finished on this module (``None`` = any).
    require_ok:
        keep only episodes that satisfied the safety budgets -- the *positive* bank.
        Pass ``False`` for a bank of unsafe episodes (the *negative* conditions).
    behavior:
        keep only episodes of one demonstration strategy (0 safe / 1 impatient / 2 offset).
    """
    obs = np.asarray(data["obs"], dtype=np.float32)
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    task_id_arr = np.asarray(data.get("task_id", np.zeros(len(obs))), dtype=np.int64)
    ep_ok = np.asarray(data.get("ep_ok", np.ones(len(obs))), dtype=np.float32)
    beh = np.asarray(data.get("behavior", np.zeros(len(obs))), dtype=np.int64)
    ends = []
    for ep in np.unique(episode_id):
        idx = np.nonzero(episode_id == ep)[0]
        last = idx[-1]
        if task_id is not None and int(task_id_arr[last]) != int(task_id):
            continue
        if require_ok and float(ep_ok[last]) < 0.5:
            continue
        if behavior is not None and int(beh[last]) != int(behavior):
            continue
        ends.append(obs[last])
    if not ends:
        return np.zeros((0, obs.shape[1]), dtype=np.float32)
    return np.stack(ends).astype(np.float32)
