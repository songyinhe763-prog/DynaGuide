"""Train the flow-matching base policy on the mixed satellite demonstrations.

The policy is *unconditional* on the objective on purpose: exactly like DynaGuide's base
diffusion policy, it learns the mixture of all demonstrated protocols and the guidance is
what selects the requested module at inference time.

Usage
-----
::

    python scripts/satellite/train_flow_policy.py --steps 2000 --objective fm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, DATA_DIR, device, set_seed  # noqa: E402

from core.flows.normalizer import LinearNormalizer  # noqa: E402
from core.flows.policy import FlowActionChunkPolicy, FlowPolicyConfig  # noqa: E402
from core.satellite.data import ChunkDataset, build_chunk_dataset, load_demos, split_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default=str(DATA_DIR / "demos_mixed.npz"))
    p.add_argument("--out", type=str, default=str(CKPT_DIR / "flow_policy.pt"))
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--objective", type=str, default="fm", choices=["fm", "cfm"])
    p.add_argument("--down_dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time_scale", type=float, default=1.0)
    args = p.parse_args()

    set_seed(args.seed)
    dev = device()

    raw = load_demos(args.demos)
    ds = ChunkDataset(
        build_chunk_dataset(raw, horizon=args.horizon, n_obs_steps=args.n_obs_steps, stride=1)
    )
    train_ds, val_ds = split_dataset(ds, 0.15, seed=args.seed)
    print(f"chunks: train={len(train_ds)} val={len(val_ds)}  obs_dim={train_ds.tensors['obs'].shape[-1]}")

    normalizer = LinearNormalizer().fit_one("action", train_ds.tensors["actions"])
    cfg = FlowPolicyConfig(
        obs_dim=train_ds.tensors["obs"].shape[-1],
        action_dim=train_ds.tensors["actions"].shape[-1],
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        down_dims=tuple(args.down_dims),
        objective=args.objective,
        time_scale=args.time_scale,
    )
    policy = FlowActionChunkPolicy(cfg, normalizer=normalizer).to(dev)
    print(f"params: {policy.trainable_parameters()/1e6:.2f}M  objective={args.objective}")

    def to_batch(idx):
        batch = {k: v.to(dev) for k, v in train_ds.tensors.items()}
        return {
            "obs": batch["obs"][idx],
            "actions": normalizer.normalize("action", batch["actions"][idx]),
        }

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    n = len(train_ds)
    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))
        batch = to_batch(idx)
        loss, info = policy.loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 20) == 0:
            val = {
                "obs": val_ds.tensors["obs"].to(dev),
                "actions": normalizer.normalize("action", val_ds.tensors["actions"].to(dev)),
            }
            with torch.no_grad():
                vloss, vinfo = policy.loss(val)
            history.append({"step": step, "train": float(loss.item()), "val": float(vloss.item())})
            print(f"step {step:5d} train={loss.item():.5f} val={vloss.item():.5f}")

    policy.eval()
    policy.save(args.out)
    blob = {
        "steps": args.steps,
        "objective": args.objective,
        "params": policy.trainable_parameters(),
        "history": history,
        "seconds": time.time() - t0,
        "demos": args.demos,
    }
    with open(str(args.out).replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    print(f"saved {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
