"""Train the latent dynamics model used by the DynaGuide guidance on the satellite task.

This is the satellite counterpart of ``train_dynaguide.py``: it regresses the *future
latent state* from the current latent state and the action chunk, with the noise
augmentation that DynaGuide shows to be essential (the model is queried with not-yet-
denoised action chunks at inference time).

Usage
-----
::

    python scripts/satellite/train_latent_dynamics.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, DATA_DIR, device, set_seed  # noqa: E402

from core.satellite.data import build_chunk_dataset, end_state_bank, load_demos  # noqa: E402
from core.satellite.latent import SemanticDynamicsModel, build_goal_bank  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default=str(DATA_DIR / "demos_mixed.npz"))
    p.add_argument("--out", type=str, default=str(CKPT_DIR / "latent_dynamics.pt"))
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise_prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    dev = device()
    raw = load_demos(args.demos)
    chunks = build_chunk_dataset(raw, horizon=args.horizon, n_obs_steps=2, stride=1)
    obs = chunks["obs"][:, -1, :]  # current proprio
    actions = chunks["actions"]
    obs_end = chunks["obs_end"]
    meta_path = Path(args.demos).with_suffix(".json")
    if meta_path.exists():
        spec_modules = list(json.loads(meta_path.read_text(encoding="utf-8"))["module_names"])
    else:
        spec_modules = [f"module_{i}" for i in range(3)]
    num_modules = len(spec_modules)
    print(f"chunks={len(obs)} latent_dim={10 + 4*num_modules} modules={spec_modules}")

    model = SemanticDynamicsModel(
        latent_dim=10 + 4 * num_modules,
        action_dim=actions.shape[-1],
        horizon=args.horizon,
        hidden=args.hidden,
        num_modules=num_modules,
        noise_prob=args.noise_prob,
    ).to(dev)
    model.encoder.fit(obs.float())
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M")

    obs, actions, obs_end = obs.to(dev), actions.to(dev), obs_end.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    history = []
    n = len(obs)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))
        out = model.loss(obs[idx], actions[idx], obs_end[idx])
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 15) == 0:
            with torch.no_grad():
                val = model.loss(obs, actions, obs_end, augment=False)
            history.append({"step": step, "train": float(out["mse"].item()), "val_clean": float(val["mse"].item())})
            print(f"step {step:5d} train_mse(noisy)={out['mse'].item():.4f} val_mse(clean)={val['mse'].item():.4f}")

    model.eval()
    # guidance conditions: positives = end states of *safe* episodes of the requested
    # module, negatives = end states of episodes that violated the safety budgets.
    goal_banks = {}
    negatives = {}
    for k, name in enumerate(spec_modules):
        bank = end_state_bank(raw, task_id=k, require_ok=True)
        if len(bank) == 0:
            bank = end_state_bank(raw, task_id=k, require_ok=False)
        if len(bank) == 0:
            continue
        goal_banks[name] = build_goal_bank(model, torch.from_numpy(bank).to(dev))

        bad = end_state_bank(raw, task_id=k, require_ok=False)
        if len(bad) == 0:
            other = [
                end_state_bank(raw, task_id=j, require_ok=False)
                for j in range(len(spec_modules))
                if j != k
            ]
            other = [o for o in other if len(o)]
            bad = np.concatenate(other, axis=0) if other else bad
        if len(bad):
            negatives[name] = build_goal_bank(model, torch.from_numpy(bad).to(dev))

    torch.save(
        {
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "num_modules": num_modules,
            "module_names": spec_modules,
            "latent_dim": 10 + 4 * num_modules,
            "horizon": args.horizon,
            "goal_banks": {k: v.cpu() for k, v in goal_banks.items()},
            "negatives": {k: v.cpu() for k, v in negatives.items()},
            "history": history,
        },
        args.out,
    )
    with open(str(args.out).replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump({"steps": args.steps, "history": history, "seconds": time.time() - t0}, f, indent=2)
    print(f"saved {args.out} in {time.time()-t0:.1f}s; goal banks: {list(goal_banks)}")


if __name__ == "__main__":
    main()
