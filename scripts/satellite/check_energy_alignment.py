"""Diagnostic: is the differentiable physics energy aligned with the simulator?

For a set of states along a base-policy rollout we take the policy's own action chunk,
descend the *energy* on the chunk with a few gradient steps, and then execute both the
original and the corrected chunk open-loop in two clones of the environment.  If the
surrogate model is faithful, the corrected chunk must produce less base drift / lower
contact force in the real simulation.

Usage
-----
::

    python scripts/satellite/check_energy_alignment.py --episodes 4 --grad_steps 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, device, set_seed  # noqa: E402

from core.flow_energies import SatellitePhysicsEnergy  # noqa: E402
from core.flows.policy import FlowActionChunkPolicy  # noqa: E402
from core.flows.samplers import FlowSampler, FlowSamplerConfig  # noqa: E402
from core.satellite.env import SatelliteEnv  # noqa: E402
from core.satellite.spec import make_sc1_spec  # noqa: E402


def execute(env, actions, n):
    """Execute ``n`` actions open-loop and return the accumulated metrics."""
    for a in actions[:n]:
        obs, _r, done, info = env.step(a)
        if done:
            break
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, default=str(CKPT_DIR / "flow_policy_payload.pt"))
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--grad_steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--csv", type=str, default="", help="optional path to dump the per-sample rows")
    args = p.parse_args()
    set_seed(0)
    dev = device()
    spec = make_sc1_spec()
    task = "payload_box"
    idx = spec.module_names.index(task)
    pol = FlowActionChunkPolicy.load(args.policy, dev=dev) if False else FlowActionChunkPolicy.load(args.policy)
    pol.to(dev).eval()
    sampler = FlowSampler(FlowSamplerConfig(n_steps=4, method="euler", t_start=1e-2, t_end=1.0, time_scale=pol.config.time_scale))
    energy = SatellitePhysicsEnergy(spec=spec)

    rows = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        env = SatelliteEnv(spec=spec, seed=seed, randomize=True)
        obs = env.reset(task=task, seed=seed)
        hist = [obs["proprio"].copy()]
        for replan in range(6):
            w = np.stack(hist[-2:])
            if w.shape[0] < 2:
                w = np.concatenate([w[:1], w])
            with torch.no_grad():
                chunk = pol.sample(torch.from_numpy(w[None]).float().to(dev), sampler=sampler)["actions"][0].cpu()
            proprio = torch.from_numpy(obs["proprio"]).float()
            ctx = {"module_index": idx}

            # descend the surrogate energy on the chunk
            a = chunk.clone().requires_grad_(True)
            for _ in range(args.grad_steps):
                cost, info_e = energy(a.unsqueeze(0), proprio, ctx)
                g = torch.autograd.grad(cost, a)[0]
                a = (a - args.lr * g / (g.norm() + 1e-8) * 2.0).detach().clamp(-1, 1).requires_grad_(True)

            env_a, env_b = env.clone(), env.clone()
            base_info = execute(env_a, chunk.numpy(), pol.config.n_action_steps)
            guided_info = execute(env_b, a.detach().numpy(), pol.config.n_action_steps)
            rows.append(
                {
                    "ep": ep,
                    "replan": replan,
                    "energy_before": float(energy(chunk.unsqueeze(0), proprio, ctx)[0]),
                    "energy_after": float(energy(a.detach().unsqueeze(0), proprio, ctx)[0]),
                    "drift_base": base_info.get("attitude_drift", 0.0),
                    "drift_guided": guided_info.get("attitude_drift", 0.0),
                    "force_base": base_info.get("max_force", 0.0),
                    "force_guided": guided_info.get("max_force", 0.0),
                }
            )
            # advance the real rollout with the *base* chunk
            for act in chunk.numpy()[: pol.config.n_action_steps]:
                obs, _r, done, _i = env.step(act)
                hist.append(obs["proprio"].copy())
                if done:
                    break
            if done:
                break

    import statistics

    def mean(k):
        return statistics.mean(r[k] for r in rows)

    better_drift = sum(r["drift_guided"] <= r["drift_base"] + 1e-9 for r in rows)
    better_force = sum(r["force_guided"] <= r["force_base"] + 1e-9 for r in rows)
    print(f"samples: {len(rows)}")
    print(f"mean energy   before={mean('energy_before'):.3f} after={mean('energy_after'):.3f} "
          f"(lower is better, surrogate decrease = {100*(1-mean('energy_after')/max(mean('energy_before'),1e-9)):.1f}%)")
    print(f"mean drift    base={mean('drift_base'):.4f} guided={mean('drift_guided'):.4f}  improved in {better_drift}/{len(rows)}")
    print(f"mean force    base={mean('force_base'):.3f} guided={mean('force_guided'):.3f}  improved in {better_force}/{len(rows)}")
    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"per-sample rows -> {args.csv}")


if __name__ == "__main__":
    main()
