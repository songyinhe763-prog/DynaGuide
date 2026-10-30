"""Evaluate the flow policy with and without DynaGuide guidance on the satellite task.

Methods
-------
``base``    the frozen flow policy (behaviour cloning on the mixed protocols)
``latent``  DynaGuide guidance: latent distance to the goal bank of the requested module
``physics`` the analytic, momentum-aware disassembly energy
``both``    the weighted sum of the two energies
``reject``  rejection sampling baseline (draw N chunks, keep the lowest-energy one)

Metrics
-------
``success``
    the requested module was detached and staged *and* the episode did not abort.
``constraint``
    success **plus** the base attitude drift and the contact force stayed inside budget.
    The on-orbit literature insists on this fourth axis: "the policy succeeded but threw
    the satellite away" must not be counted as a success.
``drift`` / ``force``
    the worst base attitude drift [rad] and contact force [N] seen in the episode.

Usage
-----
::

    python scripts/satellite/run_flow_dynaguide.py --episodes 8 \
        --methods base latent physics both reject --out results/satellite/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, DATA_DIR, RESULTS_DIR, device, set_seed  # noqa: E402

from core.flow_dynaguide import make_flow_guidance  # noqa: E402
from core.flow_energies import CompositeEnergy, LatentMatchEnergy, SatellitePhysicsEnergy  # noqa: E402
from core.flows.policy import FlowActionChunkPolicy  # noqa: E402
from core.flows.samplers import FlowSampler, FlowSamplerConfig  # noqa: E402
from core.satellite.env import SatelliteEnv  # noqa: E402
from core.satellite.latent import SemanticDynamicsModel  # noqa: E402
from core.satellite.spec import make_sc1_spec  # noqa: E402

METHODS = ("base", "latent", "physics", "both", "reject")


# ----------------------------------------------------------------- loading ----
def load_policy(path: str, dev: str) -> FlowActionChunkPolicy:
    return FlowActionChunkPolicy.load(path, map_location=dev).to(dev)


def load_dynamics(path: str, dev: str):
    blob = torch.load(path, map_location=dev, weights_only=False)
    model = SemanticDynamicsModel(
        latent_dim=blob["latent_dim"], action_dim=7, horizon=blob["horizon"], num_modules=blob["num_modules"]
    ).to(dev)
    model.load_state_dict(blob["model"])
    model.encoder.load_state_dict(blob["encoder"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    banks = {k: v.to(dev) for k, v in blob["goal_banks"].items()}
    negatives = {k: v.to(dev) for k, v in blob.get("negatives", {}).items()}
    return model, banks, negatives, blob


# ---------------------------------------------------------------- energy ----
def build_energy(method: str, spec, dynamics, banks, negatives, task_name: str, alpha: float):
    terms = []
    if method in ("latent", "both", "reject-latent"):
        bank = banks.get(task_name)
        neg = negatives.get(task_name)
        if bank is not None:
            terms.append(
                LatentMatchEnergy(
                    latent_predictor=dynamics.state_action_embedding,
                    positives=bank,
                    negatives=neg,
                    alpha=alpha,
                    name="latent",
                    weight=1.0,
                )
            )
    if method in ("physics", "both", "reject-physics"):
        terms.append(SatellitePhysicsEnergy(spec=spec, name="physics", weight=1.0))
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return CompositeEnergy(terms=terms)


# --------------------------------------------------------------- rollout ----
def rollout_episode(
    env: SatelliteEnv,
    policy: FlowActionChunkPolicy,
    task_name: str,
    sampler: FlowSampler,
    method: str,
    spec,
    dynamics=None,
    banks=None,
    negatives=None,
    scale: float = 1.0,
    weight_mode: str = "fp",
    alpha: float = 30.0,
    n_samples: int = 16,
    grad_clip: float = 5.0,
    ss: int = 1,
    seed: int = 0,
) -> Dict:
    dev = next(policy.parameters()).device
    obs = env.reset(task=task_name, seed=seed)
    cfg = policy.config
    history: List[np.ndarray] = [obs["proprio"].copy()]
    module_index = env.spec.module_names.index(task_name)

    energy = build_energy(
        {"reject": "both"}.get(method, method), spec, dynamics, banks, negatives, task_name, alpha
    )
    modality = "reject" if method == "reject" else ("gradient" if energy is not None else "none")

    guide = None
    if modality == "gradient" and energy is not None:
        guide = make_flow_guidance(
            energy, scale=scale, weight_mode=weight_mode, mode="endpoint", clip_norm=grad_clip
        )

    sampler = FlowSampler(
        FlowSamplerConfig(
            n_steps=sampler.config.n_steps,
            method=sampler.config.method,
            t_start=sampler.config.t_start,
            t_end=sampler.config.t_end,
            ss=ss,
            time_scale=cfg.time_scale,
        )
    )

    guidance_log = []
    info: Dict = {}
    while True:
        window = np.stack(history[-cfg.n_obs_steps :])
        if window.shape[0] < cfg.n_obs_steps:
            pad = np.repeat(window[:1], cfg.n_obs_steps - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        obs_t = torch.from_numpy(window[None]).float().to(dev)
        proprio_t = torch.from_numpy(obs["proprio"][None]).float().to(dev)

        context = {"proprio": proprio_t, "state": proprio_t, "module_index": module_index}

        if modality == "reject" and energy is not None:
            candidates = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = policy.sample(obs_t, sampler=sampler)
                    candidates.append(out["n_action"])
            best, best_e = None, None
            for cand in candidates:
                e, _info = energy(cand[0], proprio_t[0], context)
                if best_e is None or float(e) < float(best_e):
                    best, best_e = cand, e
            action_chunk = policy.normalizer.unnormalize("action", best)
        else:
            out = policy.sample(obs_t, sampler=sampler, guidance=guide, ctx=context, record=guide is not None)
            action_chunk = out["actions"]
            if out.get("trace") is not None:
                trace = out["trace"]
                guidance_log.append(
                    {
                        "guidance_norms": trace.as_dict()["guidance_norms"],
                        "energies": [float(e) for e in trace.energies],
                    }
                )

        acts = action_chunk[0, : cfg.n_action_steps].detach().cpu().numpy()
        for a in acts:
            obs, _reward, done, info = env.step(a)
            history.append(obs["proprio"].copy())
            if done:
                break
        if done:
            break

    return {
        "task": task_name,
        "method": method,
        "success": float(info.get("success", False)),
        "constraint": float(info.get("constraint_satisfied_success", False)),
        "drift": float(info.get("attitude_drift", 0.0)),
        "force": float(info.get("max_force", 0.0)),
        "steps": float(info.get("steps", 0)),
        "failure": info.get("failure_reason", ""),
        "guidance_log": guidance_log[:2],
    }


# ------------------------------------------------------------- evaluation ----
def evaluate(
    methods: List[str],
    tasks: List[str],
    episodes: int,
    policy,
    sampler,
    spec,
    dynamics,
    banks,
    negatives,
    **kwargs,
) -> List[Dict]:
    rows = []
    for method in methods:
        for task in tasks:
            results = []
            t0 = time.time()
            for ep in range(episodes):
                results.append(
                    rollout_episode(
                        SatelliteEnv(spec=spec, seed=1000 + ep, randomize=True),
                        policy,
                        task,
                        sampler,
                        method,
                        spec,
                        dynamics=dynamics,
                        banks=banks,
                        negatives=negatives,
                        seed=1000 + ep,
                        **kwargs,
                    )
                )
            n = len(results)
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "episodes": n,
                    "success": float(np.mean([r["success"] for r in results])),
                    "constraint": float(np.mean([r["constraint"] for r in results])),
                    "drift": float(np.mean([r["drift"] for r in results])),
                    "force": float(np.mean([r["force"] for r in results])),
                    "steps": float(np.mean([r["steps"] for r in results])),
                    "seconds": time.time() - t0,
                }
            )
            print(
                f"{method:8s} {task:15s} success={rows[-1]['success']:.2f} "
                f"constraint={rows[-1]['constraint']:.2f} drift={rows[-1]['drift']:.3f} "
                f"force={rows[-1]['force']:.2f} ({rows[-1]['seconds']:.1f}s)"
            )
    return rows


def to_markdown(rows: List[Dict]) -> str:
    header = "| method | task | success | constraint-satisfied | base drift [rad] | max force [N] |\n|---|---|---|---|---|---|"
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['task']} | {r['success']*100:.0f}% | {r['constraint']*100:.0f}% | "
            f"{r['drift']:.3f} | {r['force']:.2f} |"
        )
    # aggregate per method
    methods = sorted({r["method"] for r in rows}, key=lambda m: METHODS.index(m) if m in METHODS else 99)
    lines.append("")
    lines.append("| method | mean success | mean constraint-satisfied | mean drift |")
    lines.append("|---|---|---|---|")
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        lines.append(
            f"| {m} | {np.mean([r['success'] for r in sub])*100:.0f}% | "
            f"{np.mean([r['constraint'] for r in sub])*100:.0f}% | "
            f"{np.mean([r['drift'] for r in sub]):.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, default=str(CKPT_DIR / "flow_policy.pt"))
    p.add_argument("--dynamics", type=str, default=str(CKPT_DIR / "latent_dynamics.pt"))
    p.add_argument("--methods", type=str, nargs="+", default=["base", "latent", "physics", "both"])
    p.add_argument("--tasks", type=str, nargs="+", default=["payload_box", "solar_panel", "battery_module"])
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument(
        "--per_method_scale",
        type=str,
        default="",
        help="per-method guidance scale, e.g. latent=1.0,physics=0.25,both=0.25",
    )
    p.add_argument("--weight", type=str, default="fp", help="fp | linear | ddim | const | noise_fraction")
    p.add_argument("--alpha", type=float, default=30.0)
    p.add_argument("--ss", type=int, default=1)
    p.add_argument("--steps", type=int, default=4, help="flow sampler steps (N)")
    p.add_argument("--sampler", type=str, default="euler", choices=["euler", "midpoint", "heun"])
    p.add_argument("--n_samples", type=int, default=16, help="rejection sampling candidates")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "eval.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_seed(args.seed)
    dev = device()

    spec = make_sc1_spec()
    policy = load_policy(args.policy, dev)
    dynamics, banks, negatives, _blob = load_dynamics(args.dynamics, dev)
    sampler = FlowSampler(
        FlowSamplerConfig(n_steps=args.steps, method=args.sampler, t_start=1e-2, t_end=1.0, time_scale=policy.config.time_scale)
    )

    tasks = [t for t in args.tasks if t in banks]
    scale_map = {}
    for part in args.per_method_scale.split(","):
        if "=" in part:
            key, value = part.split("=")
            scale_map[key.strip()] = float(value)
    rows = []
    for method in args.methods:
        rows.extend(
            evaluate(
                methods=[method],
                tasks=tasks,
                episodes=args.episodes,
                policy=policy,
                sampler=sampler,
                spec=spec,
                dynamics=dynamics,
                banks=banks,
                negatives=negatives,
                scale=scale_map.get(method, args.scale),
                weight_mode=args.weight,
                alpha=args.alpha,
                ss=args.ss,
                n_samples=args.n_samples,
                grad_clip=args.grad_clip,
            )
        )
    table = to_markdown(rows)
    print()
    print(table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "rows": rows, "table": table}, f, indent=2)
    with open(str(args.out).replace(".json", ".md"), "w", encoding="utf-8") as f:
        f.write("# Flow-DynaGuide on the free-floating satellite disassembly task\n\n")
        f.write(f"scale={args.scale}, weight={args.weight}, alpha={args.alpha}, ss={args.ss}, "
                f"sampler={args.sampler}, steps={args.steps}, episodes/task={args.episodes}\n\n")
        f.write(table + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
