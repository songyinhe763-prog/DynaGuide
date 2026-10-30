"""Sweep the guidance hyperparameters (scale / weight schedule / sampler steps / ss).

Guidance strength is notoriously sensitive (QGF's official implementation sweeps it value by
value), so this script wraps :func:`scripts.satellite.run_flow_dynaguide.evaluate` and writes
one markdown table per swept axis.

Usage
-----
::

    python scripts/satellite/sweep_guidance.py --policy ... --dynamics ... \
        --scales 0.25 0.5 1.0 2.0 --methods both --episodes 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DIR, RESULTS_DIR, device, set_seed  # noqa: E402
from run_flow_dynaguide import evaluate, load_dynamics, load_policy, to_markdown  # noqa: E402

from core.flows.samplers import FlowSampler, FlowSamplerConfig  # noqa: E402
from core.satellite.spec import make_sc1_spec  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, default=str(CKPT_DIR / "flow_policy_payload.pt"))
    p.add_argument("--dynamics", type=str, default=str(CKPT_DIR / "latent_payload.pt"))
    p.add_argument("--tasks", type=str, nargs="+", default=["payload_box"])
    p.add_argument("--methods", type=str, nargs="+", default=["both"])
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--weights", type=str, nargs="+", default=["fp", "linear", "const"])
    p.add_argument("--alphas", type=float, nargs="+", default=[15.0, 30.0])
    p.add_argument("--steps", type=int, nargs="+", default=[4])
    p.add_argument("--ss", type=int, nargs="+", default=[1])
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "sweep.md"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_seed(args.seed)
    dev = device()

    spec = make_sc1_spec()
    policy = load_policy(args.policy, dev)
    dynamics, banks, negatives, _blob = load_dynamics(args.dynamics, dev)
    tasks = [t for t in args.tasks if t in banks]

    all_rows = []
    for scale in args.scales:
        for weight in args.weights:
            for alpha in args.alphas:
                for steps in args.steps:
                    for ss in args.ss:
                        sampler = FlowSampler(
                            FlowSamplerConfig(
                                n_steps=steps,
                                method="euler",
                                t_start=1e-2,
                                t_end=1.0,
                                time_scale=policy.config.time_scale,
                            )
                        )
                        rows = evaluate(
                            methods=args.methods,
                            tasks=tasks,
                            episodes=args.episodes,
                            policy=policy,
                            sampler=sampler,
                            spec=spec,
                            dynamics=dynamics,
                            banks=banks,
                            negatives=negatives,
                            scale=scale,
                            weight_mode=weight,
                            alpha=alpha,
                            ss=ss,
                        )
                        for r in rows:
                            r.update({"scale": scale, "weight": weight, "alpha": alpha, "steps": steps, "ss": ss})
                        all_rows.extend(rows)

    lines = ["# Guidance hyperparameter sweep", ""]
    header = "| method | scale | weight | alpha | steps | ss | success | constraint-satisfied | drift |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    for method in args.methods:
        lines += [f"## {method}", "", header, sep]
        sub = [r for r in all_rows if r["method"] == method]
        sub.sort(key=lambda r: (-r["constraint"], r["drift"]))
        for r in sub:
            lines.append(
                f"| {r['method']} | {r['scale']:.2f} | {r['weight']} | {r['alpha']:.0f} | {r['steps']} | {r['ss']} | "
                f"{r['success']*100:.0f}% | {r['constraint']*100:.0f}% | {r['drift']:.3f} |"
            )
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    json.dump({"rows": all_rows, "config": vars(args)}, open(out.with_suffix(".json"), "w"), indent=2)
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
