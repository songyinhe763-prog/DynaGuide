"""Collect scripted demonstrations for the satellite disassembly benchmark.

The dataset is deliberately **multimodal**: each episode uses one of three demonstration
strategies (``safe`` / ``impatient`` / ``offset``, see
:class:`core.satellite.expert.ScriptedExpert`).  Behaviour cloning on this mixture
reproduces the DynaGuide setting -- the base policy learns an average of the strategies,
which is not the safe one -- and the guidance at inference time is what selects the safe
behaviour.

Usage
-----
::

    python scripts/satellite/make_demos.py --num_episodes 120 --noise 0.05

The dataset is stored as ``data/satellite/demos_<tag>.npz`` with ``obs`` ``(N, D)``,
``actions`` ``(N, 7)`` (normalised to ``[-1, 1]``), ``task_id``, ``episode_id``,
``behavior`` and ``ep_ok`` (per-episode safety flag).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, set_seed  # noqa: E402

from core.satellite import collect_demonstrations, make_sc1_spec  # noqa: E402
from core.satellite.data import save_demos  # noqa: E402


def parse_mix(text: str) -> dict:
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, value = part.split("=")
        out[name.strip()] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=120)
    parser.add_argument("--per_task", type=int, default=0, help="episodes per module (overrides num_episodes)")
    parser.add_argument("--noise", type=float, default=0.05, help="action noise scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="mixed")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["payload_box", "solar_panel", "battery_module"],
        help="modules to demonstrate (one module per episode)",
    )
    parser.add_argument(
        "--behavior_mix",
        type=str,
        default="safe=0.6,impatient=0.2,offset=0.2",
        help="demonstration strategy mixture, e.g. safe=0.6,impatient=0.2,offset=0.2",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    behavior_mix = parse_mix(args.behavior_mix)

    spec = make_sc1_spec()
    per_task = args.per_task or max(1, args.num_episodes // len(args.tasks))
    parts = []
    for i, task in enumerate(args.tasks):
        data = collect_demonstrations(
            spec=spec,
            num_episodes=per_task,
            tasks_per_episode=[task],
            seed=args.seed + 100 * i,
            noise=args.noise,
            behavior_mix=behavior_mix,
            verbose=args.verbose,
        )
        parts.append(data)
        print(
            f"[{task}] episodes={per_task} transitions={len(data['actions'])} "
            f"success={data['success_rate']:.2f} constraint={data['constraint_rate']:.2f}"
        )

    merged = {
        "obs": np.concatenate([p["obs"] for p in parts], axis=0),
        "actions": np.concatenate([p["actions"] for p in parts], axis=0),
        "task_id": np.concatenate([p["task_id"] for p in parts], axis=0),
        "episode_id": np.concatenate([p["episode_id"] + k * 100_000 for k, p in enumerate(parts)], axis=0),
        "behavior": np.concatenate([p["behavior"] for p in parts], axis=0),
        "ep_ok": np.concatenate([p["ep_ok"] for p in parts], axis=0),
    }
    out = DATA_DIR / f"demos_{args.tag}.npz"
    save_demos(str(out), merged)

    all_infos = [info for p in parts for info in p["infos"]]
    by_behavior = {}
    for name in behavior_mix:
        sub = [i for i in all_infos if i["behavior"] == name]
        if sub:
            by_behavior[name] = {
                "episodes": len(sub),
                "success": float(np.mean([i["success"] for i in sub])),
                "constraint": float(np.mean([i["constraint"] for i in sub])),
                "max_force": float(np.mean([i["max_force"] for i in sub])),
                "drift": float(np.mean([i["attitude_drift"] for i in sub])),
            }
    meta = {
        "tag": args.tag,
        "num_transitions": int(len(merged["actions"])),
        "module_names": list(spec.module_names),
        "behavior_names": list(behavior_mix),
        "proprio_dim": int(merged["obs"].shape[1]),
        "action_dim": int(merged["actions"].shape[1]),
        "success_rate": float(np.mean([i["success"] for i in all_infos])),
        "constraint_rate": float(np.mean([i["constraint"] for i in all_infos])),
        "per_behavior": by_behavior,
        "tasks": list(args.tasks),
        "noise": args.noise,
    }
    with open(DATA_DIR / f"demos_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(by_behavior, indent=2))
    print(f"saved {out} ({meta['num_transitions']} transitions, proprio_dim={meta['proprio_dim']})")


if __name__ == "__main__":
    main()
