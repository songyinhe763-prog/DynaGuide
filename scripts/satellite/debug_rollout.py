"""Debug helper: roll out the base flow policy and print the per-replan state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.flows.policy import FlowActionChunkPolicy  # noqa: E402
from core.flows.samplers import FlowSampler, FlowSamplerConfig  # noqa: E402
from core.satellite.env import SatelliteEnv  # noqa: E402
from core.satellite.spec import make_sc1_spec  # noqa: E402


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints/satellite/flow_policy_payload.pt")
    task = sys.argv[2] if len(sys.argv) > 2 else "payload_box"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    torch.manual_seed(0)
    spec = make_sc1_spec()
    env = SatelliteEnv(spec=spec, seed=seed, randomize=True)
    pol = FlowActionChunkPolicy.load(ckpt)
    sm = FlowSampler(
        FlowSamplerConfig(n_steps=4, method="euler", t_start=1e-2, t_end=1.0, time_scale=pol.config.time_scale)
    )
    obs = env.reset(task=task, seed=seed)
    hist = [obs["proprio"].copy()]
    mod = spec.module(task)
    idx = spec.module_names.index(task)
    done, info = False, {}
    for step_i in range(0, spec.horizon, pol.config.n_action_steps):
        w = np.stack(hist[-2:])
        if w.shape[0] < 2:
            w = np.concatenate([w[:1], w])
        with torch.no_grad():
            out = pol.sample(torch.from_numpy(w[None]).float(), sampler=sm)
        acts = out["actions"][0, : pol.config.n_action_steps].numpy()
        gd = float(np.linalg.norm(env.ee_pos - mod.grasp_pose))
        print(
            f"replan {step_i:3d} ee={np.round(env.ee_pos,3)} grasp_d={gd:.3f} "
            f"grasped={env.grasped} twist={env.twist[idx]:.2f} pull={env.pull[idx]:.3f} "
            f"drift={env.attitude_drift:.3f} a0={np.round(acts[0],2)} grip={acts[:,6].mean():.2f}"
        )
        for a in acts:
            obs, _r, done, info = env.step(a)
            hist.append(obs["proprio"].copy())
            if done:
                break
        if done:
            break
    print("FINAL done=", done, {k: info.get(k) for k in ("success", "constraint_satisfied_success", "failure_reason", "max_force", "attitude_drift", "steps")})


if __name__ == "__main__":
    main()
