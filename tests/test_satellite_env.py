"""Tests of the free-floating satellite disassembly environment and scripted expert."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.satellite import (  # noqa: E402
    SatelliteEnv,
    ScriptedExpert,
    collect_demonstrations,
    make_sc1_spec,
    run_episode,
)
from core.satellite.env import module_block, proprio_dim  # noqa: E402


def test_proprio_layout():
    spec = make_sc1_spec()
    env = SatelliteEnv(spec=spec, seed=0)
    assert env.proprio_dim == proprio_dim(3) == 20 + 5 * 3 + 1
    obs = env.reset(task="payload_box")
    assert obs["proprio"].shape == (env.proprio_dim,)
    # the module block must encode the connection state
    blk = obs["proprio"][module_block(0)]
    assert blk[0] == 1.0, "payload_box must start attached"


def test_expert_solves_each_single_module_task():
    spec = make_sc1_spec()
    for name in spec.module_names:
        env = SatelliteEnv(spec=spec, seed=1, randomize=False)
        env.reset(task=name)
        expert = ScriptedExpert(spec=spec)
        rollout = run_episode(env, expert, noise=0.0)
        info = rollout["info"]
        assert info["success"], f"{name} not solved: {info}"
        assert info["max_force"] < spec.force_limit * 0.5, info
        assert info["attitude_drift"] < spec.attitude_budget * 0.2, info


def test_expert_solves_multi_objective_episode():
    import copy

    spec = copy.deepcopy(make_sc1_spec())
    spec.horizon = 320
    env = SatelliteEnv(spec=spec, seed=2, randomize=False)
    env.reset(tasks=["solar_panel", "payload_box"])
    expert = ScriptedExpert(spec=spec)
    rollout = run_episode(env, expert, noise=0.0)
    assert rollout["info"]["done_tasks"] == ["solar_panel", "payload_box"], rollout["info"]
    assert rollout["info"]["success"], rollout["info"]


def test_wrong_protocol_causes_base_reaction():
    """Pulling a bolted module without unscrewing it must push the satellite around."""
    spec = make_sc1_spec()
    env = SatelliteEnv(spec=spec, seed=3, randomize=False)
    env.reset(task="payload_box")
    expert = ScriptedExpert(spec=spec)

    # approach + grasp with the expert, then yank axially without ever twisting
    while env.grasped is None:
        env.step(expert.act(env))
    axis = spec.module("payload_box").joint.axis_np
    for _ in range(12):
        a = np.zeros(7)
        a[0:3] = axis * 0.5  # naive pull, no twist
        a[6] = 1.0
        _, _, done, info = env.step(a)
        if done:
            break
    assert env.max_force > 1.0, "a blocked pull must generate contact force"
    assert env.attitude_drift > 1e-3, "the reaction must move the free-floating base"
    assert env.twist[0] < 1e-6, "the naive policy never unscrewed the bolt"


def test_determinism():
    spec = make_sc1_spec()
    env_a = SatelliteEnv(spec=spec, seed=5, randomize=True)
    env_b = SatelliteEnv(spec=spec, seed=5, randomize=True)
    env_a.reset(seed=11)
    env_b.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(15):
        a = rng.uniform(-1, 1, size=7)
        oa, ra, da, ia = env_a.step(a)
        ob, rb, db, ib = env_b.step(a)
        assert np.allclose(oa["proprio"], ob["proprio"])
        assert ra == rb


def test_force_limit_failure_is_flagged():
    spec = make_sc1_spec()
    spec.force_limit = 0.5  # lower the safety budget on purpose
    env = SatelliteEnv(spec=spec, seed=4, randomize=False)
    env.reset(task="battery_module")
    expert = ScriptedExpert(spec=spec)
    while env.grasped is None:
        env.step(expert.act(env))
    axis = spec.module("battery_module").joint.axis_np
    done = False
    for _ in range(spec.horizon):
        a = np.zeros(7)
        a[0:3] = -axis * 1.0  # push the module into the body -> blocked
        a[6] = 1.0
        _, _, done, info = env.step(a)
        if done:
            break
    assert info["failed"] and info["failure_reason"] in ("force_limit", "attitude_drift"), info


def test_collect_demonstrations_shapes():
    data = collect_demonstrations(num_episodes=4, tasks_per_episode=("payload_box",), seed=7, noise=0.05)
    assert data["obs"].shape[0] == data["actions"].shape[0] == data["task_id"].shape[0]
    assert data["obs"].shape[1] == proprio_dim(3)
    assert data["actions"].shape[1] == 7
    assert "behavior" in data and "ep_ok" in data


def test_safe_only_demonstrations_are_reliable():
    """With a single (safe) strategy the expert must nearly always succeed."""
    data = collect_demonstrations(
        num_episodes=6,
        tasks_per_episode=("payload_box",),
        seed=8,
        noise=0.02,
        behavior_mix={"safe": 1.0},
    )
    assert data["success_rate"] > 0.8, data["success_rate"]
    assert data["constraint_rate"] > 0.8, data["constraint_rate"]


def test_unsafe_behaviours_are_captured_by_the_metrics():
    """The impatient strategy must be less safe than the safe one (momentum reaction)."""
    safe = collect_demonstrations(
        num_episodes=5, tasks_per_episode=("payload_box",), seed=9, noise=0.0, behavior_mix={"safe": 1.0}
    )
    impatient = collect_demonstrations(
        num_episodes=5, tasks_per_episode=("payload_box",), seed=9, noise=0.0, behavior_mix={"impatient": 1.0}
    )
    drift_safe = np.mean([i["attitude_drift"] for i in safe["infos"]])
    drift_impatient = np.mean([i["attitude_drift"] for i in impatient["infos"]])
    force_safe = np.mean([i["max_force"] for i in safe["infos"]])
    force_impatient = np.mean([i["max_force"] for i in impatient["infos"]])
    assert drift_impatient > drift_safe, (drift_safe, drift_impatient)
    assert force_impatient > force_safe, (force_safe, force_impatient)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
