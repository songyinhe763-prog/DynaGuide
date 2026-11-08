"""Scripted expert that demonstrates the three disassembly protocols.

The expert is a deterministic phase machine driven by the *environment state* (not by a
private counter), so it also works for multi-objective episodes in which the requested
module changes after each capture:

1. **approach** the grasp point of the requested module,
2. **grasp**,
3. **prepare** the fastener -- twist for a ``bolt``, press for a ``latch``, nothing for a
   ``peg``,
4. **extract** along the joint axis with enough axial force to pass the breakaway,
5. **stow** the module at its staging pose (this is where ``capture`` is scored).

Because the protocol steps are executed with purely axial / purely rotational commands,
the scripted expert produces almost no blocked motion and therefore almost no base
reaction: it is the "safe" reference behaviour.  Small action noise is injected to make
the demonstrations diverse enough for behaviour cloning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.satellite import math3d as m3
from core.satellite.env import SatelliteEnv
from core.satellite.spec import SatelliteSpec, make_sc1_spec


@dataclass
class ScriptedExpert:
    """Phase-machine expert. ``act(env)`` returns a normalised 7-dim action.

    ``behavior`` selects the *demonstration strategy*, which is what makes the dataset
    multimodal:

    ``safe``
        canonical protocol: twist (bolt) / press (latch), then a purely axial pull.
    ``impatient``
        pulls first for a few steps (the fastener is not released yet, so the wrench is
        reacted by the free-floating base and the satellite starts to rotate) and only then
        executes the protocol.  Mechanically the episode can still succeed -- which is
        exactly the case that the constraint-aware metric must catch.
    ``offset``
        extracts along an axis tilted by ``tilt`` radians, so tangential motion is
        resisted by the fastener: contact forces plus a momentum kick.

    Mixing the three strategies yields the "heterogeneous demonstration quality" setting
    that DynaGuide's guidance is meant to fix at inference time.
    """

    spec: SatelliteSpec = field(default_factory=make_sc1_spec)
    behavior: str = "safe"
    k_p: float = 1.2  # proportional gain on the position error (in action units / m)
    approach_tol: float = 0.004
    twist_step: float = 0.10  # rad per step
    press_step: float = 0.0025  # m per step
    pull_ratio: float = 1.7  # axial command = breakaway / k_imp * pull_ratio
    stow_tol: float = 0.006
    impatient_steps: int = 4
    impatient_scale: float = 0.18
    tilt: float = 0.25  # rad, extraction-axis tilt for the "offset" behavior
    _t: int = field(default=0, init=False, repr=False)
    _grasp_steps: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._t = 0
        self._grasp_steps = 0

    # ------------------------------------------------------------------ api ----
    def act(self, env: SatelliteEnv) -> np.ndarray:
        spec = env.spec
        name = env.task
        if name is None:
            return np.zeros(7)
        idx = spec.module_names.index(name)
        mod = spec.modules[idx]
        j = mod.joint
        self._t += 1

        if env.stowed[idx]:
            return np.zeros(7)

        action = np.zeros(7, dtype=np.float64)

        if env.grasped is None:
            # ------------------------------------------------------ approach ----
            delta = mod.grasp_pose - env.ee_pos
            dist = float(np.linalg.norm(delta))
            if dist > self.approach_tol:
                action[0:3] = np.clip(self.k_p * delta / spec.dp_scale, -1.0, 1.0)
                action[6] = -1.0
            else:
                action[6] = 1.0  # close the gripper
            return action

        if not env.detached[idx]:
            self._grasp_steps += 1
            axis = j.axis_np
            prepared = True
            if j.kind == "bolt":
                prepared = env.twist[idx] >= j.required_twist - spec.twist_tol
            elif j.kind == "latch":
                prepared = env.press[idx] >= j.press_depth - 1e-9

            if not prepared:
                # impatient demonstrations yank before preparing the fastener: the fastener
                # is not released, so the whole wrench goes into the free-floating base
                if self.behavior == "impatient" and self._grasp_steps <= self.impatient_steps:
                    action[0:3] = axis * self.impatient_scale
                elif j.kind == "bolt":
                    remaining = j.required_twist - env.twist[idx]
                    step = min(self.twist_step, remaining)
                    action[3:6] = axis * (step / spec.drot_scale)
                else:
                    remaining = j.press_depth - env.press[idx]
                    step = min(self.press_step, remaining)
                    action[0:3] = -axis * (step / spec.dp_scale)
                action[6] = 1.0
                return action

            # --------------------------------------------------------- extract ----
            direction = axis
            if self.behavior == "offset":
                tilt_axis = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(tilt_axis, axis))) > 0.9:
                    tilt_axis = np.array([1.0, 0.0, 0.0])
                perp = np.cross(axis, tilt_axis)
                perp = perp / max(np.linalg.norm(perp), 1e-9)
                direction = axis * np.cos(self.tilt) + perp * np.sin(self.tilt)
                direction = direction / np.linalg.norm(direction)
            pull = max(self.pull_ratio * j.breakaway_force / spec.k_imp, 2e-3)
            pull = min(pull, 3.0 * j.breakaway_force / spec.k_imp)
            action[0:3] = direction * (pull / spec.dp_scale)
            action[6] = 1.0
            return action

        # ------------------------------------------------------------- stow ----
        delta = mod.capture_np - env.ee_pos
        dist = float(np.linalg.norm(delta))
        if dist > self.stow_tol:
            action[0:3] = np.clip(self.k_p * delta / spec.dp_scale, -1.0, 1.0)
        action[6] = 1.0
        return np.clip(action, -1.0, 1.0)


def run_episode(
    env: SatelliteEnv,
    expert: ScriptedExpert,
    noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    max_steps: Optional[int] = None,
) -> Dict:
    """Roll out the scripted expert for one (possibly multi-objective) episode."""
    rng = rng or np.random.default_rng(0)
    expert.reset()
    obs = env._obs()
    max_steps = max_steps or env.spec.horizon
    obs_list = [obs["proprio"].copy()]
    act_list: List[np.ndarray] = []
    task_list: List[int] = []
    info: Dict = {}
    for _ in range(max_steps):
        task_list.append(int(env.task_index))
        a = expert.act(env)
        if noise > 0:
            a = np.clip(a + rng.normal(scale=noise, size=7), -1.0, 1.0)
        next_obs, _reward, done, info = env.step(a)
        obs_list.append(next_obs["proprio"].copy())
        act_list.append(a.astype(np.float32))
        if done:
            break
    return {
        "obs": np.stack(obs_list),
        "actions": np.stack(act_list) if act_list else np.zeros((0, 7), np.float32),
        "task_id": np.asarray(task_list, dtype=np.int64),
        "info": info,
        "success": bool(info.get("success", False)),
        "constraint": bool(info.get("constraint_satisfied_success", False)),
        "tasks": list(env.tasks),
        "steps": len(act_list),
        "behavior": expert.behavior,
    }


DEFAULT_BEHAVIOR_MIX = {"safe": 0.6, "impatient": 0.2, "offset": 0.2}


def collect_demonstrations(
    spec: Optional[SatelliteSpec] = None,
    num_episodes: int = 60,
    tasks_per_episode: Sequence[str] = ("payload_box",),
    seed: int = 0,
    noise: float = 0.06,
    randomize: bool = True,
    behavior_mix: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> Dict:
    """Collect a behaviour-cloning dataset with the scripted expert.

    Returns a dict with stacked ``obs`` ``(N, D)``, ``actions`` ``(N, 7)``, ``task_id``
    ``(N,)`` (index of the module being worked on), ``episode_id`` ``(N,)``,
    ``behavior`` ``(N,)`` (0 safe / 1 impatient / 2 offset) and ``ep_ok`` ``(N,)`` (1 if
    the *episode* satisfied both safety budgets).  ``ep_ok`` is what splits the goal bank
    into positive (safe) and negative (unsafe) guidance conditions.
    """
    spec = spec or make_sc1_spec()
    rng = np.random.default_rng(seed)
    behavior_mix = behavior_mix or dict(DEFAULT_BEHAVIOR_MIX)
    names = list(behavior_mix.keys())
    probs = np.asarray([behavior_mix[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    env = SatelliteEnv(spec=spec, seed=seed, randomize=randomize)

    all_obs, all_act, all_task, all_ep, all_beh, all_ok = [], [], [], [], [], []
    infos = []
    for ep in range(num_episodes):
        behavior = names[int(rng.choice(len(names), p=probs))]
        expert = ScriptedExpert(spec=spec, behavior=behavior)
        env.reset(tasks=list(tasks_per_episode), seed=int(rng.integers(1 << 30)))
        expert_noise = noise * (0.5 + rng.random())
        rollout = run_episode(env, expert, noise=expert_noise, rng=rng)
        obs, acts = rollout["obs"], rollout["actions"]
        if acts.shape[0] == 0:
            continue
        n = acts.shape[0]
        all_obs.append(obs[:-1])
        all_act.append(acts)
        all_task.append(rollout["task_id"][:n])
        all_ep.append(np.full(n, ep, dtype=np.int64))
        all_beh.append(np.full(n, names.index(behavior), dtype=np.int64))
        all_ok.append(np.full(n, 1.0 if rollout["constraint"] else 0.0, dtype=np.float32))
        infos.append(
            {
                "behavior": behavior,
                "success": rollout["success"],
                "constraint": rollout["constraint"],
                "max_force": rollout["info"].get("max_force", 0.0),
                "attitude_drift": rollout["info"].get("attitude_drift", 0.0),
            }
        )
        if verbose:
            print(
                f"ep {ep:03d} {behavior:10s} steps={n:3d} success={rollout['success']} "
                f"constraint={rollout['constraint']} drift={rollout['info'].get('attitude_drift', 0):.3f} "
                f"force={rollout['info'].get('max_force', 0):.2f}"
            )

    data = {
        "obs": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_act, axis=0),
        "task_id": np.concatenate(all_task, axis=0),
        "episode_id": np.concatenate(all_ep, axis=0),
        "behavior": np.concatenate(all_beh, axis=0),
        "ep_ok": np.concatenate(all_ok, axis=0),
        "module_names": list(spec.module_names),
        "behavior_names": np.asarray(names),
        "proprio_dim": int(env.proprio_dim),
        "success_rate": float(np.mean([i["success"] for i in infos])) if infos else 0.0,
        "constraint_rate": float(np.mean([i["constraint"] for i in infos])) if infos else 0.0,
        "infos": infos,
    }
    return data
