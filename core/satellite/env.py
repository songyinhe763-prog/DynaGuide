"""Free-floating satellite disassembly environment.

Design goals
------------
* **Deterministic and dependency free** (``numpy`` only) so that the whole
  DynaGuide-for-flow pipeline can be trained and evaluated on a CPU.
* **Momentum conserving**: the arm is mounted on the same free-floating body as the
  modules, so every contact wrench used to free a module is reacted by the base.  This
  is the microgravity phenomenon the guidance is supposed to tame: a policy that simply
  yanks at a bolted module "succeeds" mechanically but throws the satellite away.
* **Joint-specific protocols**: bolts need a twist before a pull, latches need a press,
  pegs need a straight breakaway pull.  A behaviour-cloning policy trained on a mixture
  of all three behaviours can pick the wrong protocol -- exactly the multimodal ambiguity
  DynaGuide steers at inference time.
* **Task lists**: a module (or a sequence of modules) can be requested, which gives the
  multi-objective experiment of the DynaGuide paper its satellite counterpart.

The simulator is quasi-static: a command is the desired end-effector displacement, is
converted into an impedance force ``F = k_imp * dp``, and the *blocked* component of that
force is applied as a reaction wrench to the base (Newton's third law).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.satellite import math3d as m3
from core.satellite.spec import SatelliteSpec, make_sc1_spec

# proprioception layout (documented; used by the latent encoder, the guidance energies
# and the analysis scripts)
BASE_POS = slice(0, 3)
BASE_QUAT = slice(3, 7)
BASE_LINVEL = slice(7, 10)
BASE_ANGVEL = slice(10, 13)
EE_POS = slice(13, 16)
EE_ROTVEC = slice(16, 19)
GRIP = 19
MODULE_BLOCK_START = 20
TIME_INDEX_OFFSET = 1


def proprio_dim(num_modules: int) -> int:
    """20 fixed entries + 5 per module + 1 time fraction."""
    return 20 + 5 * num_modules + 1


def module_block(i: int) -> slice:
    """Per-module block: ``[attached, twist_ratio, press_ratio, pull_ratio, grasp_dist]``."""
    start = MODULE_BLOCK_START + 5 * i
    return slice(start, start + 5)


class SatelliteEnv:
    """Gym-like environment (no ``gym`` dependency)."""

    def __init__(
        self,
        spec: Optional[SatelliteSpec] = None,
        seed: int = 0,
        randomize: bool = True,
        task: Optional[str] = None,
        tasks: Optional[Sequence[str]] = None,
        use_task_conditioning: bool = False,
    ):
        self.spec = spec or make_sc1_spec()
        self.num_modules = len(self.spec.modules)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.use_task_conditioning = use_task_conditioning
        self.proprio_dim = proprio_dim(self.num_modules)
        self.action_dim = 7
        self._initial_tasks = list(tasks) if tasks is not None else ([task] if task else [])
        self.reset()

    # ------------------------------------------------------------------ obs ----
    def _proprio(self) -> np.ndarray:
        s = self.spec
        parts: List[np.ndarray] = [
            self.base_pos.astype(np.float64),  # 0:3
            self.base_quat.astype(np.float64),  # 3:7
            self.base_linvel.astype(np.float64),  # 7:10
            self.base_angvel.astype(np.float64),  # 10:13
            self.ee_pos.astype(np.float64),  # 13:16
            m3.quat_to_rotvec(self.ee_quat).astype(np.float64),  # 16:19
            np.array([self.grip], dtype=np.float64),  # 19
        ]
        for i, mod in enumerate(s.modules):
            j = mod.joint
            attached = 0.0 if (self.detached[i] or self.stowed[i]) else 1.0
            twist_ratio = float(np.clip(self.twist[i] / max(j.required_twist, 1e-6), 0.0, 1.5))
            press_ratio = float(np.clip(self.press[i] / max(j.press_depth, 1e-6), 0.0, 1.5))
            pull_ratio = float(np.clip(self.pull[i] / max(j.travel, 1e-6), 0.0, 1.5))
            grasp_dist = float(np.linalg.norm(self.ee_pos - mod.grasp_pose))
            parts.append(
                np.array(
                    [attached, twist_ratio, press_ratio, pull_ratio, grasp_dist],
                    dtype=np.float64,
                )
            )
        parts.append(np.array([self.step_count / max(self.spec.horizon, 1)], dtype=np.float64))
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.proprio_dim, (obs.shape, self.proprio_dim)
        return obs

    def _obs(self) -> Dict[str, np.ndarray]:
        obs = {"proprio": self._proprio()}
        if self.use_task_conditioning:
            onehot = np.zeros(self.num_modules, dtype=np.float32)
            if self.task is not None:
                onehot[self.spec.module_names.index(self.task)] = 1.0
            obs["task_onehot"] = onehot
        return obs

    @property
    def task_index(self) -> int:
        return 0 if self.task is None else self.spec.module_names.index(self.task)

    # ---------------------------------------------------------------- reset ----
    def reset(
        self,
        task: Optional[str] = None,
        tasks: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if tasks is not None:
            self.tasks = list(tasks)
        elif task is not None:
            self.tasks = [task]
        elif not hasattr(self, "tasks"):
            self.tasks = list(self._initial_tasks)
        self.task_idx = 0
        self.task = self.tasks[0] if self.tasks else None

        s = self.spec
        self.step_count = 0

        if self.randomize:
            tilt = self.rng.normal(scale=0.06, size=3)
            self.base_quat = m3.rotvec_quat(tilt)
            self.base_pos = self.rng.normal(scale=0.005, size=3)
            self.base_linvel = self.rng.normal(scale=2e-3, size=3)
            self.base_angvel = self.rng.normal(scale=2e-3, size=3)
        else:
            self.base_quat = m3.quat_identity()
            self.base_pos = np.zeros(3)
            self.base_linvel = np.zeros(3)
            self.base_angvel = np.zeros(3)
        self.base_quat_init = self.base_quat.copy()

        jitter = self.rng.normal(scale=0.01, size=3) if self.randomize else np.zeros(3)
        self.ee_pos = np.array(s.ee_start, dtype=np.float64) + jitter
        self.ee_quat = m3.quat_identity()
        self.grip = 0.0
        self.grasped: Optional[int] = None

        self.detached = [False] * self.num_modules
        self.stowed = [False] * self.num_modules
        self.twist = [0.0] * self.num_modules
        self.press = [0.0] * self.num_modules
        self.pull = [0.0] * self.num_modules
        self.free_pos = [np.array(m.grasp_pose, dtype=np.float64) for m in s.modules]
        self.free_vel = [np.zeros(3) for _ in range(self.num_modules)]

        self.max_force = 0.0
        self.force_impulse = 0.0
        self.attitude_drift = 0.0
        self.captured = False
        self.failed = False
        self.failure_reason = ""
        self.done_tasks: List[str] = []
        return self._obs()

    # ----------------------------------------------------------------- step ----
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        s = self.spec
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.shape[0] != 7:
            raise ValueError(f"expected 7 action dims, got {a.shape}")

        dp = a[0:3] * s.dp_scale  # desired ee displacement, base frame
        dth = a[3:6] * s.drot_scale  # desired ee rotation (rotvec), base frame
        grip_cmd = float(a[6])
        self.step_count += 1

        # ------------------------------------------------------- grasp logic ----
        want_grip = grip_cmd > 0.0
        if self.grasped is None and want_grip:
            best, best_d = None, 1e9
            for i, mod in enumerate(s.modules):
                if self.stowed[i]:
                    continue
                d = float(np.linalg.norm(self.ee_pos - mod.grasp_pose))
                if d < s.grasp_tol and d < best_d:
                    best, best_d = i, d
            if best is not None:
                self.grasped = best
        if self.grasped is not None and not want_grip:
            self.grasped = None
        self.grip = 1.0 if self.grasped is not None else (0.5 if want_grip else 0.0)

        # ----------------------------------------------------- ee kinematics ----
        self.ee_pos = self.ee_pos + dp
        self.ee_quat = m3.quat_normalize(m3.quat_multiply(m3.rotvec_quat(dth), self.ee_quat))
        lim = np.array([0.45, 0.45, 0.55])
        self.ee_pos = np.clip(self.ee_pos, -lim, lim)

        contact_force = 0.0
        blocked = np.zeros(3)

        # ------------------------------------------------- joint progression ----
        if self.grasped is not None:
            i = self.grasped
            mod = s.modules[i]
            j = mod.joint
            axis = j.axis_np  # joint axes are expressed in the (base) body frame
            axial = float(np.dot(dp, axis))
            dp_ax = axial * axis
            dp_tan = dp - dp_ax

            if not self.detached[i]:
                # 紧固件状态机 —— 这是“怎么拆”的全部规则
                #   螺栓：只有绕轴的正向转动才算拧（twist 累积），拧够才解锁；
                #   卡扣：只有沿轴向内压才算（press 累积），压够才解锁；
                #   销：  无需准备动作，直接看拉力是否超过破断力。
                if j.kind == "bolt" and self.twist[i] < j.required_twist:
                    self.twist[i] += max(0.0, float(np.dot(dth, axis)))
                if j.kind == "latch" and self.press[i] < j.press_depth:
                    self.press[i] += max(0.0, -axial)

                unlocked = True
                if j.kind == "bolt":
                    unlocked = self.twist[i] >= (j.required_twist - s.twist_tol)
                elif j.kind == "latch":
                    unlocked = self.press[i] >= (j.press_depth - 1e-9)

                if not unlocked:
                    # tangential motion is always resisted while the module is fastened
                    blocked = blocked + dp_tan
                    if axial > 0:
                        blocked = blocked + dp_ax  # cannot pull yet
                    elif axial < 0 and not (j.kind == "latch" and self.press[i] < j.press_depth):
                        blocked = blocked + dp_ax  # cannot push the module into the body
                else:
                    blocked = blocked + dp_tan
                    if axial > 0:
                        pull_force = s.k_imp * axial
                        if pull_force > j.breakaway_force:
                            self.pull[i] = min(j.travel, self.pull[i] + axial)
                        else:
                            blocked = blocked + dp_ax  # not enough force: still stuck
                    elif axial < 0:
                        blocked = blocked + dp_ax  # the body blocks inward motion

                if self.pull[i] >= j.travel - 1e-9:
                    self.detached[i] = True
                    self.free_pos[i] = self.ee_pos.copy()
                    self.free_vel[i] = np.zeros(3)

            contact_force = float(np.linalg.norm(blocked) * s.k_imp)

            if contact_force > 1e-9:
                # 反作用力偶。力矩 = r × (−F)，r 是基座质心到安装点的矢量。
                #   基座惯量只有 0.6 kg·m² 量级，一次几牛的偏载就足以让姿态明显漂移，
                #   所以“温柔地拆”不是风格问题，而是能不能完成任务的问题。
                direction = blocked / max(float(np.linalg.norm(blocked)), 1e-12)
                rot = m3.quat_matrix(self.base_quat)
                f_world = rot @ (direction * contact_force)
                r_arm = rot @ mod.attach_np
                self.base_linvel = self.base_linvel + (-f_world) * s.dt / s.base_mass
                torque = np.cross(r_arm, -f_world)
                inertia = np.asarray(s.base_inertia, dtype=np.float64)
                self.base_angvel = self.base_angvel + torque * s.dt / inertia
                self.force_impulse += contact_force * s.dt

        # a grasped free module rides with the end effector
        if self.grasped is not None and self.detached[self.grasped]:
            self.free_pos[self.grasped] = self.ee_pos.copy()

        # -------------------------------------------------------- base motion ----
        self.base_pos = self.base_pos + self.base_linvel * s.dt
        self.base_quat = m3.integrate_quat(self.base_quat, self.base_angvel, s.dt)
        self.attitude_drift = m3.quat_angle(self.base_quat_init, self.base_quat)
        self.base_angvel *= (1.0 - 0.01 * s.dt)  # passive nutation damping (weak)
        self.base_linvel *= (1.0 - 0.01 * s.dt)
        self.max_force = max(self.max_force, contact_force)
        self.last_force = contact_force  # per-step contact force, used by the trace plots

        # ------------------------------------------------- removal / advance ----
        # The task is "remove the requested module": once a module is fully extracted it
        # counts as done and the episode moves on to the next objective (if any).  With
        # ``spec.require_stow = True`` the module must additionally be brought to its
        # staging pose before it counts -- the stricter variant used for ablations.
        if self.grasped is not None and self.detached[self.grasped]:
            idx = self.grasped
            mod = s.modules[idx]
            reached = float(np.linalg.norm(self.ee_pos - mod.capture_np)) < s.capture_tol
            if (not s.require_stow) or reached:
                self.captured = True
                self.stowed[idx] = True
                self.done_tasks.append(mod.name)
                self.free_pos[idx] = self.ee_pos.copy()
                self.grasped = None
                self.grip = 0.0
                if self.task_idx + 1 < len(self.tasks):
                    self.task_idx += 1
                    self.task = self.tasks[self.task_idx]
                    self.captured = False  # keep going with the next objective
                else:
                    self.task = self.tasks[self.task_idx] if self.tasks else self.task

        # ------------------------------------------------------- termination ----
        all_done = bool(self.tasks) and len(self.done_tasks) >= len(self.tasks)
        if all_done:
            self.failed = False
        elif self.attitude_drift > s.attitude_budget:
            self.failed, self.failure_reason = True, "attitude_drift"
        elif self.max_force > s.force_limit:
            self.failed, self.failure_reason = True, "force_limit"
        elif self.step_count >= s.horizon:
            self.failed, self.failure_reason = True, "timeout"

        done = bool(all_done or self.failed)
        success = bool(all_done and not self.failed)
        pressure_ok = self.max_force <= s.force_budget
        attitude_ok = self.attitude_drift <= s.attitude_budget
        info = {
            "success": success,
            "constraint_satisfied_success": bool(success and pressure_ok and attitude_ok),
            "captured": bool(self.captured),
            "detached": bool(any(self.detached[i] for i in range(self.num_modules) if self.stowed[i]))
            if any(self.stowed)
            else False,
            "failed": bool(self.failed),
            "failure_reason": self.failure_reason,
            "max_force": float(self.max_force),
            "force_limit": float(s.force_limit),
            "force_budget": float(s.force_budget),
            "attitude_drift": float(self.attitude_drift),
            "attitude_budget": float(s.attitude_budget),
            "force_ok": bool(pressure_ok),
            "attitude_ok": bool(attitude_ok),
            "steps": self.step_count,
            "grasped": None if self.grasped is None else self.spec.module_names[self.grasped],
            "task": self.task,
            "done_tasks": list(self.done_tasks),
        }
        reward = 1.0 if success else -0.02
        return self._obs(), reward, done, info

    # -------------------------------------------------------------- helpers ----
    def is_success(self) -> Dict[str, bool]:
        return {"task": bool(self.tasks) and len(self.done_tasks) == len(self.tasks) and not self.failed}

    @property
    def module_states(self) -> List[Dict]:
        return [
            dict(
                name=mod.name,
                attached=not (self.detached[i] or self.stowed[i]),
                twist=self.twist[i],
                press=self.press[i],
                pull=self.pull[i],
                pos=self.free_pos[i].copy(),
            )
            for i, mod in enumerate(self.spec.modules)
        ]

    def clone(self) -> "SatelliteEnv":
        import copy

        return copy.deepcopy(self)
