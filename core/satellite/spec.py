"""Specifications of the satellite-disassembly benchmark used by the paper experiments.

The scene is a simplified, *free-floating* satellite mock-up.  A dead (uncontrolled)
satellite body carries a handful of serviceable modules, each fastened by a different
mechanical joint type.  The robot arm is anchored to the same free-floating body (this is
the SC-1 climbing/servicing concept of the project), therefore every contact wrench that
the arm applies to a module is reacted by the base -- the central microgravity
complication that the guidance is meant to handle.

Joint types
-----------
``bolt``   the fastener must be *unscrewed* first (rotate about the bolt axis by
           ``required_twist``) before it can be pulled out; pulling first just torques the
           base.
``latch``  a spring latch must be *pressed in* first (``press_depth``) and can then be
           pulled.
``peg``    a friction peg: it releases once the axial pull exceeds ``breakaway_force``
           (no twist), which is the hardest one for momentum management.

Action space (normalised to ``[-1, 1]``, dimension 7)
-----------------------------------------------------
``a[0:3]``  end-effector translation command in the *base* frame
``a[3:6]``  end-effector rotation command (rotation vector) in the *base* frame
``a[6]``    gripper, ``> 0`` closes / grasps

The command is interpreted as a *displacement intent*; the environment turns it into an
impedance force ``F = k_imp * dp`` which is what generates contact forces and base
reaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class JointSpec:
    """Mechanical fastener connecting a module to the satellite body."""

    kind: str  # bolt | latch | peg
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    required_twist: float = 0.0  # rad, about `axis`, before extraction is allowed
    breakaway_force: float = 5.0  # N, axial pull needed to release
    press_depth: float = 0.0  # m, inward travel needed first (latch)
    travel: float = 0.06  # m, axial travel to fully extract the module

    def __post_init__(self):
        assert self.kind in ("bolt", "latch", "peg"), self.kind
        n = float(np.linalg.norm(self.axis))
        object.__setattr__(self, "axis", tuple(float(a) / n for a in self.axis))

    @property
    def axis_np(self) -> np.ndarray:
        return np.array(self.axis, dtype=np.float64)


@dataclass(frozen=True)
class ModuleSpec:
    """A serviceable module (payload box, solar panel, battery ...)."""

    name: str
    size: Tuple[float, float, float]  # full extents [m]
    mass: float  # kg
    attach_pos: Tuple[float, float, float]  # mount point in the base frame [m]
    joint: JointSpec
    capture_pos: Tuple[float, float, float]  # staging pose in the base frame [m]
    priority: int = 0  # higher = more valuable (used for ordered/multi-objective tasks)
    grasp_face: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # grasp offset from attach

    @property
    def attach_np(self) -> np.ndarray:
        return np.array(self.attach_pos, dtype=np.float64)

    @property
    def capture_np(self) -> np.ndarray:
        return np.array(self.capture_pos, dtype=np.float64)

    @property
    def grasp_pose(self) -> np.ndarray:
        """Grasp point in the base frame: mount point plus the face offset."""
        return self.attach_np + np.array(self.grasp_face, dtype=np.float64)


@dataclass
class SatelliteSpec:
    """Full description of the simulation + task."""

    name: str = "sc1_dead_satellite"
    base_mass: float = 28.27  # kg, SC-1 concept total dry mass
    base_inertia: Tuple[float, float, float] = (0.62, 0.58, 0.41)  # kg m^2, diagonal
    base_size: Tuple[float, float, float] = (0.62, 0.38, 0.21)  # m
    modules: List[ModuleSpec] = field(default_factory=list)

    ee_start: Tuple[float, float, float] = (0.0, 0.0, 0.30)
    dp_scale: float = 0.02  # m per unit action
    drot_scale: float = 0.20  # rad per unit action
    k_imp: float = 900.0  # impedance stiffness [N/m] (displacement -> force)
    force_limit: float = 8.0  # N, hard abort threshold on the contact force
    force_budget: float = 4.0  # N, operational budget scored by ``constraint_satisfied``
    grasp_tol: float = 0.030  # m
    capture_tol: float = 0.030  # m
    twist_tol: float = 0.22  # rad tolerance on the required twist (thread backlash)
    attitude_budget: float = 0.12  # rad allowed base attitude drift per episode
    horizon: int = 120
    require_stow: bool = False  # if True the module must also reach its staging pose
    dt: float = 0.05  # s

    def module(self, name: str) -> ModuleSpec:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(name)

    @property
    def module_names(self) -> List[str]:
        return [m.name for m in self.modules]


# ------------------------------------------------------------------ scenes ----
def make_sc1_spec(
    base_mass: float = 28.27,
    base_inertia: Tuple[float, float, float] = (0.62, 0.58, 0.41),
) -> SatelliteSpec:
    """The three-module benchmark scene used in ``scripts/satellite``."""

    modules = [
        ModuleSpec(
            name="payload_box",
            size=(0.16, 0.12, 0.10),
            mass=3.2,
            attach_pos=(0.22, 0.00, 0.10),
            joint=JointSpec(
                kind="bolt",
                axis=(1.0, 0.0, 0.0),
                required_twist=0.5 * np.pi,  # quarter turn
                breakaway_force=4.0,
                travel=0.09,
            ),
            capture_pos=(0.34, -0.16, 0.16),
            priority=2,
            grasp_face=(-0.09, 0.0, 0.0),
        ),
        ModuleSpec(
            name="solar_panel",
            size=(0.30, 0.10, 0.02),
            mass=1.4,
            attach_pos=(-0.20, 0.12, 0.04),
            joint=JointSpec(
                kind="latch",
                axis=(-1.0, 0.0, 0.0),
                required_twist=0.0,
                breakaway_force=2.0,
                press_depth=0.006,
                travel=0.07,
            ),
            capture_pos=(-0.30, 0.26, 0.10),
            priority=1,
            grasp_face=(0.08, 0.0, 0.0),
        ),
        ModuleSpec(
            name="battery_module",
            size=(0.12, 0.14, 0.12),
            mass=2.6,
            attach_pos=(-0.05, -0.18, 0.02),
            joint=JointSpec(
                kind="peg",
                axis=(0.0, -1.0, 0.0),
                required_twist=0.0,
                breakaway_force=7.0,
                travel=0.10,
            ),
            capture_pos=(-0.10, -0.30, 0.12),
            priority=3,
            grasp_face=(0.0, 0.09, 0.0),
        ),
    ]
    return SatelliteSpec(
        base_mass=base_mass, base_inertia=base_inertia, modules=modules
    )


def make_single_module_spec(name: str = "payload_box") -> SatelliteSpec:
    """Ablation scene with a single module (used by the difficulty ablation)."""
    spec = make_sc1_spec()
    keep = [m for m in spec.modules if m.name == name]
    if not keep:
        raise KeyError(name)
    spec.modules = keep
    return spec


TASK_NAMES = ("payload_box", "solar_panel", "battery_module")


def task_success_metrics(info: Dict) -> Dict[str, float]:
    """Convert a rollout ``info`` dict into the scalar metrics used in the tables."""
    return {
        "success": float(info.get("success", False)),
        "detached": float(info.get("detached", False)),
        "captured": float(info.get("captured", False)),
        "force_violation": float(info.get("max_force", 0.0) > info.get("force_limit", 1e9)),
        "attitude_drift": float(info.get("attitude_drift", 0.0)),
        "max_force": float(info.get("max_force", 0.0)),
        "steps": float(info.get("steps", 0)),
    }

