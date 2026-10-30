"""Free-floating satellite disassembly benchmark used by the flow-DynaGuide experiments."""

from core.satellite.env import (  # noqa: F401
    SatelliteEnv,
    proprio_dim,
    module_block,
    BASE_POS,
    BASE_QUAT,
    BASE_LINVEL,
    BASE_ANGVEL,
    EE_POS,
    EE_ROTVEC,
    GRIP,
)
from core.satellite.expert import (  # noqa: F401
    ScriptedExpert,
    collect_demonstrations,
    run_episode,
)
from core.satellite.spec import (  # noqa: F401
    JointSpec,
    ModuleSpec,
    SatelliteSpec,
    TASK_NAMES,
    make_sc1_spec,
    make_single_module_spec,
)

__all__ = [
    "SatelliteEnv",
    "ScriptedExpert",
    "collect_demonstrations",
    "run_episode",
    "JointSpec",
    "ModuleSpec",
    "SatelliteSpec",
    "TASK_NAMES",
    "make_sc1_spec",
    "make_single_module_spec",
    "proprio_dim",
    "module_block",
    "BASE_POS",
    "BASE_QUAT",
    "BASE_LINVEL",
    "BASE_ANGVEL",
    "EE_POS",
    "EE_ROTVEC",
    "GRIP",
]
