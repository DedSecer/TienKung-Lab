# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
# Pure PPO configuration (without AMP)
from legged_lab.envs.tienkung.ppo_walk_cfg import TienKungPPOWalkAgentCfg
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
# SAC+AMP configurations
from legged_lab.envs.tienkung.sac_amp_walk_cfg import (
    TienKungSACWalkAgentCfg,
    TienKungSACWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.sac_amp_run_cfg import (
    TienKungSACRunAgentCfg,
    TienKungSACRunFlatEnvCfg,
)
# Pure SAC configuration (without AMP)
from legged_lab.envs.tienkung.sac_walk_cfg import (
    TienKungSACWalkAgentCfg as TienKungPureSACWalkAgentCfg,
    TienKungSACWalkFlatEnvCfg as TienKungPureSACWalkFlatEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

# Register PPO-based tasks
task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
# Register Pure PPO task (without AMP) 
task_registry.register("ppo_walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungPPOWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)

# Register SAC+AMP tasks (based on paper: "Unlocking the Potential of Soft Actor-Critic for Imitation Learning")
task_registry.register("sac_amp_walk", TienKungEnv, TienKungSACWalkFlatEnvCfg(), TienKungSACWalkAgentCfg())
task_registry.register("sac_amp_run", TienKungEnv, TienKungSACRunFlatEnvCfg(), TienKungSACRunAgentCfg())

# Register Pure SAC tasks (without AMP, standard off-policy RL)
task_registry.register("sac_walk", TienKungEnv, TienKungPureSACWalkFlatEnvCfg(), TienKungPureSACWalkAgentCfg())
