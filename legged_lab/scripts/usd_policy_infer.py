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

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# add argparse arguments
parser = argparse.ArgumentParser(description="Policy inference for TienKung robot in a USD environment.")
parser.add_argument("--task", type=str, default="walk", help="Name of the task.")
parser.add_argument("--policy_path", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--usd_path", type=str, default="../sense/museum/museum.usd", help="Path to custom USD environment file (default: ../sense/museum/museum.usd).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import torch

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.infer.common import load_policy, configure_env_for_usd, prepare_usd_stage


def main():
    """Main function."""
    # load the trained jit policy
    policy, device = load_policy(args_cli.policy_path)

    # get environment configuration
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # prepare cleaned USD stage
    usd_path = os.path.abspath(args_cli.usd_path)
    temp_usd_path = prepare_usd_stage(usd_path)

    # configure environment for USD inference
    configure_env_for_usd(env_cfg, temp_usd_path, num_envs=args_cli.num_envs,
                          seed=args_cli.seed, device=device)

    # set forward velocity commands
    env_cfg.commands.ranges.lin_vel_x = (0.8, 0.8)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)

    # create environment
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    print(f"[INFO] Created environment: {env_class_name}")

    # setup keyboard control if not headless
    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841
        print("[INFO] Keyboard control enabled. Use arrow keys to control the robot.")

    # run inference with the policy
    obs, _ = env.get_observations()
    print("[INFO] Starting policy inference...")

    with torch.inference_mode():
        while simulation_app.is_running():
            action = policy(obs)
            obs, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
