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

"""
This script demonstrates policy inference in a prebuilt USD environment for TienKung robot.

In this example, we use a locomotion policy to control the TienKung robot. The robot was trained
using the walk task. The robot is commanded to move forward at a constant velocity.

Usage:
    # Run with default warehouse USD environment
    python legged_lab/scripts/usd_policy_infer.py --task walk --policy_path /path/to/exported/policy.pt

    # Run with custom USD environment
    python legged_lab/scripts/usd_policy_infer.py --task walk --policy_path /path/to/exported/policy.pt --usd_path /path/to/custom.usd

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

# add argparse arguments
parser = argparse.ArgumentParser(description="Policy inference for TienKung robot in a USD environment.")
parser.add_argument("--task", type=str, default="walk", help="Name of the task.")
parser.add_argument("--policy_path", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument("--usd_path", type=str, default=None, help="Path to custom USD environment file.")
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
import io
import os
import torch

import omni

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from legged_lab.envs import *  # noqa:F401, F403
from pxr import Usd


def remove_usd_robots(stage):
    """
    Remove all robots from the USD stage (except those under /World/envs).
    
    This is useful when the USD file contains pre-existing robot models
    that you want to remove before spawning your own robot.
    
    Args:
        stage: The USD stage object
    """
    
    # List of common robot prim paths to remove
    robot_paths_to_check = [
        "/World/walkers1",
    ]
    
    removed_count = 0
    for prim_path in robot_paths_to_check:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            stage.RemovePrim(prim_path)
            print(f"[INFO] Removed prim at {prim_path}")
            removed_count += 1
    
    if removed_count == 0:
        print("[INFO] No pre-existing robots found in USD scene")


def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.policy_path)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(file, map_location=device)
    print(f"[INFO] Loaded policy from: {policy_path}")

    # get environment configuration
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # modify configuration for USD environment inference
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 1000.0  # Long episode for demo
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (0.8, 0.8)  # Forward velocity
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)

    # set terrain to USD or default warehouse
    if args_cli.usd_path is not None:
        usd_path = os.path.abspath(args_cli.usd_path)
        print(f"[INFO] Using custom USD environment: {usd_path}")
    else:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"
        print(f"[INFO] Using default warehouse USD environment: {usd_path}")

    # override terrain configuration for USD environment
    env_cfg.scene.terrain_type = "usd"
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.usd_path = usd_path

    # disable height scanner for USD environment (mesh not available)
    env_cfg.scene.height_scanner.enable_height_scan = False

    if args_cli.seed is not None:
        env_cfg.scene.seed = args_cli.seed

    # set device
    env_cfg.device = device

    # IMPORTANT: Remove robots from USD BEFORE creating environment
    # This prevents PhysX from creating tensor views for prims that will be deleted
    stage = Usd.Stage.Open(usd_path)
    remove_usd_robots(stage)
    # Save the modified USD to a temporary file
    import tempfile
    temp_usd = tempfile.NamedTemporaryFile(suffix=".usd", delete=False)
    temp_usd_path = temp_usd.name
    temp_usd.close()
    stage.Export(temp_usd_path)
    print(f"[INFO] Saved cleaned USD to temporary file: {temp_usd_path}")
    
    # Update config to use the cleaned USD
    env_cfg.scene.usd_path = temp_usd_path

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
