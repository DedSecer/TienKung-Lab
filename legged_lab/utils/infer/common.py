# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
Common utilities shared between usd_policy_infer.py and usd_policy_infer_ros2.py.

This module contains:
- Policy loading from JIT-exported checkpoints
- Environment configuration for USD-based inference
- USD stage cleanup (removing pre-existing robots)
"""

import io
import os
import tempfile

import omni
import torch
from pxr import Usd


def load_policy(policy_path: str, device: str = None):
    """
    Load a JIT-exported policy model.

    Args:
        policy_path: Path to the exported policy .pt file.
        device: Target device (e.g. "cuda:0" or "cpu"). If None, auto-detects.

    Returns:
        Tuple of (policy, device) where policy is the loaded torch.jit model.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    policy_path = os.path.abspath(policy_path)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=device)
    print(f"[INFO] Loaded policy from: {policy_path}")

    return policy, device


def configure_env_for_usd(env_cfg, usd_path: str, num_envs: int = 1, seed: int = None,
                          device: str = None):
    """
    Configure environment for USD-based policy inference.

    Modifies env_cfg in-place to disable noise, domain randomization,
    set terrain to USD mode, and disable height scanning.

    Args:
        env_cfg: Environment configuration object.
        usd_path: Absolute path to the USD environment file.
        num_envs: Number of environments to simulate.
        seed: Random seed (optional).
        device: Target device (optional).
    """
    # Disable noise and domain randomization for inference
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 1000.0  # Long episode for demo
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0

    # Override terrain configuration for USD environment
    env_cfg.scene.terrain_type = "usd"
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.usd_path = usd_path

    # Disable height scanner for USD environment (mesh not available)
    env_cfg.scene.height_scanner.enable_height_scan = False

    if seed is not None:
        env_cfg.scene.seed = seed

    if device is not None:
        env_cfg.device = device


def remove_usd_robots(stage):
    """
    Remove all robots from the USD stage (except those under /World/envs).

    This is useful when the USD file contains pre-existing robot models
    that you want to remove before spawning your own robot.

    Args:
        stage: The USD stage object.
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


def prepare_usd_stage(usd_path: str) -> str:
    """
    Open a USD file, remove pre-existing robots, and export to a temporary file.

    Args:
        usd_path: Absolute path to the original USD file.

    Returns:
        Path to the cleaned temporary USD file.
    """
    usd_path = os.path.abspath(usd_path)
    print(f"[INFO] Using USD environment: {usd_path}")

    stage = Usd.Stage.Open(usd_path)
    remove_usd_robots(stage)

    # Save the modified USD to a temporary file
    temp_usd = tempfile.NamedTemporaryFile(suffix=".usd", delete=False)
    temp_usd_path = temp_usd.name
    temp_usd.close()
    stage.Export(temp_usd_path)
    print(f"[INFO] Saved cleaned USD to temporary file: {temp_usd_path}")

    return temp_usd_path
