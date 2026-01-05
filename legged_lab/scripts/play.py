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

import argparse
import os

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOffPolicyRunner, AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def _export_sac_policy(policy, normalizer, export_dir):
    """Export SAC policy to JIT and ONNX formats.
    
    SAC uses a custom SACActor module that is not subscriptable (unlike nn.Sequential),
    so we need custom export logic to handle the input dimension discovery.
    
    Args:
        policy: The SAC Actor-Critic policy module
        normalizer: The observation normalizer (or None)
        export_dir: Directory to save exported models
    """
    import copy
    
    os.makedirs(export_dir, exist_ok=True)
    
    # ===== Export as JIT =====
    class SACPolicyExporter(torch.nn.Module):
        def __init__(self, actor, normalizer):
            super().__init__()
            self.actor = copy.deepcopy(actor)
            self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()
        
        def forward(self, x):
            return self.actor(self.normalizer(x))
        
        @torch.jit.export
        def reset(self):
            pass
    
    exporter = SACPolicyExporter(policy.actor, normalizer)
    exporter.to("cpu")
    
    jit_path = os.path.join(export_dir, "policy.pt")
    traced_script_module = torch.jit.script(exporter)
    traced_script_module.save(jit_path)
    print(f"[INFO] Exported SAC policy as JIT to: {jit_path}")
    
    # ===== Export as ONNX =====
    # Get input dimension from SACActor's backbone
    # SACActor.backbone is nn.Sequential, first layer is Linear
    input_dim = policy.actor.backbone[0].in_features
    
    obs = torch.zeros(1, input_dim)
    onnx_path = os.path.join(export_dir, "policy.onnx")
    
    torch.onnx.export(
        exporter,
        obs,
        onnx_path,
        export_params=True,
        opset_version=11,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] Exported SAC policy as ONNX to: {onnx_path}")


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner | AmpOffPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    # Check if this is a SAC policy (SACActor is not subscriptable, need special handling)
    is_sac_policy = hasattr(runner.alg.policy, 'actor') and hasattr(runner.alg.policy.actor, 'backbone')
    
    if is_sac_policy:
        # SAC policy: use custom export since isaaclab_rl's exporter expects nn.Sequential
        _export_sac_policy(runner.alg.policy, runner.obs_normalizer, export_model_dir)
    else:
        # PPO policy: use standard isaaclab_rl exporter
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
