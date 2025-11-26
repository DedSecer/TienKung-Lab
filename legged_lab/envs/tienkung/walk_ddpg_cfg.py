# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
DDPG Configuration for TienKung Walk Environment

This configuration file defines the settings for training a humanoid robot
to walk using the DDPG (Deep Deterministic Policy Gradient) algorithm.
"""

import math
from dataclasses import asdict

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.envs.tienkung.walk_cfg import GaitCfg
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


@configclass
class DDPGRewardCfg:
    """Simplified reward configuration for DDPG training.
    
    DDPG tends to work better with denser, simpler reward signals.
    """
    # Velocity tracking (main objectives)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5}
    )
    
    # Stability rewards
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    
    # Energy efficiency
    energy = RewTerm(func=mdp.energy, weight=-5e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    
    # Safety constraints
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    
    # Contact rewards
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class DDPGPolicyCfg:
    """DDPG Actor-Critic network configuration."""
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "relu"
    action_scale: float = 0.25  # Same as PPO action_scale


@configclass
class DDPGAlgorithmCfg:
    """DDPG algorithm hyperparameters."""
    # Replay buffer
    buffer_size: int = 1000000
    batch_size: int = 256
    
    # Learning rates
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    
    # RL parameters
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    
    # Exploration noise
    exploration_noise_std: float = 0.1
    exploration_noise_clip: float = 0.3
    
    # TD3 improvements
    policy_noise_std: float = 0.2
    policy_noise_clip: float = 0.5
    policy_update_freq: int = 2
    
    # Warmup
    warmup_steps: int = 10000


@configclass
class TienKungWalkDDPGEnvCfg:
    """Environment configuration for DDPG training."""
    amp_motion_files_display = ["legged_lab/envs/tienkung/datasets/motion_visualization/walk.txt"]
    device: str = "cuda:0"
    
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=TIENKUNG2LITE_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    
    reward = DDPGRewardCfg()
    gait = GaitCfg()
    
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )
    
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )


@configclass
class TienKungWalkDDPGAgentCfg:
    """DDPG agent configuration for training."""
    seed: int = 42
    device: str = "cuda:0"
    
    # Training parameters
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    
    # Observation normalization
    empirical_normalization: bool = False
    
    # Policy network configuration
    policy: DDPGPolicyCfg = DDPGPolicyCfg()
    
    # Algorithm configuration
    algorithm: DDPGAlgorithmCfg = DDPGAlgorithmCfg()
    
    # Runner configuration
    runner_class_name: str = "DDPGRunner"
    experiment_name: str = "walk_ddpg"
    run_name: str = ""
    
    # Logging
    logger: str = "tensorboard"
    neptune_project: str = "walk_ddpg"
    wandb_project: str = "walk_ddpg"
    
    # Resume training
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for runner compatibility."""
        return {
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "empirical_normalization": self.empirical_normalization,
            "runner_class_name": self.runner_class_name,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
            "policy": {
                "actor_hidden_dims": list(self.policy.actor_hidden_dims),
                "critic_hidden_dims": list(self.policy.critic_hidden_dims),
                "activation": self.policy.activation,
                "action_scale": self.policy.action_scale,
            },
            "algorithm": {
                "buffer_size": self.algorithm.buffer_size,
                "batch_size": self.algorithm.batch_size,
                "actor_learning_rate": self.algorithm.actor_learning_rate,
                "critic_learning_rate": self.algorithm.critic_learning_rate,
                "gamma": self.algorithm.gamma,
                "tau": self.algorithm.tau,
                "exploration_noise_std": self.algorithm.exploration_noise_std,
                "exploration_noise_clip": self.algorithm.exploration_noise_clip,
                "policy_noise_std": self.algorithm.policy_noise_std,
                "policy_noise_clip": self.algorithm.policy_noise_clip,
                "policy_update_freq": self.algorithm.policy_update_freq,
                "warmup_steps": self.algorithm.warmup_steps,
            },
        }
