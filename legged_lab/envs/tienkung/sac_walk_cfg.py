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

"""Pure SAC Walk Task Configuration.

A standard SAC configuration without AMP (Adversarial Motion Priors).

This configuration is useful for:
- Baseline comparisons without imitation learning
- Tasks where you don't have motion capture data
- Pure reinforcement learning experiments

Key hyperparameters (based on SAC paper):
- Replay memory size: 10^6
- Batch size: 256
- Updates per step: 1
- n-step return: 1
- Target smoothing coefficient τ: 0.005
- Discount factor γ: 0.99
- Actor/Critic learning rate: 3e-4
- Actor/Critic hidden layers: [256, 256] or [1024, 512]
"""

import math
from dataclasses import dataclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


@configclass
class GaitCfg:
    """Gait configuration for walking."""
    gait_air_ratio_l: float = 0.5
    gait_air_ratio_r: float = 0.5
    gait_phase_offset_l: float = 0.5
    gait_phase_offset_r: float = 0.0
    gait_cycle: float = 0.6


@configclass
class SACRewardCfg:
    """Reward configuration for pure SAC walk task.
    
    Note: Without AMP, we rely more heavily on hand-crafted rewards
    to achieve natural-looking locomotion.
    """
    # Core locomotion rewards
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5})
    
    # Stability rewards
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-2.0
    )
    
    # Energy efficiency
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # Contact rewards
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Foot rewards
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    
    # Joint rewards
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )
    
    # Gait rewards (important for natural locomotion without AMP)
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.5)
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.5)
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=1.0)
    
    # Action smoothness (more important without AMP)
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.001)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.002)
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.5)
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.5)
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.5)


@configclass
class TienKungSACWalkFlatEnvCfg:
    """Environment configuration for pure SAC Walk task."""
    
    # Required by TienKungEnv for visualization (even without AMP training)
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
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    reward = SACRewardCfg()
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
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5), 
            ang_vel_z=(-1.57, 1.57), 
            heading=(-math.pi, math.pi)
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
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass 
class RslRlPureSacActorCriticCfg:
    """Configuration for pure SAC Actor-Critic network."""
    class_name: str = "SACActorCritic"
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    activation: str = "gelu"
    init_noise_std: float = 1.0
    action_scale: float = 1.0
    
    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 256]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256]


@configclass
class RslRlPureSacAlgorithmCfg:
    """Configuration for pure SAC algorithm (without AMP).
    
    Standard SAC hyperparameters for locomotion.
    """
    class_name: str = "SAC"
    
    # SAC core parameters
    sac_replay_buffer_size: int = 1_000_000  # 10^6
    batch_size: int = 256
    updates_per_step: int = 1
    n_step_return: int = 1
    tau: float = 0.005  # Target smoothing coefficient
    gamma: float = 0.99  # Discount factor
    actor_lr: float = 3e-4  # Actor learning rate
    critic_lr: float = 3e-4  # Critic learning rate
    alpha_lr: float = 3e-4  # Alpha learning rate
    init_alpha: float = 0.2
    auto_alpha: bool = True  # Automatic alpha tuning
    max_grad_norm: float = 1.0
    warm_up_steps: int = 1000  # Larger warm-up for pure RL


@configclass
class TienKungSACWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    """Agent configuration for pure SAC Walk task (without AMP)."""
    
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24  # Collection steps per iteration
    max_iterations = 50000
    empirical_normalization = False
    
    # SAC Actor-Critic policy
    policy = RslRlPureSacActorCriticCfg()
    
    # SAC algorithm (without AMP)
    algorithm = RslRlPureSacAlgorithmCfg()
    
    clip_actions = None
    save_interval = 100
    runner_class_name = "OffPolicyRunner"  # Pure off-policy runner (no AMP)
    experiment_name = "sac_walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "sac_walk"
    wandb_project = "sac_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
