# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
TienKung velocity tracking configuration using unitree_rl_lab style rewards and observations.

This configuration follows the unitree_rl_lab approach:
- Standard PPO (without AMP)
- Gait phase observations
- Curriculum learning for velocity commands
- Simplified reward design

Usage:
    python legged_lab/scripts/train.py --task unitree_style_walk --num_envs 4096 --headless
    python legged_lab/scripts/play.py --task unitree_style_walk
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

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


# ===========================
# Gait Configuration (similar to unitree_rl_lab)
# ===========================

@configclass
class GaitCfg:
    """Gait parameters for periodic walking pattern."""
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.6  # Gait cycle period in seconds (unitree_rl_lab uses 0.6)


# ===========================
# Reward Configuration (unitree_rl_lab style)
# ===========================

@configclass
class UnitreeStyleRewardCfg:
    """
    Reward configuration following unitree_rl_lab design philosophy.
    
    Key differences from original TienKung-Lab:
    - Simpler reward structure
    - Gait-based periodic rewards
    - No AMP discriminator rewards
    - Emphasis on tracking and stability
    """
    
    # -- Tracking rewards (primary objectives)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, 
        weight=1.0, 
        params={"std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.5, 
        params={"std": math.sqrt(0.25)}
    )
    
    # -- Base motion penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    
    # -- Joint penalties
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    
    # -- Posture penalties (humanoid specific)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, 
        weight=-2.0
    )
    
    # -- Joint deviation penalties (keep arms stable)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pitch_.*_joint",
                    "shoulder_roll_.*_joint",
                    "shoulder_yaw_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_yaw_.*_joint", "hip_roll_.*_joint"])},
    )
    
    # -- Feet rewards (unitree_rl_lab gait style)
    gait_feet_frc_perio = RewTerm(
        func=mdp.gait_feet_frc_perio, 
        weight=0.5, 
        params={"delta_t": 0.02}
    )
    gait_feet_spd_perio = RewTerm(
        func=mdp.gait_feet_spd_perio, 
        weight=0.5, 
        params={"delta_t": 0.02}
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    
    # -- Contact penalties
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
    
    # -- Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


# ===========================
# Environment Configuration
# ===========================

@configclass
class TienKungUnitreeStyleEnvCfg:
    """
    Environment configuration using unitree_rl_lab training approach.
    
    Key features:
    - Standard PPO training (no AMP)
    - Gait phase in observations
    - Curriculum learning for velocity commands
    - Domain randomization for sim-to-real
    """
    
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
        actor_obs_history_length=1,  # Single frame (unitree_rl_lab style, no history stacking)
        critic_obs_history_length=1,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    
    reward = UnitreeStyleRewardCfg()
    gait = GaitCfg()
    
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=0.2,  # Scale down angular velocity (unitree_rl_lab style)
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=0.05,  # Scale down joint velocity (unitree_rl_lab style)
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,  # Less standing (unitree_rl_lab style)
        rel_heading_envs=1.0,
        heading_command=False,  # No heading command (unitree_rl_lab style)
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            # Start with small range, curriculum will expand
            lin_vel_x=(-0.1, 0.1), 
            lin_vel_y=(-0.1, 0.1), 
            ang_vel_z=(-0.1, 0.1), 
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
                    "static_friction_range": (0.3, 1.0),  # Wider range (unitree_rl_lab style)
                    "dynamic_friction_range": (0.3, 1.0),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-1.0, 3.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (0.0, 0.0),  # Zero initial velocity (unitree_rl_lab style)
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (1.0, 1.0),  # Reset to default (unitree_rl_lab style)
                    "velocity_range": (-1.0, 1.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(5.0, 5.0),  # More frequent pushes
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


# ===========================
# Agent Configuration (Standard PPO, no AMP)
# ===========================

@configclass
class TienKungUnitreeStyleAgentCfg(RslRlOnPolicyRunnerCfg):
    """
    PPO agent configuration following unitree_rl_lab approach.
    
    Key differences from original TienKung-Lab:
    - Uses standard OnPolicyRunner (not AmpOnPolicyRunner)
    - No discriminator, no motion files
    - Standard PPO algorithm
    """
    
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",  # Standard PPO, not AMPPPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )
    
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"  # Standard runner, not AmpOnPolicyRunner
    experiment_name = "unitree_style_walk"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
