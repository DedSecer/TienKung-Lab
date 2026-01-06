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
Pure PPO training configuration for TienKung Walk task (without AMP).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .walk_cfg import TienKungWalkFlatEnvCfg  # Reuse the environment config


@configclass
class TienKungPPOWalkAgentCfg(RslRlOnPolicyRunnerCfg):
    """Pure PPO Agent Configuration (without AMP)."""
    
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
        class_name="PPO",  # 使用纯 PPO，而不是 AMPPPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"  # 使用纯 OnPolicyRunner，而不是 AmpOnPolicyRunner
    experiment_name = "ppo_walk"  # 区分实验名称
    run_name = ""
    logger = "tensorboard"
    neptune_project = "ppo_walk"
    wandb_project = "ppo_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    
    # PPO 不需要 AMP 相关参数
    min_normalized_std = [0.05] * 20
