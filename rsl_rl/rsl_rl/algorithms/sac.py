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

"""Pure SAC Algorithm Implementation.

This is a standard Soft Actor-Critic (SAC) implementation without AMP.
Based on the paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
https://arxiv.org/abs/1801.01290

Key components:
- Off-policy SAC with entropy regularization
- Automatic temperature (alpha) tuning
- Twin Q-networks with soft target updates
- n-step returns for improved learning

Hyperparameters (recommended):
- Replay memory size: 10^6 - 10^7
- Batch size: 256 - 16384
- Updates per step: 1 - 8
- n-step return: 1 - 5
- Target smoothing coefficient τ: 0.005 - 0.05
- Discount factor γ: 0.99
- Actor learning rate: 3e-4 - 1e-3
- Critic learning rate: 3e-4 - 1e-3
- Initial alpha: 0.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules.sac_actor_critic import SACActorCritic
from rsl_rl.storage.sac_replay_buffer import SACReplayBuffer


class SAC:
    """Pure Soft Actor-Critic Algorithm.
    
    A standard off-policy maximum entropy reinforcement learning algorithm
    that learns a stochastic policy to maximize both the expected return
    and the entropy of the policy.
    
    Key features:
    - Off-policy learning with replay buffer
    - Entropy regularization for exploration
    - Automatic temperature adjustment
    - Twin Q-networks to reduce overestimation
    - n-step returns for faster credit assignment
    """

    policy: SACActorCritic
    """The SAC actor-critic module."""

    def __init__(
        self,
        policy: SACActorCritic,
        sac_replay_buffer_size: int = 1_000_000,
        min_std=None,
        # SAC parameters
        batch_size: int = 256,
        updates_per_step: int = 1,
        n_step_return: int = 1,
        tau: float = 0.005,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        target_entropy: float = None,
        auto_alpha: bool = True,
        max_grad_norm: float = 1.0,
        warm_up_steps: int = 1000,
        # Device
        device: str = "cpu",
        # Multi-GPU
        multi_gpu_cfg: dict | None = None,
    ):
        """Initialize SAC algorithm.
        
        Args:
            policy: SAC Actor-Critic network
            sac_replay_buffer_size: Size of replay buffer
            min_std: Minimum standard deviation for policy
            batch_size: SAC batch size
            updates_per_step: Number of gradient updates per env step
            n_step_return: n-step return for TD learning
            tau: Target network soft update coefficient
            gamma: Discount factor
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            alpha_lr: Temperature learning rate
            init_alpha: Initial temperature value
            target_entropy: Target entropy (default: -dim(action))
            auto_alpha: Whether to automatically tune alpha
            max_grad_norm: Maximum gradient norm for clipping
            warm_up_steps: Warm-up steps before training starts
            device: Compute device
            multi_gpu_cfg: Multi-GPU configuration
        """
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        
        # Multi-GPU configuration
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # ===== Policy (SAC Actor-Critic) =====
        self.policy = policy
        self.policy.to(self.device)
        self.min_std = min_std

        # ===== SAC Parameters =====
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.n_step_return = n_step_return
        self.tau = tau
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.warm_up_steps = warm_up_steps
        self.training_step = 0

        # ===== Entropy Temperature (Alpha) =====
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -self.policy.num_actions  # Default: -dim(A)
        self.target_entropy = target_entropy
        
        # Learnable log_alpha for automatic tuning
        self.log_alpha = torch.tensor(
            [torch.log(torch.tensor(init_alpha))], 
            dtype=torch.float32, 
            device=self.device, 
            requires_grad=True
        )
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=alpha_lr)

        # ===== Optimizers =====
        # Actor optimizer
        actor_params = list(self.policy.actor.backbone.parameters()) + \
                       list(self.policy.actor.mean_layer.parameters()) + \
                       list(self.policy.actor.log_std_layer.parameters())
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr)
        
        # Critic optimizer
        critic_params = list(self.policy.q1.parameters()) + list(self.policy.q2.parameters())
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr)
        
        # Learning rate (for logging compatibility)
        self.learning_rate = actor_lr

        # ===== SAC Replay Buffer =====
        self.sac_storage: SACReplayBuffer = None  # Initialized in init_storage
        self.sac_replay_buffer_size = sac_replay_buffer_size

        # ===== RND (placeholder for compatibility) =====
        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None

        # ===== Symmetry (placeholder for compatibility) =====
        self.symmetry = None

        # ===== Current transition (for collect step) =====
        self.current_obs = None
        self.current_privileged_obs = None
        self.current_actions = None

    def init_storage(
        self, 
        training_type, 
        num_envs, 
        num_transitions_per_env, 
        actor_obs_shape, 
        critic_obs_shape, 
        actions_shape
    ):
        """Initialize storage buffers.
        
        Args:
            training_type: Type of training ("rl")
            num_envs: Number of parallel environments
            num_transitions_per_env: Transitions per env per iteration
            actor_obs_shape: Shape of actor observations
            critic_obs_shape: Shape of critic observations
            actions_shape: Shape of actions
        """
        obs_dim = actor_obs_shape[0]
        action_dim = actions_shape[0]
        privileged_obs_dim = critic_obs_shape[0]
        
        # SAC replay buffer with proper parallel environment support
        self.sac_storage = SACReplayBuffer(
            num_envs=num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            buffer_size=self.sac_replay_buffer_size,
            device=self.device,
            n_step=self.n_step_return,
            gamma=self.gamma,
        )
        
        # Initialize privileged obs storage
        self.sac_storage.init_privileged_obs(privileged_obs_dim)

    def act(self, obs, critic_obs):
        """Sample action for environment interaction.
        
        Args:
            obs: Actor observations
            critic_obs: Critic observations
            
        Returns:
            actions: Sampled actions
        """
        # Store current observations for later
        self.current_obs = obs.clone()
        self.current_privileged_obs = critic_obs.clone()
        
        # Sample action from policy
        with torch.no_grad():
            actions = self.policy.act(obs)
        
        self.current_actions = actions.clone()
        return actions.detach()

    def process_env_step(self, rewards, dones, infos):
        """Process environment step and store transitions.
        
        Args:
            rewards: Rewards from environment
            dones: Done flags
            infos: Additional info dict
        """
        # This method is for compatibility with on-policy interface
        # For off-policy, we use store_transition directly
        pass

    def store_transition(
        self, 
        obs, 
        actions, 
        rewards, 
        next_obs, 
        dones,
        privileged_obs=None,
        next_privileged_obs=None,
    ):
        """Store a transition in the replay buffer.
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observations
            dones: Done flags
            privileged_obs: Current privileged observations
            next_privileged_obs: Next privileged observations
        """
        self.sac_storage.insert(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            dones=dones,
            privileged_observations=privileged_obs,
            next_privileged_observations=next_privileged_obs,
        )
        
        # Increment training step
        self.training_step += 1

    def compute_returns(self, last_critic_obs):
        """Placeholder for compatibility with on-policy interface."""
        pass

    def update(self):
        """Update policy and critics.
        
        Returns:
            loss_dict: Dictionary of losses for logging
        """
        # Check if we have enough samples
        if not self.sac_storage.is_ready(min_size=max(self.batch_size, self.warm_up_steps)):
            return {
                "value_function": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "alpha": self.alpha,
            }

        # Accumulate losses
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        mean_entropy = 0.0
        mean_q_value = 0.0

        # Multiple SAC updates per step
        for _ in range(self.updates_per_step):
            # ===== Sample from SAC replay buffer (with n-step returns) =====
            batch = self.sac_storage.sample(self.batch_size)
            obs, actions, n_step_rewards, next_obs, dones, n_step_gammas = batch[:6]
            if len(batch) > 6:
                privileged_obs, next_privileged_obs = batch[6], batch[7]
            else:
                privileged_obs, next_privileged_obs = obs, next_obs

            # ===== Update Critic (Q-networks) with n-step returns =====
            with torch.no_grad():
                # Sample next action and compute target Q
                next_actions, next_log_prob, _ = self.policy.sample_action(next_obs)
                q1_target, q2_target = self.policy.get_target_q_values(next_privileged_obs, next_actions)
                min_q_target = torch.min(q1_target, q2_target)
                # n-step Soft Q target: R_n + γ^n * (Q_target - α * log_prob)
                target_q = n_step_rewards + (1 - dones) * n_step_gammas * (min_q_target - self.alpha * next_log_prob)

            # Current Q estimates
            q1, q2 = self.policy.get_q_values(privileged_obs, actions)
            
            # Critic loss (MSE)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            # Update critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.q1.parameters()) + list(self.policy.q2.parameters()),
                self.max_grad_norm
            )
            self.critic_optimizer.step()

            # ===== Update Actor =====
            # Sample new actions for current states
            new_actions, log_prob, _ = self.policy.sample_action(obs)
            
            # Get Q-values for new actions
            q1_new, q2_new = self.policy.get_q_values(privileged_obs, new_actions)
            min_q_new = torch.min(q1_new, q2_new)
            
            # SAC actor loss: J_π(θ) = E[α·log_prob - Q]
            actor_loss = (self.alpha * log_prob - min_q_new).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.actor.backbone.parameters()) + 
                list(self.policy.actor.mean_layer.parameters()) + 
                list(self.policy.actor.log_std_layer.parameters()),
                self.max_grad_norm
            )
            self.actor_optimizer.step()

            # ===== Update Temperature (Alpha) =====
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
                total_alpha_loss += alpha_loss.item()

            # ===== Update Target Networks =====
            self.policy.soft_update_target(self.tau)

            # Accumulate for logging
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            mean_entropy += (-log_prob.mean()).item()
            mean_q_value += min_q_new.mean().item()

        # Average losses
        num_updates = self.updates_per_step
        loss_dict = {
            "value_function": total_critic_loss / num_updates,
            "surrogate": total_actor_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "q_value": mean_q_value / num_updates,
            "alpha": self.alpha,
        }

        return loss_dict

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs (placeholder)."""
        if not self.is_multi_gpu:
            return
        # TODO: Implement multi-GPU support for SAC
        pass

    def reduce_parameters(self):
        """Collect gradients from all GPUs (placeholder)."""
        if not self.is_multi_gpu:
            return
        # TODO: Implement multi-GPU support for SAC
        pass
