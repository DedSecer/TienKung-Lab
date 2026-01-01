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

"""AMP+SAC Algorithm Implementation.

Based on the paper: "Unlocking the Potential of Soft Actor-Critic for Imitation Learning"
Key components:
- Off-policy SAC with entropy regularization
- AMP discriminator for imitation learning
- Automatic temperature tuning
- Twin Q-networks with soft target updates

Paper hyperparameters:
- Replay memory size: 10^7
- Batch size: 16384
- Updates per step: 8
- n-step return: 3
- Target smoothing coefficient τ: 0.05
- Discount factor γ: 0.99
- Actor learning rate: 0.001
- Actor hidden layers: [1024, 512]
- Critic hidden layers: [1024, 512]
- AMP loss coefficient λ_AMP: 0.1
- Gradient penalty coefficient λ_GP: 0.01
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules.sac_actor_critic import SACActorCritic
from rsl_rl.storage import ReplayBuffer
from rsl_rl.storage.sac_replay_buffer import SACReplayBuffer


class AMPSAC:
    """Soft Actor-Critic with Adversarial Motion Priors for Imitation Learning.
    
    Combines off-policy SAC with AMP discriminator for learning natural locomotion
    from motion capture data.
    
    Key features (from paper):
    - Off-policy learning with replay buffer
    - Entropy regularization for exploration
    - Automatic temperature adjustment
    - Twin Q-networks to reduce overestimation
    - AMP discriminator for style imitation
    """

    policy: SACActorCritic
    """The SAC actor-critic module."""

    def __init__(
        self,
        policy: SACActorCritic,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size: int = 100000,
        sac_replay_buffer_size: int = 10_000_000,
        min_std=None,
        # SAC parameters (from paper)
        batch_size: int = 16384,
        updates_per_step: int = 8,
        n_step_return: int = 3,
        tau: float = 0.05,
        gamma: float = 0.99,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        alpha_lr: float = 1e-3,
        init_alpha: float = 0.2,
        target_entropy: float = None,
        auto_alpha: bool = True,
        max_grad_norm: float = 1.0,
        warm_up_steps: int = 100,
        # AMP parameters (from paper)
        amp_loss_coef: float = 0.1,
        amp_grad_penalty_coef: float = 0.01,
        amp_batch_size: int = 8192,
        # Device
        device: str = "cpu",
        # Multi-GPU (not implemented for SAC yet)
        multi_gpu_cfg: dict | None = None,
    ):
        """Initialize AMP+SAC algorithm.
        
        Args:
            policy: SAC Actor-Critic network
            discriminator: AMP discriminator network
            amp_data: AMP motion data loader
            amp_normalizer: AMP observation normalizer
            amp_replay_buffer_size: Size of AMP replay buffer
            sac_replay_buffer_size: Size of SAC replay buffer (10^7)
            min_std: Minimum standard deviation for policy
            batch_size: SAC batch size (16384)
            updates_per_step: Number of gradient updates per env step (8)
            n_step_return: n-step return (3)
            tau: Target network soft update coefficient (0.05)
            gamma: Discount factor (0.99)
            actor_lr: Actor learning rate (0.001)
            critic_lr: Critic learning rate (0.001)
            alpha_lr: Temperature learning rate (0.001)
            init_alpha: Initial temperature value
            target_entropy: Target entropy (default: -dim(action))
            auto_alpha: Whether to automatically tune alpha
            max_grad_norm: Maximum gradient norm for clipping
            warm_up_steps: Warm-up steps before training starts
            amp_loss_coef: AMP loss coefficient λ_AMP (0.1)
            amp_grad_penalty_coef: Gradient penalty coefficient λ_GP (0.01)
            amp_batch_size: AMP batch size (8192)
            device: Compute device
            multi_gpu_cfg: Multi-GPU configuration (not implemented)
        """
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        
        # Multi-GPU (placeholder)
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
        # Actor optimizer (with discriminator encoder as per paper)
        actor_params = list(self.policy.actor_backbone.parameters()) + \
                       list(self.policy.actor_mean.parameters()) + \
                       list(self.policy.actor_log_std.parameters())
        self.actor_optimizer = optim.AdamW(actor_params, lr=actor_lr)
        
        # Critic optimizer
        critic_params = list(self.policy.q1.parameters()) + list(self.policy.q2.parameters())
        self.critic_optimizer = optim.AdamW(critic_params, lr=critic_lr)
        
        # Learning rate (for logging compatibility)
        self.learning_rate = actor_lr

        # ===== AMP Components =====
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_loss_coef = amp_loss_coef
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self.amp_batch_size = amp_batch_size
        
        # AMP replay buffer (for policy-generated transitions)
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        
        # AMP discriminator optimizer
        self.discriminator_optimizer = optim.AdamW(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 10e-4},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10e-2},
            ],
            lr=actor_lr
        )

        # ===== SAC Replay Buffer =====
        self.sac_storage: SACReplayBuffer = None  # Initialized in init_storage
        self.sac_replay_buffer_size = sac_replay_buffer_size

        # ===== RND (not implemented for SAC) =====
        self.rnd = None
        self.rnd_optimizer = None
        self.intrinsic_rewards = None

        # ===== Symmetry (not implemented for SAC) =====
        self.symmetry = None

        # ===== Current transition (for collect step) =====
        self.current_obs = None
        self.current_privileged_obs = None
        self.current_actions = None
        self.current_amp_obs = None

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
        
        # SAC replay buffer
        self.sac_storage = SACReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            buffer_size=self.sac_replay_buffer_size,
            device=self.device,
            n_step=self.n_step_return,
            gamma=self.gamma,
        )
        
        # Initialize privileged obs storage
        self.sac_storage.init_privileged_obs(privileged_obs_dim)

    def act(self, obs, critic_obs, amp_obs):
        """Sample action for environment interaction.
        
        Args:
            obs: Actor observations
            critic_obs: Critic observations
            amp_obs: AMP observations for discriminator
            
        Returns:
            actions: Sampled actions
        """
        # Store current observations for later
        self.current_obs = obs.clone()
        self.current_privileged_obs = critic_obs.clone()
        self.current_amp_obs = amp_obs.clone()
        
        # Sample action from policy
        with torch.no_grad():
            actions = self.policy.act(obs)
        
        self.current_actions = actions.clone()
        return actions.detach()

    def process_env_step(self, rewards, dones, infos, amp_obs):
        """Process environment step and store transitions.
        
        Args:
            rewards: Rewards from environment
            dones: Done flags
            infos: Additional info dict
            amp_obs: Next AMP observations
        """
        # Add to SAC replay buffer
        # Note: rewards already include AMP reward from discriminator
        self.sac_storage.insert(
            observations=self.current_obs,
            actions=self.current_actions,
            rewards=rewards,
            next_observations=infos.get("next_obs", self.current_obs),  # Will be updated
            dones=dones,
            privileged_observations=self.current_privileged_obs,
            next_privileged_observations=self.current_privileged_obs,  # Will be updated
        )
        
        # Add to AMP replay buffer
        self.amp_storage.insert(self.current_amp_obs, amp_obs)
        
        # Increment training step
        self.training_step += 1

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
        """Directly store a transition (alternative to process_env_step).
        
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

    def compute_returns(self, last_critic_obs):
        """Placeholder for compatibility with on-policy interface."""
        pass

    def update(self):
        """Update policy, critics, and discriminator.
        
        Returns:
            loss_dict: Dictionary of losses for logging
        """
        # Check if we have enough samples
        if not self.sac_storage.is_ready(min_size=max(self.batch_size, self.warm_up_steps)):
            return {
                "value_function": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "amp": 0.0,
                "amp_grad_pen": 0.0,
                "amp_policy_pred": 0.0,
                "amp_expert_pred": 0.0,
                "alpha": self.alpha,
            }

        # Accumulate losses
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        total_amp_loss = 0.0
        total_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_entropy = 0.0

        # Multiple updates per step (paper: 8 epochs)
        for _ in range(self.updates_per_step):
            # ===== Sample from SAC replay buffer (with n-step returns) =====
            batch = self.sac_storage.sample(self.batch_size)
            # batch now includes n-step gamma as the 6th element
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
                # where R_n is the n-step cumulative discounted reward
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
            
            # Actor loss: minimize -Q + alpha * log_prob
            actor_loss = (self.alpha * log_prob - min_q_new).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.actor_backbone.parameters()) + 
                list(self.policy.actor_mean.parameters()) + 
                list(self.policy.actor_log_std.parameters()),
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

            # ===== Update AMP Discriminator =====
            # Sample from AMP buffers
            amp_policy_batch = list(self.amp_storage.feed_forward_generator(1, self.amp_batch_size))[0]
            amp_expert_batch = list(self.amp_data.feed_forward_generator(1, self.amp_batch_size))[0]
            
            policy_state, policy_next_state = amp_policy_batch
            expert_state, expert_next_state = amp_expert_batch
            
            # Normalize if needed
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            
            # Discriminator predictions
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            
            # Discriminator loss (least-squares GAN)
            expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = F.mse_loss(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            
            # Gradient penalty
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=10
            )
            
            # Total discriminator loss
            disc_loss = self.amp_loss_coef * amp_loss + self.amp_grad_penalty_coef * grad_pen_loss
            
            # Update discriminator
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()
            
            # Update normalizer
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # Accumulate for logging
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_amp_loss += amp_loss.item()
            total_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            mean_entropy += (-log_prob.mean()).item()

        # Average losses
        num_updates = self.updates_per_step
        loss_dict = {
            "value_function": total_critic_loss / num_updates,
            "surrogate": total_actor_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "amp": total_amp_loss / num_updates,
            "amp_grad_pen": total_grad_pen_loss / num_updates,
            "amp_policy_pred": mean_policy_pred / num_updates,
            "amp_expert_pred": mean_expert_pred / num_updates,
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
