# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
DDPG (Deep Deterministic Policy Gradient) Algorithm

This is an off-policy actor-critic algorithm designed for continuous action spaces.
It extends DQN concepts to continuous domains by:
1. Using a deterministic policy (actor) instead of discrete Q-values
2. Using a Q-function (critic) that evaluates state-action pairs
3. Using experience replay for sample efficiency
4. Using target networks with soft updates for stability

This implementation also includes TD3 (Twin Delayed DDPG) improvements:
- Twin critics to reduce overestimation bias
- Delayed policy updates
- Target policy smoothing

Reference:
- DDPG: https://arxiv.org/abs/1509.02971
- TD3: https://arxiv.org/abs/1802.09477
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.ddpg_actor_critic import DDPGActorCritic
from rsl_rl.storage.ddpg_replay_buffer import DDPGReplayBuffer


class DDPG:
    """Deep Deterministic Policy Gradient algorithm with TD3 improvements."""

    def __init__(
        self,
        policy: DDPGActorCritic,
        num_envs: int,
        num_obs: int,
        num_actions: int,
        # Replay buffer parameters
        buffer_size: int = 1000000,
        batch_size: int = 256,
        # Learning parameters
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        # Exploration parameters
        exploration_noise_std: float = 0.1,
        exploration_noise_clip: float = 0.3,
        # TD3 parameters
        policy_noise_std: float = 0.2,
        policy_noise_clip: float = 0.5,
        policy_update_freq: int = 2,
        # Warmup parameters
        warmup_steps: int = 10000,
        # Device
        device: str = "cpu",
        # Multi-GPU (for compatibility)
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        """Initialize DDPG algorithm.

        Args:
            policy: Actor-Critic policy network.
            num_envs: Number of parallel environments.
            num_obs: Dimension of observations.
            num_actions: Dimension of actions.
            buffer_size: Size of replay buffer.
            batch_size: Mini-batch size for training.
            actor_learning_rate: Learning rate for actor.
            critic_learning_rate: Learning rate for critic.
            gamma: Discount factor.
            tau: Soft update coefficient for target networks.
            exploration_noise_std: Std of exploration noise.
            exploration_noise_clip: Clipping range for exploration noise.
            policy_noise_std: Std of target policy smoothing noise.
            policy_noise_clip: Clipping range for target policy noise.
            policy_update_freq: Frequency of delayed policy updates.
            warmup_steps: Number of random action steps before training.
            device: Device to run on.
            multi_gpu_cfg: Multi-GPU configuration (for compatibility).
        """
        if kwargs:
            print(f"DDPG.__init__ got unexpected arguments: {list(kwargs.keys())}")

        self.device = device
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions

        # Networks
        self.policy = policy.to(device)
        self.policy_target = copy.deepcopy(policy).to(device)
        self.policy_target.eval()

        # Freeze target networks
        for param in self.policy_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            list(self.policy.critic.parameters()) + list(self.policy.critic2.parameters()),
            lr=critic_learning_rate,
        )

        # Replay buffer
        self.replay_buffer = DDPGReplayBuffer(
            obs_dim=num_obs,
            action_dim=num_actions,
            buffer_size=buffer_size,
            device=device,
        )

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise_std = exploration_noise_std
        self.exploration_noise_clip = exploration_noise_clip
        self.policy_noise_std = policy_noise_std
        self.policy_noise_clip = policy_noise_clip
        self.policy_update_freq = policy_update_freq
        self.warmup_steps = warmup_steps

        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.learning_rate = actor_learning_rate

        # Transition storage for current step
        self._current_obs = None
        self._current_actions = None

        # For compatibility with RND
        self.rnd = None
        self.intrinsic_rewards = None

        # Multi-GPU (for compatibility)
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

    def init_storage(self, *args, **kwargs):
        """Initialize storage (for compatibility with runner interface)."""
        pass

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> torch.Tensor:
        """Select action for given observation.

        Args:
            obs: Current observations.
            critic_obs: Critic observations (unused, for compatibility).

        Returns:
            Actions tensor.
        """
        self._current_obs = obs.clone()

        # Random actions during warmup
        if self.total_steps < self.warmup_steps:
            actions = torch.rand(obs.shape[0], self.num_actions, device=self.device) * 2 - 1
            actions = actions * self.policy.action_scale
        else:
            # Get actions with exploration noise
            with torch.no_grad():
                actions = self.policy.act_with_noise(
                    obs,
                    noise_std=self.exploration_noise_std,
                    noise_clip=self.exploration_noise_clip,
                )

        self._current_actions = actions.clone()
        self.total_steps += obs.shape[0]

        return actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_obs: torch.Tensor = None,
    ):
        """Process environment step and store transition.

        Args:
            rewards: Rewards from environment.
            dones: Done flags from environment.
            infos: Info dict from environment.
            next_obs: Next observations (if not in infos).
        """
        # Get next observations
        if next_obs is None:
            if "observations" in infos and "policy" in infos["observations"]:
                next_obs = infos["observations"]["policy"]
            else:
                # Will be handled by the runner
                return

        # Store transition in replay buffer
        self.replay_buffer.insert(
            states=self._current_obs,
            actions=self._current_actions,
            rewards=rewards,
            next_states=next_obs,
            dones=dones.float(),
        )

    def compute_returns(self, *args, **kwargs):
        """Compute returns (not used in DDPG, for compatibility)."""
        pass

    def update(self) -> dict:
        """Update policy and value networks.

        Returns:
            Dictionary of loss values.
        """
        # Don't update until we have enough samples
        if len(self.replay_buffer) < self.batch_size or self.total_steps < self.warmup_steps:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "q1_value": 0.0,
                "q2_value": 0.0,
            }

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_q1_value = 0.0
        total_q2_value = 0.0
        num_updates = 0

        # Perform multiple updates per call (similar to num_learning_epochs in PPO)
        num_updates_per_call = 10

        for _ in range(num_updates_per_call):
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )

            # Update critics
            critic_loss, q1_value, q2_value = self._update_critic(
                states, actions, rewards, next_states, dones
            )
            total_critic_loss += critic_loss
            total_q1_value += q1_value
            total_q2_value += q2_value

            # Delayed policy update (TD3 style)
            self.update_count += 1
            if self.update_count % self.policy_update_freq == 0:
                actor_loss = self._update_actor(states)
                total_actor_loss += actor_loss

                # Soft update target networks
                self._soft_update_target()

            num_updates += 1

        # Average losses
        return {
            "actor_loss": total_actor_loss / max(num_updates // self.policy_update_freq, 1),
            "critic_loss": total_critic_loss / num_updates,
            "q1_value": total_q1_value / num_updates,
            "q2_value": total_q2_value / num_updates,
        }

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple:
        """Update critic networks.

        Args:
            states: Batch of states.
            actions: Batch of actions.
            rewards: Batch of rewards.
            next_states: Batch of next states.
            dones: Batch of done flags.

        Returns:
            Tuple of (critic_loss, q1_mean, q2_mean).
        """
        with torch.no_grad():
            # Get target actions with smoothing noise (TD3)
            next_actions = self.policy_target.act(next_states)
            noise = torch.randn_like(next_actions) * self.policy_noise_std
            noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)
            next_actions = (next_actions + noise).clamp(
                -self.policy.action_scale, self.policy.action_scale
            )

            # Get target Q-values (use minimum of two critics)
            target_q1, target_q2 = self.policy_target.get_q_values(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Compute TD target
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Get current Q-values
        current_q1, current_q2 = self.policy.get_q_values(states, actions)

        # Compute critic loss
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(
            current_q2, target_q
        )

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.critic.parameters()) + list(self.policy.critic2.parameters()),
            max_norm=1.0,
        )
        self.critic_optimizer.step()

        return critic_loss.item(), current_q1.mean().item(), current_q2.mean().item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network.

        Args:
            states: Batch of states.

        Returns:
            Actor loss value.
        """
        # Compute actor loss (maximize Q-value)
        actions = self.policy.act(states)
        q_value = self.policy.get_q_value(states, actions)
        actor_loss = -q_value.mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update_target(self):
        """Soft update target networks."""
        with torch.no_grad():
            for param, target_param in zip(
                self.policy.parameters(), self.policy_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs (for multi-GPU training)."""
        if not self.is_multi_gpu:
            return

        model_params = [self.policy.state_dict(), self.policy_target.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.policy_target.load_state_dict(model_params[1])
