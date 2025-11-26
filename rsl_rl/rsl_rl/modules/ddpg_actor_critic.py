# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class DDPGActorCritic(nn.Module):
    """Actor-Critic network for DDPG algorithm.

    This implements:
    - Actor: Deterministic policy that outputs actions directly
    - Critic: Q-function that estimates Q(s, a)
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims: list = [256, 256, 256],
        activation: str = "relu",
        action_scale: float = 1.0,
        **kwargs,
    ):
        """Initialize the Actor-Critic networks.

        Args:
            num_obs: Dimension of observations.
            num_actions: Dimension of actions.
            actor_hidden_dims: Hidden layer sizes for actor network.
            critic_hidden_dims: Hidden layer sizes for critic network.
            activation: Activation function name.
            action_scale: Scale factor for action output.
        """
        if kwargs:
            print(
                "DDPGActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.action_scale = action_scale

        activation_fn = resolve_nn_activation(activation)

        # Build Actor network (deterministic policy)
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        actor_layers.append(nn.Tanh())  # Output in [-1, 1], then scale
        self.actor = nn.Sequential(*actor_layers)

        # Build Critic network (Q-function: Q(s, a))
        # Input: concatenation of state and action
        critic_input_dim = num_obs + num_actions
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # Build second Critic for Twin Delayed DDPG (TD3) style
        critic2_layers = []
        critic2_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic2_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic2_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            critic2_layers.append(activation_fn)
        critic2_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic2 = nn.Sequential(*critic2_layers)

        print(f"DDPG Actor MLP: {self.actor}")
        print(f"DDPG Critic MLP: {self.critic}")
        print(f"DDPG Critic2 MLP: {self.critic2}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in [self.actor, self.critic, self.critic2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)

        # Special initialization for output layers
        # Actor output layer - smaller weights for stable initial actions
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                last_linear = layer
        nn.init.uniform_(last_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(last_linear.bias, -3e-3, 3e-3)

    def forward(self):
        raise NotImplementedError

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action from actor.

        Args:
            obs: Observations tensor.

        Returns:
            Actions tensor scaled by action_scale.
        """
        return self.actor(obs) * self.action_scale

    def act_with_noise(
        self,
        obs: torch.Tensor,
        noise_std: float = 0.1,
        noise_clip: float = 0.3,
    ) -> torch.Tensor:
        """Get action with exploration noise.

        Args:
            obs: Observations tensor.
            noise_std: Standard deviation of Gaussian noise.
            noise_clip: Clipping range for noise.

        Returns:
            Noisy actions tensor.
        """
        action = self.act(obs)
        noise = torch.randn_like(action) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        noisy_action = action + noise
        return noisy_action.clamp(-self.action_scale, self.action_scale)

    def get_q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first critic.

        Args:
            obs: Observations tensor.
            action: Actions tensor.

        Returns:
            Q-value tensor.
        """
        x = torch.cat([obs, action], dim=-1)
        return self.critic(x)

    def get_q_values(self, obs: torch.Tensor, action: torch.Tensor) -> tuple:
        """Get Q-values from both critics (for TD3-style double Q-learning).

        Args:
            obs: Observations tensor.
            action: Actions tensor.

        Returns:
            Tuple of (Q1, Q2) value tensors.
        """
        x = torch.cat([obs, action], dim=-1)
        return self.critic(x), self.critic2(x)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action for inference (no noise).

        Args:
            obs: Observations tensor.

        Returns:
            Actions tensor.
        """
        return self.act(obs)

    def reset(self, dones=None):
        """Reset method for compatibility."""
        pass

    def load_state_dict(self, state_dict, strict=True):
        """Load model parameters."""
        super().load_state_dict(state_dict, strict=strict)
        return True
