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

"""SAC Actor-Critic module for AMP+SAC framework.

Based on the paper: "Unlocking the Potential of Soft Actor-Critic for Imitation Learning"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class SACActorCritic(nn.Module):
    """SAC Actor-Critic with separate actor and twin Q-networks.
    
    Following the paper's architecture:
    - Actor: MLP with GELU activation, outputs mean and log_std of Gaussian
    - Critic: Twin Q-networks with GELU activation
    - Uses tanh squashing for bounded actions
    - Implements reparameterization trick for gradient flow
    """
    
    is_recurrent = False
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[1024, 512],
        critic_hidden_dims=[1024, 512],
        activation="gelu",
        init_noise_std=1.0,
        action_scale=1.0,
        **kwargs,
    ):
        """Initialize SAC Actor-Critic.
        
        Args:
            num_actor_obs: Dimension of actor observations
            num_critic_obs: Dimension of critic observations
            num_actions: Dimension of action space
            actor_hidden_dims: Hidden layer dimensions for actor [1024, 512]
            critic_hidden_dims: Hidden layer dimensions for critics [1024, 512]
            activation: Activation function (GELU as per paper)
            init_noise_std: Initial noise standard deviation
            action_scale: Scale factor for actions
        """
        if kwargs:
            print(
                "SACActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.num_actions = num_actions
        self.action_scale = action_scale
        activation_fn = resolve_nn_activation(activation)
        
        # ===== Actor Network =====
        # Outputs mean and log_std of Gaussian distribution
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            actor_layers.append(activation_fn)
        self.actor_backbone = nn.Sequential(*actor_layers)
        
        # Separate heads for mean and log_std
        self.actor_mean = nn.Linear(actor_hidden_dims[-1], num_actions)
        self.actor_log_std = nn.Linear(actor_hidden_dims[-1], num_actions)
        
        # ===== Twin Q-Networks (Critic) =====
        # Q1 network
        q1_layers = []
        q1_input_dim = num_critic_obs + num_actions
        q1_layers.append(nn.Linear(q1_input_dim, critic_hidden_dims[0]))
        q1_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            q1_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            q1_layers.append(activation_fn)
        q1_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2 network (twin)
        q2_layers = []
        q2_layers.append(nn.Linear(q1_input_dim, critic_hidden_dims[0]))
        q2_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            q2_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            q2_layers.append(activation_fn)
        q2_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*q2_layers)
        
        # ===== Target Q-Networks =====
        # Q1 target
        q1_target_layers = []
        q1_target_layers.append(nn.Linear(q1_input_dim, critic_hidden_dims[0]))
        q1_target_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            q1_target_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            q1_target_layers.append(activation_fn)
        q1_target_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.q1_target = nn.Sequential(*q1_target_layers)
        
        # Q2 target
        q2_target_layers = []
        q2_target_layers.append(nn.Linear(q1_input_dim, critic_hidden_dims[0]))
        q2_target_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            q2_target_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            q2_target_layers.append(activation_fn)
        q2_target_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.q2_target = nn.Sequential(*q2_target_layers)
        
        # Initialize target networks with same weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Freeze target networks (updated via soft update)
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        # Action distribution (populated during act)
        self.distribution = None
        self._action_mean = None
        self._action_std = None
        
        print(f"SAC Actor Backbone: {self.actor_backbone}")
        print(f"SAC Q1 Network: {self.q1}")
        print(f"SAC Q2 Network: {self.q2}")
        
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def actor(self):
        return self

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        return self.act_inference(observations)

    @property
    def action_mean(self):
        return self._action_mean if self._action_mean is not None else torch.zeros(1)

    @property
    def action_std(self):
        return self._action_std if self._action_std is not None else torch.ones(1)

    @property
    def entropy(self):
        if self.distribution is not None:
            return self.distribution.entropy().sum(dim=-1)
        return torch.zeros(1)

    def get_action_distribution(self, observations):
        """Get action distribution from actor network.
        
        Args:
            observations: Actor observations
            
        Returns:
            mean, log_std of the Gaussian distribution
        """
        features = self.actor_backbone(observations)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample_action(self, observations, deterministic=False):
        """Sample action using reparameterization trick with tanh squashing.
        
        Args:
            observations: Actor observations
            deterministic: If True, return mean action
            
        Returns:
            action: Squashed action
            log_prob: Log probability of the action
            mean: Mean of the distribution (for logging)
        """
        mean, log_std = self.get_action_distribution(observations)
        std = torch.exp(log_std)
        
        # Store for properties
        self._action_mean = mean
        self._action_std = std
        
        if deterministic:
            action = torch.tanh(mean) * self.action_scale
            return action, None, mean
        
        # Reparameterization trick
        normal = Normal(mean, std)
        self.distribution = normal
        
        # Sample using rsample for gradient flow
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t) * self.action_scale
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound: log_prob = log_prob - log(1 - tanh^2(x_t))
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2) / (self.action_scale ** 2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean

    def act(self, observations, **kwargs):
        """Sample action for environment interaction."""
        action, _, _ = self.sample_action(observations, deterministic=False)
        return action

    def act_inference(self, observations):
        """Get deterministic action for inference."""
        action, _, _ = self.sample_action(observations, deterministic=True)
        return action

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution."""
        if self.distribution is not None:
            return self.distribution.log_prob(actions).sum(dim=-1)
        return torch.zeros(actions.shape[0])

    def get_q_values(self, observations, actions):
        """Get Q-values from both Q-networks.
        
        Args:
            observations: Critic observations
            actions: Actions
            
        Returns:
            q1_value, q2_value: Q-values from both networks
        """
        sa = torch.cat([observations, actions], dim=-1)
        q1_value = self.q1(sa)
        q2_value = self.q2(sa)
        return q1_value, q2_value

    def get_target_q_values(self, observations, actions):
        """Get Q-values from target Q-networks.
        
        Args:
            observations: Critic observations
            actions: Actions
            
        Returns:
            q1_target_value, q2_target_value: Target Q-values
        """
        sa = torch.cat([observations, actions], dim=-1)
        q1_target_value = self.q1_target(sa)
        q2_target_value = self.q2_target(sa)
        return q1_target_value, q2_target_value

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value (for compatibility with PPO interface).
        
        For SAC, we compute V(s) = E[Q(s,a) - alpha * log(pi(a|s))]
        This is approximated using a sampled action.
        """
        # Sample action for the given state
        action, log_prob, _ = self.sample_action(critic_observations, deterministic=False)
        # Get Q-values
        q1, q2 = self.get_q_values(critic_observations, action)
        # Use minimum for conservative estimate
        min_q = torch.min(q1, q2)
        return min_q

    def soft_update_target(self, tau=0.005):
        """Soft update target networks.
        
        Args:
            tau: Target smoothing coefficient (default: 0.005)
        """
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model."""
        super().load_state_dict(state_dict, strict=strict)
        return True
