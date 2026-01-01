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

"""SAC Replay Buffer for off-policy learning with parallel environments.

Based on the paper: "Unlocking the Potential of Soft Actor-Critic for Imitation Learning"
Paper settings:
- Replay memory size: 10^7
- Batch size: 16384
- n-step return: 3

IMPORTANT: This buffer is designed for vectorized environments where multiple
environments run in parallel. The buffer maintains temporal consistency per
environment to correctly compute n-step returns.

Buffer Layout:
- Shape: [buffer_size, num_envs, dim]
- Each time index stores data from ALL environments at that timestep
- N-step sampling operates along the time dimension for each environment
"""

from __future__ import annotations

import numpy as np
import torch


class SACReplayBuffer:
    """Fixed-size replay buffer for SAC off-policy learning with parallel environments.
    
    Stores transitions (s, a, r, s', done) for experience replay.
    Supports n-step returns with correct temporal alignment per environment.
    
    NOTE: Buffer is stored on CPU to save GPU memory. 
    Samples are moved to compute device during training.
    
    Key Design:
    - Buffer shape: [buffer_size, num_envs, dim]
    - Time index (ptr) advances once per step across ALL environments
    - N-step return correctly accumulates rewards along time dimension per env
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        buffer_size: int = 10_000_000,
        device: str = "cuda:0",
        storage_device: str = "cpu",
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        """Initialize SAC Replay Buffer for parallel environments.
        
        Args:
            num_envs: Number of parallel environments
            obs_dim: Dimension of observations
            action_dim: Dimension of actions
            buffer_size: Maximum number of timesteps to store (default: 10^6)
            device: Device for sampled batches during training
            storage_device: Device for storing buffer (default: CPU)
            n_step: Number of steps for n-step returns (default: 3)
            gamma: Discount factor (default: 0.99)
        """
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.storage_device = storage_device
        self.n_step = n_step
        self.gamma = gamma
        
        # Allocate memory on CPU: [buffer_size, num_envs, dim]
        # Each time index stores data from all environments at that timestep
        self.observations = torch.zeros(buffer_size, num_envs, obs_dim, device=storage_device)
        self.actions = torch.zeros(buffer_size, num_envs, action_dim, device=storage_device)
        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=storage_device)
        self.next_observations = torch.zeros(buffer_size, num_envs, obs_dim, device=storage_device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=storage_device)
        
        # For privileged observations (critic)
        self.privileged_observations = None
        self.next_privileged_observations = None
        
        # Pointer and size (in timesteps, not total transitions)
        self.ptr = 0
        self.size = 0

    def init_privileged_obs(self, privileged_obs_dim: int):
        """Initialize buffer for privileged observations.
        
        Args:
            privileged_obs_dim: Dimension of privileged observations
        """
        self.privileged_observations = torch.zeros(
            self.buffer_size, self.num_envs, privileged_obs_dim, device=self.storage_device
        )
        self.next_privileged_observations = torch.zeros(
            self.buffer_size, self.num_envs, privileged_obs_dim, device=self.storage_device
        )

    def insert(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        privileged_observations: torch.Tensor = None,
        next_privileged_observations: torch.Tensor = None,
    ):
        """Add new transitions from all environments at current timestep.
        
        Args:
            observations: Current observations [num_envs, obs_dim]
            actions: Actions taken [num_envs, action_dim]
            rewards: Rewards received [num_envs, 1] or [num_envs]
            next_observations: Next observations [num_envs, obs_dim]
            dones: Done flags [num_envs, 1] or [num_envs]
            privileged_observations: Current privileged obs (optional)
            next_privileged_observations: Next privileged obs (optional)
        """
        # Move data to storage device (CPU)
        observations = observations.to(self.storage_device)
        actions = actions.to(self.storage_device)
        rewards = rewards.to(self.storage_device)
        next_observations = next_observations.to(self.storage_device)
        dones = dones.to(self.storage_device)
        
        # Reshape rewards and dones if needed
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Store at current time index
        self.observations[self.ptr] = observations
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_observations[self.ptr] = next_observations
        self.dones[self.ptr] = dones
        
        if privileged_observations is not None and self.privileged_observations is not None:
            privileged_observations = privileged_observations.to(self.storage_device)
            next_privileged_observations = next_privileged_observations.to(self.storage_device)
            self.privileged_observations[self.ptr] = privileged_observations
            self.next_privileged_observations[self.ptr] = next_privileged_observations
        
        # Advance pointer
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int = 16384):
        """Sample a batch of transitions with n-step returns.
        
        Correctly handles n-step returns by sampling along the time dimension
        for each environment, ensuring temporal consistency.
        
        When n_step > 1, computes:
        - n-step cumulative reward: r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
        - n-step next state: s_{t+k} where k is termination step (or n)
        - n-step done: whether episode ended within n steps
        - n-step gamma: γ^k (for target computation)
        
        Args:
            batch_size: Number of transitions to sample (default: 16384)
            
        Returns:
            Tuple of (observations, actions, n_step_rewards, n_step_next_observations, 
                      n_step_dones, n_step_gamma)
            All tensors are moved to the compute device (GPU).
        """
        # Ensure we have enough timesteps for n-step
        max_start_time = self.size - self.n_step
        if max_start_time <= 0:
            return self._sample_1step(batch_size)
        
        # Sample (time_index, env_index) pairs
        time_indices = np.random.randint(0, max_start_time, size=batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=batch_size)
        
        # Initialize n-step returns
        n_step_rewards = torch.zeros(batch_size, 1, device=self.storage_device)
        n_step_dones = torch.zeros(batch_size, 1, device=self.storage_device)
        n_step_gammas = torch.ones(batch_size, 1, device=self.storage_device) * (self.gamma ** self.n_step)
        
        # Track effective next state time index
        # Default: t + n - 1 (to get next_obs which is s_{t+n})
        effective_next_time = np.array((time_indices + self.n_step - 1) % self.buffer_size)
        
        # Compute n-step rewards along time dimension
        discount = 1.0
        for step in range(self.n_step):
            step_time = (time_indices + step) % self.buffer_size
            
            # Get rewards and dones for this step [batch_size, 1]
            step_rewards = self.rewards[step_time, env_indices]
            step_dones = self.dones[step_time, env_indices]
            
            # Accumulate discounted rewards (only for non-terminated samples)
            n_step_rewards += discount * step_rewards * (1 - n_step_dones)
            
            # Check for episode termination
            terminated_now = (step_dones > 0) & (n_step_dones == 0)
            terminated_now_np = terminated_now.squeeze().cpu().numpy()
            
            # Update effective next time for terminated samples
            if terminated_now_np.any():
                effective_next_time[terminated_now_np] = step_time[terminated_now_np]
                n_step_gammas[terminated_now] = self.gamma ** (step + 1)
            
            # Mark as done
            n_step_dones = torch.maximum(n_step_dones, step_dones)
            
            discount *= self.gamma
        
        # Gather samples and move to compute device
        batch = (
            self.observations[time_indices, env_indices].to(self.device),
            self.actions[time_indices, env_indices].to(self.device),
            n_step_rewards.to(self.device),
            self.next_observations[effective_next_time, env_indices].to(self.device),
            n_step_dones.to(self.device),
            n_step_gammas.to(self.device),
        )
        
        if self.privileged_observations is not None:
            batch = batch + (
                self.privileged_observations[time_indices, env_indices].to(self.device),
                self.next_privileged_observations[effective_next_time, env_indices].to(self.device),
            )
        
        return batch

    def _sample_1step(self, batch_size: int):
        """Fallback 1-step sampling when buffer is too small for n-step.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Standard 1-step transitions with gamma as additional output
        """
        # Sample (time_index, env_index) pairs
        time_indices = np.random.randint(0, self.size, size=min(batch_size, self.size * self.num_envs))
        env_indices = np.random.randint(0, self.num_envs, size=len(time_indices))
        
        batch = (
            self.observations[time_indices, env_indices].to(self.device),
            self.actions[time_indices, env_indices].to(self.device),
            self.rewards[time_indices, env_indices].to(self.device),
            self.next_observations[time_indices, env_indices].to(self.device),
            self.dones[time_indices, env_indices].to(self.device),
            torch.ones(len(time_indices), 1, device=self.device) * self.gamma,
        )
        
        if self.privileged_observations is not None:
            batch = batch + (
                self.privileged_observations[time_indices, env_indices].to(self.device),
                self.next_privileged_observations[time_indices, env_indices].to(self.device),
            )
        
        return batch

    def sample_generator(self, num_mini_batches: int, batch_size: int):
        """Generate mini-batches for training.
        
        Args:
            num_mini_batches: Number of mini-batches to generate
            batch_size: Size of each mini-batch
            
        Yields:
            Mini-batch tuples
        """
        for _ in range(num_mini_batches):
            yield self.sample(batch_size)

    def __len__(self):
        """Return total number of transitions stored."""
        return self.size * self.num_envs

    def is_ready(self, min_size: int = 1000):
        """Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum number of transitions required
            
        Returns:
            True if buffer has enough samples
        """
        return len(self) >= min_size

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
