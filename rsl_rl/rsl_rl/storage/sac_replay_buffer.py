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

"""SAC Replay Buffer for off-policy learning.

Based on the paper: "Unlocking the Potential of Soft Actor-Critic for Imitation Learning"
Paper settings:
- Replay memory size: 10^7
- Batch size: 16384
- n-step return: 3
"""

from __future__ import annotations

import numpy as np
import torch


class SACReplayBuffer:
    """Fixed-size replay buffer for SAC off-policy learning.
    
    Stores transitions (s, a, r, s', done) for experience replay.
    Supports n-step returns for improved learning efficiency.
    
    NOTE: Buffer is stored on CPU to save GPU memory. 
    Samples are moved to compute device during training.
    
    Following paper settings (adjusted for memory):
    - Buffer capacity: 10^6 (reduced from 10^7 for memory efficiency)
    - Batch size: 16384
    - n-step return: 3
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        buffer_size: int = 10_000_000,
        device: str = "cuda:0",  # Compute device for sampling
        storage_device: str = "cpu",  # Store buffer on CPU to save GPU memory
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        """Initialize SAC Replay Buffer.
        
        Args:
            obs_dim: Dimension of observations
            action_dim: Dimension of actions
            buffer_size: Maximum size of buffer (default: 10^6)
            device: Device for sampled batches during training
            storage_device: Device for storing buffer (default: CPU)
            n_step: Number of steps for n-step returns (default: 3)
            gamma: Discount factor (default: 0.99)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device  # Compute device for training
        self.storage_device = storage_device  # Storage device (CPU to save memory)
        self.n_step = n_step
        self.gamma = gamma
        
        # Allocate memory on CPU to save GPU memory
        self.observations = torch.zeros(buffer_size, obs_dim, device=storage_device)
        self.actions = torch.zeros(buffer_size, action_dim, device=storage_device)
        self.rewards = torch.zeros(buffer_size, 1, device=storage_device)
        self.next_observations = torch.zeros(buffer_size, obs_dim, device=storage_device)
        self.dones = torch.zeros(buffer_size, 1, device=storage_device)
        
        # For privileged observations (critic)
        self.privileged_observations = None
        self.next_privileged_observations = None
        
        # n-step return buffer (temporary storage)
        self.n_step_buffer = []
        
        # Pointer and size
        self.ptr = 0
        self.size = 0

    def init_privileged_obs(self, privileged_obs_dim: int):
        """Initialize buffer for privileged observations.
        
        Args:
            privileged_obs_dim: Dimension of privileged observations
        """
        self.privileged_observations = torch.zeros(
            self.buffer_size, privileged_obs_dim, device=self.storage_device
        )
        self.next_privileged_observations = torch.zeros(
            self.buffer_size, privileged_obs_dim, device=self.storage_device
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
        """Add new transitions to the buffer.
        
        Handles batch insertion for vectorized environments.
        
        Args:
            observations: Current observations [num_envs, obs_dim]
            actions: Actions taken [num_envs, action_dim]
            rewards: Rewards received [num_envs, 1] or [num_envs]
            next_observations: Next observations [num_envs, obs_dim]
            dones: Done flags [num_envs, 1] or [num_envs]
            privileged_observations: Current privileged obs (optional)
            next_privileged_observations: Next privileged obs (optional)
        """
        num_transitions = observations.shape[0]
        
        # Move data to storage device (CPU) to save GPU memory
        observations = observations.to(self.storage_device)
        actions = actions.to(self.storage_device)
        rewards = rewards.to(self.storage_device)
        next_observations = next_observations.to(self.storage_device)
        dones = dones.to(self.storage_device)
        if privileged_observations is not None:
            privileged_observations = privileged_observations.to(self.storage_device)
            next_privileged_observations = next_privileged_observations.to(self.storage_device)
        
        # Reshape rewards and dones if needed
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Calculate indices for insertion
        start_idx = self.ptr
        end_idx = self.ptr + num_transitions
        
        if end_idx > self.buffer_size:
            # Wrap around
            first_part = self.buffer_size - start_idx
            second_part = end_idx - self.buffer_size
            
            self.observations[start_idx:self.buffer_size] = observations[:first_part]
            self.observations[:second_part] = observations[first_part:]
            
            self.actions[start_idx:self.buffer_size] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]
            
            self.rewards[start_idx:self.buffer_size] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]
            
            self.next_observations[start_idx:self.buffer_size] = next_observations[:first_part]
            self.next_observations[:second_part] = next_observations[first_part:]
            
            self.dones[start_idx:self.buffer_size] = dones[:first_part]
            self.dones[:second_part] = dones[first_part:]
            
            if privileged_observations is not None and self.privileged_observations is not None:
                self.privileged_observations[start_idx:self.buffer_size] = privileged_observations[:first_part]
                self.privileged_observations[:second_part] = privileged_observations[first_part:]
                self.next_privileged_observations[start_idx:self.buffer_size] = next_privileged_observations[:first_part]
                self.next_privileged_observations[:second_part] = next_privileged_observations[first_part:]
        else:
            self.observations[start_idx:end_idx] = observations
            self.actions[start_idx:end_idx] = actions
            self.rewards[start_idx:end_idx] = rewards
            self.next_observations[start_idx:end_idx] = next_observations
            self.dones[start_idx:end_idx] = dones
            
            if privileged_observations is not None and self.privileged_observations is not None:
                self.privileged_observations[start_idx:end_idx] = privileged_observations
                self.next_privileged_observations[start_idx:end_idx] = next_privileged_observations
        
        self.ptr = (self.ptr + num_transitions) % self.buffer_size
        self.size = min(self.size + num_transitions, self.buffer_size)

    def sample(self, batch_size: int = 16384):
        """Sample a batch of transitions with n-step returns.
        
        When n_step > 1, computes:
        - n-step cumulative reward: r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{n-1}*r_{t+n-1}
        - n-step next state: s_{t+n}
        - n-step done: whether episode ended within n steps
        - n-step gamma: γ^n (for target computation)
        
        Args:
            batch_size: Number of transitions to sample (default: 16384)
            
        Returns:
            Tuple of (observations, actions, n_step_rewards, n_step_next_observations, 
                      n_step_dones, n_step_gamma)
            If privileged observations are stored, also returns them.
            All tensors are moved to the compute device (GPU).
        """
        # Sample starting indices (ensure we have n_step valid transitions ahead)
        max_start_idx = self.size - self.n_step
        if max_start_idx <= 0:
            # Not enough samples for n-step, fall back to 1-step
            return self._sample_1step(batch_size)
        
        # Sample start indices
        start_indices = np.random.choice(max_start_idx, size=batch_size, replace=True)
        
        # Initialize n-step returns
        n_step_rewards = torch.zeros(batch_size, 1, device=self.storage_device)
        n_step_dones = torch.zeros(batch_size, 1, device=self.storage_device)
        n_step_gammas = torch.ones(batch_size, 1, device=self.storage_device) * (self.gamma ** self.n_step)
        
        # Compute n-step rewards and find terminal states
        discount = 1.0
        for step in range(self.n_step):
            step_indices = (start_indices + step) % self.buffer_size
            step_rewards = self.rewards[step_indices]
            step_dones = self.dones[step_indices]
            
            # Accumulate discounted rewards
            n_step_rewards += discount * step_rewards * (1 - n_step_dones)
            
            # Check for episode termination
            # If episode terminates at step k < n, we use s_{t+k+1} and γ^{k+1}
            terminated_mask = (step_dones > 0) & (n_step_dones == 0)
            n_step_dones = torch.maximum(n_step_dones, step_dones)
            
            # Update gamma for terminated episodes
            if step < self.n_step - 1:
                n_step_gammas[terminated_mask.squeeze()] = self.gamma ** (step + 1)
            
            discount *= self.gamma
        
        # Get n-step next observations (or terminal state if episode ended)
        # For simplicity, we use the state at t+n (or the last valid state)
        n_step_next_indices = (start_indices + self.n_step) % self.buffer_size
        
        # Sample from CPU storage and move to compute device
        batch = (
            self.observations[start_indices].to(self.device),
            self.actions[start_indices].to(self.device),
            n_step_rewards.to(self.device),
            self.next_observations[n_step_next_indices].to(self.device),
            n_step_dones.to(self.device),
            n_step_gammas.to(self.device),  # Return the effective gamma for each sample
        )
        
        if self.privileged_observations is not None:
            batch = batch + (
                self.privileged_observations[start_indices].to(self.device),
                self.next_privileged_observations[n_step_next_indices].to(self.device),
            )
        
        return batch

    def _sample_1step(self, batch_size: int):
        """Fallback 1-step sampling when buffer is too small for n-step.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Standard 1-step transitions with gamma as additional output
        """
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=True)
        
        batch = (
            self.observations[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_observations[indices].to(self.device),
            self.dones[indices].to(self.device),
            torch.ones(len(indices), 1, device=self.device) * self.gamma,  # 1-step gamma
        )
        
        if self.privileged_observations is not None:
            batch = batch + (
                self.privileged_observations[indices].to(self.device),
                self.next_privileged_observations[indices].to(self.device),
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
        return self.size

    def is_ready(self, min_size: int = 1000):
        """Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum number of samples required
            
        Returns:
            True if buffer has enough samples
        """
        return self.size >= min_size

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
