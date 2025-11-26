# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

from __future__ import annotations

import numpy as np
import torch


class DDPGReplayBuffer:
    """Experience replay buffer for DDPG algorithm.

    Stores (state, action, reward, next_state, done) tuples for off-policy learning.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        """Initialize the replay buffer.

        Args:
            obs_dim: Dimension of observations.
            action_dim: Dimension of actions.
            buffer_size: Maximum size of the buffer.
            device: Device to store tensors on.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device

        # Pre-allocate memory
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, action_dim, device=device)
        self.rewards = torch.zeros(buffer_size, 1, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.dones = torch.zeros(buffer_size, 1, device=device)

        self.ptr = 0  # Current position in buffer
        self.size = 0  # Current size of buffer

    def insert(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add a batch of transitions to the buffer.

        Args:
            states: Batch of states (num_envs, obs_dim).
            actions: Batch of actions (num_envs, action_dim).
            rewards: Batch of rewards (num_envs,) or (num_envs, 1).
            next_states: Batch of next states (num_envs, obs_dim).
            dones: Batch of done flags (num_envs,) or (num_envs, 1).
        """
        batch_size = states.shape[0]

        # Reshape rewards and dones if necessary
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        # Handle wraparound
        if self.ptr + batch_size > self.buffer_size:
            # Split the batch
            first_part = self.buffer_size - self.ptr
            second_part = batch_size - first_part

            # First part
            self.states[self.ptr : self.buffer_size] = states[:first_part]
            self.actions[self.ptr : self.buffer_size] = actions[:first_part]
            self.rewards[self.ptr : self.buffer_size] = rewards[:first_part]
            self.next_states[self.ptr : self.buffer_size] = next_states[:first_part]
            self.dones[self.ptr : self.buffer_size] = dones[:first_part]

            # Second part (wrap around)
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]
        else:
            end_idx = self.ptr + batch_size
            self.states[self.ptr : end_idx] = states
            self.actions[self.ptr : end_idx] = actions
            self.rewards[self.ptr : end_idx] = rewards
            self.next_states[self.ptr : end_idx] = next_states
            self.dones[self.ptr : end_idx] = dones

        # Update pointer and size
        self.ptr = (self.ptr + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
