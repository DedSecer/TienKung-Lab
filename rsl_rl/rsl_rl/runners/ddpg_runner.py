# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""
DDPG On-Policy Runner

This runner handles the training loop for DDPG algorithm,
managing environment interaction, data collection, and policy updates.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms.ddpg import DDPG
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.ddpg_actor_critic import DDPGActorCritic
from rsl_rl.utils import store_code_state


class DDPGRunner:
    """Runner for DDPG training and evaluation."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        """Initialize DDPG runner.

        Args:
            env: Vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging.
            device: Device to run on.
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Configure multi-GPU
        self._configure_multi_gpu()

        # Get observation dimensions
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        num_actions = self.env.num_actions

        # Create policy network
        policy = DDPGActorCritic(
            num_obs=num_obs,
            num_actions=num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # Create DDPG algorithm
        self.alg: DDPG = DDPG(
            policy=policy,
            num_envs=self.env.num_envs,
            num_obs=num_obs,
            num_actions=num_actions,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **self.alg_cfg,
        )

        # Store training configuration
        self.num_steps_per_env = self.cfg.get("num_steps_per_env", 24)
        self.save_interval = self.cfg.get("save_interval", 100)
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)

        # Logging setup
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop.

        Args:
            num_learning_iterations: Number of training iterations.
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        # Initialize logger
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            else:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        obs = self.obs_normalizer(obs)

        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        # Synchronize parameters for multi-GPU
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Training loop
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # Rollout
            for _ in range(self.num_steps_per_env):
                # Select action
                with torch.no_grad():
                    actions = self.alg.act(obs)

                # Step environment
                next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                # Move to device and normalize
                next_obs = next_obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_obs_normalized = self.obs_normalizer(next_obs)

                # Process step (store transition)
                self.alg.process_env_step(
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                    next_obs=next_obs_normalized,
                )

                # Update current observation
                obs = next_obs_normalized

                # Book keeping
                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])

                    cur_reward_sum += rewards
                    cur_episode_length += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            collection_time = time.time() - start
            start = time.time()

            # Update policy
            loss_dict = self.alg.update()

            learn_time = time.time() - start
            self.current_learning_iteration = it

            # Logging
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            # Save code state on first iteration
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save final model
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Log training progress.

        Args:
            locs: Local variables dictionary.
            width: Log string width.
            pad: Padding for log entries.
        """
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        # Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Training metrics
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
            )

        # Buffer stats
        self.writer.add_scalar(
            "Buffer/size", len(self.alg.replay_buffer), locs["it"]
        )
        self.writer.add_scalar(
            "Buffer/total_steps", self.alg.total_steps, locs["it"]
        )

        # Print log
        str_title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Buffer size:':>{pad}} {len(self.alg.replay_buffer)}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key}:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                f"""{'Buffer size:':>{pad}} {len(self.alg.replay_buffer)}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            infos: Additional info to save.
        """
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "target_model_state_dict": self.alg.policy_target.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "total_steps": self.alg.total_steps,
            "infos": infos,
        }

        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()

        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint.
            load_optimizer: Whether to load optimizer state.

        Returns:
            Loaded infos dictionary.
        """
        loaded_dict = torch.load(path, weights_only=False)

        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.policy_target.load_state_dict(loaded_dict["target_model_state_dict"])

        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(
                loaded_dict["critic_optimizer_state_dict"]
            )

        if self.empirical_normalization and "obs_norm_state_dict" in loaded_dict:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

        self.current_learning_iteration = loaded_dict.get("iter", 0)
        self.alg.total_steps = loaded_dict.get("total_steps", 0)

        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        """Get policy for inference.

        Args:
            device: Device to move policy to.

        Returns:
            Inference policy function.
        """
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)

        policy = self.alg.policy.act_inference
        if self.cfg.get("empirical_normalization", False):
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))

        return policy

    def train_mode(self):
        """Set networks to training mode."""
        self.alg.policy.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.alg.policy.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """Add git repository to logging."""
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self):
        """Configure multi-GPU training."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )

        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        torch.cuda.set_device(self.gpu_local_rank)
