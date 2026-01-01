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

"""AMP+SAC Off-Policy Runner.

Based on the paper: "Unlocking the Potential of Soft Actor-Critic for Imitation Learning"

Key differences from on-policy runner:
- Uses replay buffer for off-policy learning
- Collects experience and updates asynchronously
- Supports n-step returns
- No GAE computation needed
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms.amp_sac import AMPSAC
from rsl_rl.env import VecEnv
from rsl_rl.modules import Discriminator, EmpiricalNormalization
from rsl_rl.modules.sac_actor_critic import SACActorCritic
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state


class AmpOffPolicyRunner:
    """Off-policy runner for AMP+SAC training and evaluation.
    
    This runner handles the training loop for off-policy algorithms like SAC,
    collecting experience into a replay buffer and performing multiple
    gradient updates per environment step.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """Initialize the off-policy runner.
        
        Args:
            env: Vectorized environment
            train_cfg: Training configuration dictionary
            log_dir: Directory for logging
            device: Compute device
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Configure multi-GPU (disabled for now)
        self._configure_multi_gpu()

        # Training type
        self.training_type = "rl"

        # Resolve dimensions
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # Resolve privileged observations type
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
        else:
            self.privileged_obs_type = None

        # Resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # Create SAC Actor-Critic policy
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: SACActorCritic = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize AMP components
        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = Discriminator(
            amp_data.observation_dim * 2,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg["amp_task_reward_lerp"],
        ).to(self.device)
        
        min_std = torch.zeros(len(train_cfg["min_normalized_std"]), device=self.device, requires_grad=False)

        # Initialize SAC algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: AMPSAC = alg_class(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # Initialize storage
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Logging setup
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """Main training loop.
        
        Args:
            num_learning_iterations: Total number of training iterations
            init_at_random_ep_len: Initialize episodes at random lengths
        """
        # Initialize logger
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        amp_obs = self.env.get_amp_obs_for_expert_trans()
        obs, privileged_obs, amp_obs = obs.to(self.device), privileged_obs.to(self.device), amp_obs.to(self.device)
        
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Synchronize parameters for multi-GPU
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Training loop
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # ===== Collect Experience =====
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    
                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_obs_for_expert_trans()
                    
                    # Move to device
                    next_obs, rewards, dones, next_amp_obs = (
                        next_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )
                    
                    # Normalize observations
                    next_obs = self.obs_normalizer(next_obs)
                    if self.privileged_obs_type is not None:
                        next_privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        next_privileged_obs = next_obs

                    # Handle terminal states for AMP
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = self.env.reset_env_ids
                    terminal_amp_states = self.env.get_amp_obs_for_expert_trans()[reset_env_ids]
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    # Compute rewards with AMP discriminator
                    rewards_with_amp = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )[0]
                    
                    # Store transition in SAC replay buffer
                    self.alg.store_transition(
                        obs=obs,
                        actions=actions,
                        rewards=rewards_with_amp,
                        next_obs=next_obs,
                        dones=dones,
                        privileged_obs=privileged_obs,
                        next_privileged_obs=next_privileged_obs,
                    )
                    
                    # Store in AMP buffer
                    self.alg.amp_storage.insert(amp_obs, next_amp_obs_with_term)
                    
                    # Update current state
                    obs = next_obs.clone()
                    privileged_obs = next_privileged_obs.clone()
                    amp_obs = next_amp_obs.clone()

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                            
                        cur_reward_sum += rewards_with_amp
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            # ===== Update Policy =====
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log info
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
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
            locs: Local variables from training loop
            width: Console output width
            pad: Padding for labels
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
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/alpha", locs["loss_dict"].get("alpha", 0.2), locs["it"])

        # Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Training
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Alpha (temperature):':>{pad}} {locs['loss_dict'].get('alpha', 0.2):.4f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                if key != "alpha":
                    log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            infos: Additional info to save
        """
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alg.alpha_optimizer.state_dict(),
            "log_alpha": self.alg.log_alpha,
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "discriminator_optimizer_state_dict": self.alg.discriminator_optimizer.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            
        Returns:
            infos: Additional info from checkpoint
        """
        loaded_dict = torch.load(path, weights_only=False)
        
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        
        # Load alpha
        if "log_alpha" in loaded_dict:
            self.alg.log_alpha = loaded_dict["log_alpha"]
            self.alg.alpha = self.alg.log_alpha.exp().item()
        
        # Load observation normalizer
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        
        # Load optimizer
        if load_optimizer and resumed_training:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if "alpha_optimizer_state_dict" in loaded_dict:
                self.alg.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
            self.alg.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        
        # Load current iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """Get policy for inference.
        
        Args:
            device: Device to move policy to
            
        Returns:
            Inference policy function
        """
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        """Set models to training mode."""
        self.alg.policy.train()
        self.alg.discriminator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """Set models to evaluation mode."""
        self.alg.policy.eval()
        self.alg.discriminator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

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
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)
