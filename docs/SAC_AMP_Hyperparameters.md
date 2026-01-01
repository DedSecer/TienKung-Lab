# SAC+AMP 超参数调优指南

基于论文 **"Unlocking the Potential of Soft Actor-Critic for Imitation Learning"** 实现的 SAC+AMP 框架超参数说明。

---

## 📊 目录

1. [SAC 核心超参数](#sac-核心超参数)
2. [AMP 判别器超参数](#amp-判别器超参数)
3. [网络架构超参数](#网络架构超参数)
4. [训练配置超参数](#训练配置超参数)
5. [环境配置超参数](#环境配置超参数)
6. [超参数调优建议](#超参数调优建议)

---

## SAC 核心超参数

这些超参数控制 Soft Actor-Critic 算法的核心行为。

| 参数名 | 论文值 | 默认值 | 说明 |
|--------|--------|--------|------|
| `sac_replay_buffer_size` | 10^7 | 10^7 | Replay Buffer 容量。存储历史经验用于离策略学习。越大越能保留多样化经验，但内存占用更高。Buffer 存储在 CPU 上以节省 GPU 内存。 |
| `batch_size` | 16384 | 16384 | 每次梯度更新采样的 transition 数量。较大的 batch 提供更稳定的梯度估计，但计算量更大。 |
| `updates_per_step` | 8 | 8 | 每次环境交互后的梯度更新次数。SAC 是离策略算法，可以进行多次更新以提高样本效率。 |
| `n_step_return` | 3 | 3 | n-step 回报的步数。使用多步回报可以加速价值传播，但可能引入更多偏差。 |
| `tau` | 0.05 | 0.05 | 目标网络软更新系数。控制目标 Q 网络的更新速度：`θ_target = τ*θ + (1-τ)*θ_target`。较小的值使训练更稳定。 |
| `gamma` | 0.99 | 0.99 | 折扣因子。控制未来奖励的重要性。接近 1 表示更重视长期回报。 |
| `actor_lr` | 0.001 | 0.001 | Actor 网络学习率。控制策略网络的学习速度。 |
| `critic_lr` | 0.001 | 0.001 | Critic 网络学习率。控制 Q 网络的学习速度。 |
| `alpha_lr` | 0.001 | 0.001 | 温度系数 α 的学习率（自动调节时使用）。 |
| `init_alpha` | 0.2 | 0.2 | 熵正则化温度系数初始值。控制探索与利用的平衡。 |
| `auto_alpha` | True | True | 是否自动调节温度系数 α。启用时，α 会根据目标熵自动调整。 |
| `target_entropy` | -dim(A) | None | 目标熵值。默认为动作空间维度的负值。用于自动调节 α。 |
| `max_grad_norm` | 1.0 | 1.0 | 梯度裁剪阈值。防止梯度爆炸。 |
| `warm_up_steps` | 100 | 100 | 预热步数。在此之前不进行策略更新，只收集经验。 |

### 详细说明

#### `sac_replay_buffer_size` (Replay Buffer 大小)
```python
sac_replay_buffer_size: int = 10_000_000  # 10^7
```
- **作用**：存储历史 transition (s, a, r, s', done) 用于经验回放
- **权衡**：
  - 较大：保留更多历史经验，有助于避免灾难性遗忘，但内存占用高
  - 较小：内存友好，但可能丢失有价值的早期经验
- **调优建议**：根据 GPU 内存调整。24GB GPU 建议 10^6 - 10^7

#### `tau` (目标网络软更新系数)
```python
tau: float = 0.05
```
- **作用**：控制目标 Q 网络更新的 "软度"
- **公式**：`θ_target = τ * θ_current + (1 - τ) * θ_target`
- **权衡**：
  - 较大（如 0.1）：目标网络更新更快，学习更激进
  - 较小（如 0.01）：更新更平滑，训练更稳定但可能更慢

#### `init_alpha` 和 `auto_alpha` (熵正则化)
```python
init_alpha: float = 0.2
auto_alpha: bool = True
```
- **作用**：α 控制探索程度。SAC 目标函数为 `max E[r + α * H(π)]`
- **调优建议**：
  - 开启 `auto_alpha` 让算法自动平衡探索与利用
  - 如果策略过于随机，减小 `init_alpha`
  - 如果策略收敛过快陷入局部最优，增大 `init_alpha`

---

## AMP 判别器超参数

这些参数控制 Adversarial Motion Prior 的判别器行为。

| 参数名 | 论文值 | 默认值 | 说明 |
|--------|--------|--------|------|
| `amp_loss_coef` | 0.1 | 0.1 | AMP 损失系数 λ_AMP。控制对抗损失在 Actor 损失中的权重。 |
| `amp_grad_penalty_coef` | 0.01 | 0.01 | 梯度惩罚系数 λ_GP。用于稳定判别器训练，防止过拟合。 |
| `amp_batch_size` | 8192 | 8192 | AMP 判别器每次更新采样的 transition 数量。 |
| `amp_reward_coef` | 0.3 | 0.3 | 判别器奖励系数。控制 AMP 奖励在总奖励中的缩放。 |
| `amp_task_reward_lerp` | 0.7 | 0.7 | 任务奖励权重。总奖励 = lerp * r_task + (1-lerp) * r_amp。 |
| `amp_num_preload_transitions` | 2×10^7 | 200000 | 预加载的专家 transition 数量。 |
| `amp_discr_hidden_dims` | [1024, 512] | [1024, 512] | 判别器网络隐藏层维度。 |
| `amp_replay_buffer_size` | 100000 | 100000 | AMP 专用 Replay Buffer 大小，存储策略生成的 (s, s') 对。 |

### 详细说明

#### `amp_task_reward_lerp` (任务与模仿奖励平衡)
```python
amp_task_reward_lerp: float = 0.7
```
- **作用**：平衡任务完成和动作模仿的重要性
- **公式**：`r_total = lerp * r_task + (1 - lerp) * r_amp`
- **论文设置**：w_task = 0.4, w_amp = 0.6（即 lerp = 0.4）
- **调优建议**：
  - 增大：更重视任务目标（速度跟踪、方向控制）
  - 减小：更重视模仿自然动作

#### `amp_loss_coef` 和 `amp_grad_penalty_coef` (判别器稳定性)
```python
amp_loss_coef: float = 0.1      # λ_AMP
amp_grad_penalty_coef: float = 0.01  # λ_GP
```
- **作用**：
  - `amp_loss_coef`：控制 AMP 对抗损失对 Actor 的影响
  - `amp_grad_penalty_coef`：梯度惩罚，防止判别器梯度过大
- **调优建议**：如果判别器训练不稳定（reward 震荡剧烈），增大 `amp_grad_penalty_coef`

---

## 网络架构超参数

控制 Actor、Critic 和 Discriminator 网络结构。

| 参数名 | 论文值 | 默认值 | 说明 |
|--------|--------|--------|------|
| `actor_hidden_dims` | [1024, 512] | [1024, 512] | Actor 网络隐藏层维度。 |
| `critic_hidden_dims` | [1024, 512] | [1024, 512] | Critic (Q) 网络隐藏层维度。 |
| `activation` | GELU | "gelu" | 激活函数。支持 elu, relu, gelu, tanh 等。 |
| `init_noise_std` | 1.0 | 1.0 | 动作噪声初始标准差。 |
| `action_scale` | 1.0 | 1.0 | 动作缩放因子。tanh 输出乘以此值。 |

### 详细说明

#### `activation` (激活函数)
```python
activation: str = "gelu"  # 论文使用 GELU
```
- **可选值**：`elu`, `relu`, `gelu`, `tanh`, `sigmoid`, `selu`, `lrelu`
- **论文选择**：Actor 和 Critic 使用 GELU，Discriminator 使用 ReLU
- **调优建议**：GELU 通常比 ReLU 更平滑，有助于策略学习

#### `actor_hidden_dims` / `critic_hidden_dims` (网络宽度)
```python
actor_hidden_dims: list = [1024, 512]
critic_hidden_dims: list = [1024, 512]
```
- **作用**：控制网络表达能力
- **权衡**：
  - 更宽/更深：表达能力更强，但训练更慢，容易过拟合
  - 更窄/更浅：训练更快，但可能欠拟合
- **调优建议**：对于复杂任务（多步态、复杂地形），保持或增大网络

---

## 训练配置超参数

控制整体训练流程。

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `seed` | 42 | 随机种子，用于复现结果。 |
| `device` | "cuda:0" | 计算设备。 |
| `num_steps_per_env` | 24 | 每个训练迭代中，每个环境收集的步数。 |
| `max_iterations` | 50000 | 最大训练迭代次数。 |
| `save_interval` | 100 | 模型保存间隔（迭代次数）。 |
| `empirical_normalization` | False | 是否使用经验归一化处理观测。 |
| `runner_class_name` | "AmpOffPolicyRunner" | 训练 Runner 类名。SAC 使用离策略 Runner。 |

### 详细说明

#### `num_steps_per_env` (收集步数)
```python
num_steps_per_env: int = 24
```
- **作用**：每次迭代每个环境收集的 transition 数量
- **总收集量**：`num_steps_per_env * num_envs = 24 * 4096 ≈ 100k` transitions/iteration
- **调优建议**：SAC 是离策略算法，此值可以较小（依赖 replay buffer）

---

## 环境配置超参数

控制仿真环境和任务设置。

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `num_envs` | 4096 | 并行环境数量。更多环境加速收集但增加 GPU 内存。 |
| `max_episode_length_s` | 20.0 | 每个 episode 最大时长（秒）。 |
| `action_scale` | 0.25 | 动作缩放因子，影响关节目标位置变化幅度。 |
| `dt` | 0.005 | 仿真时间步长（秒）。 |
| `decimation` | 4 | 策略相对于仿真的频率降采样因子。策略频率 = 1/(dt*decimation) ≈ 50Hz。 |

### 步态相关参数 (GaitCfg)

| 参数名 | Walk 值 | Run 值 | 说明 |
|--------|---------|--------|------|
| `gait_air_ratio_l` | 0.38 | 0.6 | 左腿腾空比例。 |
| `gait_air_ratio_r` | 0.38 | 0.6 | 右腿腾空比例。 |
| `gait_phase_offset_l` | 0.38 | 0.6 | 左腿相位偏移。 |
| `gait_phase_offset_r` | 0.88 | 0.1 | 右腿相位偏移。 |
| `gait_cycle` | 0.85 | 0.5 | 完整步态周期（秒）。Run 周期更短。 |

### 域随机化参数 (Domain Randomization)

| 参数名 | 范围 | 说明 |
|--------|------|------|
| `base_mass` | [-5, 5] kg | 基座质量随机偏移。 |
| `pd_gains_factor` | [0.9, 1.1] | PD 增益缩放因子。 |
| `friction_coefficient` | [0.25, 1.75] | 地面摩擦系数。 |

---

## 超参数调优建议

### 1. 内存不足 (CUDA OOM)

```python
# 方案 A: 减少并行环境数
num_envs = 2048  # 从 4096 减少

# 方案 B: 减少 batch size
batch_size = 8192  # 从 16384 减少

# 方案 C: 减少 replay buffer 大小
sac_replay_buffer_size = 1_000_000  # 从 10^7 减少到 10^6
```

### 2. 训练不稳定

```python
# 降低学习率
actor_lr = 3e-4
critic_lr = 3e-4

# 减小目标网络更新速度
tau = 0.01  # 从 0.05 减小

# 增加梯度惩罚
amp_grad_penalty_coef = 0.05  # 从 0.01 增加
```

### 3. 策略过于保守/探索不足

```python
# 增加初始熵温度
init_alpha = 0.5  # 从 0.2 增加

# 确保自动调节开启
auto_alpha = True
```

### 4. 动作不够自然/模仿效果差

```python
# 增加 AMP 奖励权重
amp_task_reward_lerp = 0.4  # 减小任务权重，增加模仿权重

# 增加 AMP 损失系数
amp_loss_coef = 0.2  # 从 0.1 增加
```

### 5. 训练太慢

```python
# 增加并行环境
num_envs = 8192

# 增加每步更新次数
updates_per_step = 16  # 从 8 增加

# 使用更大的 batch
batch_size = 32768
```

---

## 论文推荐配置

以下是论文 **Table I** 和 **Table II** 中的完整配置：

```python
@configclass
class RslRlSacAlgorithmCfg:
    """论文推荐的 SAC 配置"""
    class_name: str = "AMPSAC"
    
    # SAC 核心参数 (Table I)
    sac_replay_buffer_size: int = 10_000_000  # 10^7
    batch_size: int = 16384
    updates_per_step: int = 8
    n_step_return: int = 3
    tau: float = 0.05
    gamma: float = 0.99
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    warm_up_steps: int = 100
    auto_alpha: bool = True
    
    # AMP 参数 (Table II)
    amp_loss_coef: float = 0.1
    amp_grad_penalty_coef: float = 0.01
    amp_batch_size: int = 8192

@configclass
class RslRlSacActorCriticCfg:
    """论文推荐的网络配置"""
    class_name: str = "SACActorCritic"
    actor_hidden_dims: list = [1024, 512]  # GELU 激活
    critic_hidden_dims: list = [1024, 512]  # GELU 激活
    activation: str = "gelu"
```

---

## 参考文献

- Lessa, N. M., Boukheddimi, M., & Kirchner, F. (2025). *Unlocking the Potential of Soft Actor-Critic for Imitation Learning*. DFKI/University of Bremen.
- Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
- Peng, X. B., et al. (2021). *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*. ACM ToG.
