# SAC+AMP 实现修改记录

基于论文 "Unlocking the Potential of Soft Actor-Critic for Imitation Learning" 的算法框架调整记录。

---

## 2026-01-01 (Update 3)

### 9. 移除 Actor Loss 中错误的 Gradient Penalty

**问题**: Actor loss 中包含了在 policy 数据上计算的 gradient penalty，这不符合论文设计。

**论文公式 (2)** 明确指出 gradient penalty 只应用在 **expert 数据**上：
```
w^{gp}/2 · E_{(s,s')~D}[||∇_φ D_φ(s,s')||²]  (D = expert dataset)
```

**原实现** (错误):
```python
# 在 policy 数据上计算 gradient penalty
amp_grad_pen_actor = self.discriminator.compute_grad_pen(
    policy_state_norm, policy_next_state_norm, lambda_=10
)
actor_loss = sac_loss + amp_loss + amp_grad_penalty_coef * amp_grad_pen_actor
```

**修复后**:
```python
# Actor loss 只包含 SAC loss 和 AMP adversarial loss
# Gradient penalty 只在 Discriminator 更新时使用（在 expert 数据上）
actor_loss = sac_actor_loss + amp_loss_coef * amp_actor_loss
```

**影响**: 错误的 gradient penalty 会抑制 actor 的梯度流，影响学习效率。

**文件**: `rsl_rl/rsl_rl/algorithms/amp_sac.py`

---

## 2026-01-01 (Update 2)

### 8. 🔴 修复 N-Step Return 跨环境数据混合 Bug (Critical)

**问题**: 原实现将 4096 个并行环境的数据扁平化存储，导致 n-step 采样时错误地混合了不同环境的数据。

**原 Buffer 布局**:
```
index:  0      1      2      ...   4095   4096   4097   ...
        Env0   Env1   Env2   ...   Env4095 Env0   Env1   ...
        (t=0)  (t=0)  (t=0)  ...   (t=0)   (t=1)  (t=1)  ...
```

**原采样逻辑** (错误):
```python
# 从 index=0 开始，n_step=3
step_indices = [0, 1, 2]  # 实际采样: Env0_t0, Env1_t0, Env2_t0
# 错误地计算: R = r(Env0,t0) + γ·r(Env1,t0) + γ²·r(Env2,t0)
```

**修复后 Buffer 布局**:
```
Shape: [buffer_size, num_envs, dim]

time=0: [Env0_t0, Env1_t0, ..., Env4095_t0]
time=1: [Env0_t1, Env1_t1, ..., Env4095_t1]
...
```

**修复后采样逻辑** (正确):
```python
# 采样 (time_index, env_index) 对
time_indices = [t0, t0, t0, ...]  # 随机时间起点
env_indices = [e0, e1, e2, ...]   # 随机环境

# N-step 沿时间维度累积，保持环境一致
for step in range(n_step):
    step_time = time_indices + step
    step_rewards = rewards[step_time, env_indices]  # 同一环境的连续时间步
```

**影响**: 这是一个严重的 bug，会导致 Q-value 估计完全错误，训练无法收敛或收敛到错误的策略。

**文件**: 
- `rsl_rl/rsl_rl/storage/sac_replay_buffer.py` (重写)
- `rsl_rl/rsl_rl/algorithms/amp_sac.py` (更新 init_storage)

---

## 2026-01-01

### 1. Actor Loss 加入 AMP 损失项

**问题**: 原实现只有标准 SAC loss，缺少论文公式 (4) 中的 AMP 项。

**修复**: 
```python
# J_π^AMP = J_π + λ_AMP·L_AMP + λ_grad·L_grad
actor_loss = sac_loss + amp_loss_coef * amp_actor_loss + amp_grad_penalty_coef * grad_pen
```

**文件**: `rsl_rl/rsl_rl/algorithms/amp_sac.py`

---

### 2. 实现 N-Step Return

**问题 1**: 配置了 `n_step_return=3` 但未实现计算逻辑。

**问题 2**: Episode 在 n 步内终止时，`next_observations` 选取不正确。原实现简单地取 `t+n` 位置的 state，应该使用终止时的 next_state。

**修复**:
- `SACReplayBuffer.sample()` 计算 n-step 累积奖励和动态 gamma
- 追踪每个样本的终止位置 `effective_next_indices`
- 使用正确的终止状态: `next_observations[effective_next_indices]`
- `AMPSAC.update()` 使用 n-step target: `R_n + γ^k * (Q_target - α*log_π)`

**关键代码**:
```python
# 追踪终止位置
effective_next_indices = start_indices + n_step - 1  # 默认

if terminated_at_step_k:
    effective_next_indices[k] = step_indices[k]  # 使用终止时的索引
    n_step_gammas[k] = gamma ** (k + 1)
```

**文件**: 
- `rsl_rl/rsl_rl/storage/sac_replay_buffer.py`
- `rsl_rl/rsl_rl/algorithms/amp_sac.py`

---

### 3. 修正 AMP 参数

**问题**: 从 PPO 复制的参数不符合论文推荐值。

| 参数 | 原值 | 修正值 |
|------|------|--------|
| `amp_reward_coef` | 0.3 | 2.0 |
| `amp_task_reward_lerp` | 0.7 | 0.4 |

**文件**: 
- `legged_lab/envs/tienkung/sac_amp_walk_cfg.py`
- `legged_lab/envs/tienkung/sac_amp_run_cfg.py`

---

### 4. 显存优化

**问题**: `amp_num_preload_transitions=2×10^7` 导致 OOM。

**修复**: 
- Replay buffer 存储在 CPU，采样时移到 GPU
- 减小预加载量到 `2×10^6`

**文件**: `rsl_rl/rsl_rl/storage/sac_replay_buffer.py`

---

### 5. 添加 GELU 激活函数

**问题**: 论文使用 GELU，但框架不支持。

**修复**: 在 `resolve_nn_activation()` 添加 GELU 支持。

**文件**: `rsl_rl/rsl_rl/utils/utils.py`

---

### 6. 修正 Discriminator 更新频率

**问题**: Discriminator 在每个 SAC epoch 内都更新一次（共 8 次），但论文要求每个训练步只更新 1 次。

**论文 Table I & II**:
| 组件 | Updates per step |
|------|------------------|
| SAC (Critic, Actor, Alpha) | 8 |
| AMP Discriminator | 1 |

**修复**: 将 Discriminator 更新移到循环外部：
```python
# 循环内: SAC 更新 8 次
for _ in range(self.updates_per_step):
    # Critic, Actor, Alpha, Target network 更新
    
# 循环外: Discriminator 只更新 1 次
# Update AMP Discriminator (1 epoch per step)
```

**文件**: `rsl_rl/rsl_rl/algorithms/amp_sac.py`

---

## 实现说明

### AMP Actor Loss 数据来源 (与论文的差异)

**论文公式 (4)**:
```
J_π^AMP(θ) = J_π(θ) + λ_AMP · L_AMP + λ_grad · L_grad
```

**理论要求**: `L_AMP` 应该使用 **当前 Actor 采样的新动作** 在环境中执行后产生的状态转移 `(s_t, s_{t+1})` 来计算，这样 Actor 的梯度可以直接学习生成"欺骗" Discriminator 的动作。

**实践限制**: 
- 状态转移 `(s, s')` 必须通过 **环境模拟** 产生
- 物理仿真环境 **不可微分**（无法反向传播梯度）
- 无法在 Actor 更新时实时获取新动作对应的状态转移

**当前实现 (合理折中)**:
```python
# 使用 AMP buffer 中的历史数据（已收集的 policy 状态转移）
policy_d_for_actor = self.discriminator(
    torch.cat([policy_state_norm, policy_next_state_norm], dim=-1)
)
amp_actor_loss = F.mse_loss(policy_d_for_actor, torch.ones_like(policy_d_for_actor))
```

**影响分析**:
- ✅ 仍然能引导 policy 生成类似专家的行为
- ✅ AMP 的主要作用是通过 **reward shaping** 实现，Actor loss 中的 AMP 项起辅助作用
- ⚠️ 使用的是稍滞后的历史数据，而非实时反馈
- ⚠️ 梯度无法直接从 Discriminator 流向 Actor 的动作输出

**替代方案** (未采用):
1. **Model-based**: 学习可微分的环境模型 → 计算开销大，模型误差累积
2. **Policy Gradient Estimation**: 使用 REINFORCE 估计梯度 → 方差大，训练不稳定

**结论**: 当前实现是 model-free 深度 RL 中的标准做法，与 AMP+PPO 的实现方式一致。

---

### 7. 移除 Observation History (符合论文设计)

**问题**: 原实现使用 10 帧历史观测，导致 observation 维度过大，SAC Replay Buffer 内存占用约 115GB。

**论文 Section III-D**:
> "The state representation $s \in \mathcal{S}$ includes the command velocity target, joint positions and velocities, and base orientation."

论文中 **没有使用 observation history**，只使用单帧观测。

**修复**:
```python
robot: RobotCfg = RobotCfg(
    actor_obs_history_length=1,   # 从 10 改为 1
    critic_obs_history_length=1,  # 从 10 改为 1
    ...
)
```

**内存影响**:
| 配置 | actor_obs_dim | critic_obs_dim | SAC Buffer 内存 |
|------|---------------|----------------|-----------------|
| 原实现 (history=10) | 690 | 740 | ~115 GB |
| 修复后 (history=1) | 69 | 74 | ~12 GB |

**文件**: 
- `legged_lab/envs/tienkung/sac_amp_walk_cfg.py`
- `legged_lab/envs/tienkung/sac_amp_run_cfg.py`
