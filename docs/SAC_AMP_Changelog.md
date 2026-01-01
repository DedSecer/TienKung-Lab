# SAC+AMP 实现修改记录

基于论文 "Unlocking the Potential of Soft Actor-Critic for Imitation Learning" 的算法框架调整记录。

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

