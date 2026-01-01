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

**问题**: 配置了 `n_step_return=3` 但未实现计算逻辑。

**修复**:
- `SACReplayBuffer.sample()` 计算 n-step 累积奖励和动态 gamma
- `AMPSAC.update()` 使用 n-step target: `R_n + γ^n * (Q_target - α*log_π)`

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

