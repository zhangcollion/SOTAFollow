# RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework

**arXiv:** [2604.15308v1](https://arxiv.org/abs/2604.15308)

**Authors:** Hao Gao, Shaoyu Chen, Yifan Zhu, Yuehao Song, Wenyu Liu, Qian Zhang, Xinggang Wang

**Affiliation:** Huazhong University of Science & Technology, Horizon Robotics

---

## 1 一句话总结

RAD-2 提出 **Generator-Discriminator 框架**，用扩散模型生成多样化轨迹候选，RL 优化的判别器进行重排序，结合 TCR-GRPO 和 On-policy Generator Optimization 在 BEV-Warp 高吞吐仿真环境中实现大规模 RL 训练，碰撞率较扩散规划器降低 56%。

---

## 2 核心贡献

1. **Generator-Discriminator 统一框架**：扩散生成器产生多样轨迹候选，RL 判别器根据长期驾驶质量重排序，解耦设计避免直接在高维轨迹空间应用稀疏奖励
2. **TCR-GRPO (时序一致组相对策略优化)**：利用时序一致性缓解信用分配问题
3. **On-policy Generator Optimization**：将闭环反馈转换为结构化纵向优化信号，推动生成器向高奖励流形迁移
4. **BEV-Warp 高吞吐仿真环境**：通过空间扭曲直接在 BEV 特征空间进行闭环评估
5. **真机部署验证**：实际道路部署展示更好的感知安全性和驾驶平滑性

---

## 3 方法详述

### 3.1 问题定义

高等级自动驾驶需要能够建模多模态未来不确定性的运动规划器，同时在闭环交互中保持鲁棒性。现有扩散规划器的问题：
- 随机不稳定性
- 纯 IL 训练缺乏纠正性负反馈

### 3.2 算法框架

![RAD-2 Training Pipeline](https://hgao-cv.github.io/RAD-2/static/images/rad2-framev2.png)

RAD-2 采用 **Generator-Discriminator 架构**：

```
RAD-2 Pipeline:
┌─────────────────────────────────────────────────────────────┐
│  (a) Pre-training Stage                                     │
│      𝒢 (Diffusion Generator) 通过 IL 初始化                 │
│      学习专家演示的多模态轨迹分布                            │
├─────────────────────────────────────────────────────────────┤
│  (b) Closed-loop Rollout                                    │
│      联合策略：𝒢 生成 + 𝒟 选择                             │
│      在 BEV-Warp 环境中交互                                │
│      生成多样化 rollout 数据                                 │
├─────────────────────────────────────────────────────────────┤
│  (c) Discriminator Optimization                             │
│      𝒟 通过 TCR-GRPO 优化                                  │
│      利用闭环反馈增强评分精度                                │
├─────────────────────────────────────────────────────────────┤
│  (d) Generator Optimization                                  │
│      𝒢 通过 On-policy Generator Optimization 更新           │
│      结构化纵向优化推动分布向安全高效行为迁移                │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 核心组件

**Generator (𝒢) - 扩散轨迹生成器：**
- 输入：当前状态 + 场景表示
- 输出：多样化的轨迹候选集合
- 训练：模仿学习初始化 + On-policy Generator Optimization 调优

**Discriminator (𝒟) - Transformer 判别器：**
- 输入：轨迹候选 + 场景上下文
- 输出：每个候选的长期驾驶质量分数
- 训练：TCR-GRPO 算法优化

### 3.4 TCR-GRPO (Temporally Consistent Group Relative Policy Optimization)

利用时序一致性缓解信用分配问题。在时序上相关的轨迹组内进行相对优势估计，减少方差。

### 3.5 On-policy Generator Optimization

将闭环反馈（低奖励 rollout）转换为结构化纵向优化信号：
- 识别低奖励轨迹的特征
- 沿纵向（时间轴）调整生成器分布
- 逐步将生成器推向高奖励流形

### 3.6 BEV-Warp 仿真环境

高吞吐仿真环境，通过空间扭曲直接在 BEV 特征空间进行闭环评估：
- 高效：无需完整图像渲染
- 保真：基于真实 BEV 特征
- 可扩展：支持大规模并行训练

---

## 4 训练与推理伪代码

```python
# RAD-2 Generator-Discriminator 训练伪代码

class Generator(nn.Module):
    """Diffusion-based trajectory generator"""
    def __init__(self, ...):
        # Transformer + Diffusion backbone
        ...

    def generate(self, state, n_candidates=10):
        """Generate diverse trajectory candidates"""
        noise = sample_noise(n_candidates)
        trajectories = self.denoise(noise, state)
        return trajectories

class Discriminator(nn.Module):
    """Trajectory quality scorer"""
    def __init__(self, ...):
        # Transformer-based scorer
        ...

    def score(self, trajectories, state):
        """Score each trajectory by long-term driving quality"""
        return self.network(trajectories, state)

def train_rad2(generator, discriminator, env, num_iterations):
    # Stage 1: Generator IL Initialization
    for step in range(il_steps):
        expert_trajs = dataset.sample()
        loss = il_loss(generator(expert_trajs), expert_trajs)
        generator_optimizer.step(loss)

    # Stage 2: Joint Training
    for iteration in range(num_iterations):
        # (b) Closed-loop Rollout
        rollout_data = []
        for _ in range(num_workers):
            state = env.reset()
            for t in range(rollout_length):
                # Generator produces candidates
                candidates = generator.generate(state, n_candidates=10)
                # Discriminator re-ranks
                scores = discriminator.score(candidates, state)
                # Select best
                best_idx = argmax(scores)
                action = candidates[best_idx]
                next_state, reward, done = env.step(action)
                rollout_data.append({
                    'state': state,
                    'candidates': candidates,
                    'scores': scores,
                    'reward': reward,
                    'selected': best_idx
                })
                if done:
                    state = env.reset()
                    break
                state = next_state

        # (c) Discriminator Optimization via TCR-GRPO
        for _ in range(discriminator_steps):
            batch = sample(rollout_data, batch_size)
            advantages = compute_tcr_grpo_advantages(batch)  # TCR-GRPO
            loss_d = compute_discriminator_loss(discriminator, batch, advantages)
            discriminator_optimizer.step(loss_d)

        # (d) Generator Optimization via On-policy
        for _ in range(generator_steps):
            batch = sample(rollout_data, batch_size)
            # Extract low-reward rollouts for structured longitudinal optimization
            low_reward_trajs = [d['candidates'][d['selected']]
                               for d in batch if d['reward'] < threshold]
            loss_g = compute_on_policy_generator_loss(generator, low_reward_trajs)
            generator_optimizer.step(loss_g)

    return generator, discriminator
```

---

## 5 实验结论

### 5.1 主要结果

| Method | Collision Rate Reduction |
|--------|-------------------------|
| Diffusion-based planners (baseline) | 0% (reference) |
| **RAD-2** | **56% reduction** |

### 5.2 Real-world Deployment

- 改善的感知安全性
- 复杂城市场景中更好的驾驶平滑性

---

## 6 KnowHow（核心洞察）

1. **Generator-Discriminator 解耦的意义**：避免直接在稀疏高维轨迹空间应用 RL，通过判别器中介将优化问题分解为"生成-评估"两步

2. **扩散模型 + RL 的互补性**：扩散模型擅长建模多模态分布，RL 提供闭环反馈和长期信用分配

3. **TCR-GRPO 的动机**：时序一致的轨迹组内相对比较比独立样本比较方差更小

4. **On-policy Generator Optimization 的洞见**：从低奖励 rollout 中提取结构化信号，避免对生成器的直接稀疏奖励反向传播

5. **BEV-Warp vs 3DGS**：BEV 特征空间操作比完整图像渲染更高效，适合大规模训练

6. **真机部署意义**：仿真到真实的迁移验证，RAD-2 已验证于实际道路

---

## 7 总结

### 三大核心贡献

1. **Generator-Discriminator 统一框架**：扩散生成器 + RL 判别器，实现多模态轨迹生成与长期质量评估的联合优化
2. **TCR-GRPO + On-policy Generator Optimization**：新的 RL 算法组合解决信用分配和高维轨迹优化问题
3. **BEV-Warp 高吞吐仿真 + 真机部署**：高效大规模训练 + 真实世界验证

### 与 RAD 的关键差异

详见下方 **RAD vs RAD-2 对比章节**。

---

## RAD vs RAD-2 对比

| 维度 | RAD | RAD-2 |
|------|-----|-------|
| **核心架构** | 单策略网络 + PPO | Generator-Discriminator 框架 |
| **轨迹生成** | 解耦离散动作空间 (横向+纵向) | 扩散模型多模态轨迹生成 |
| **RL 算法** | PPO + GAE | TCR-GRPO + On-policy Generator Optimization |
| **训练环境** | 3DGS 照片级数字孪生 | BEV-Warp (BEV 特征空间) |
| **环境交互** | 照片级真实渲染 | 高吞吐 BEV 特征级仿真 |
| **优化目标** | 动作维度解耦优化 | 轨迹流形整体优化 |
| **碰撞率降低** | 3x vs IL 方法 | 56% vs 扩散规划器 |
| **训练阶段** | 三阶段 (感知预训练→规划预训练→RL微调) | 四阶段 (Generator IL→Rollout→Discriminator→Generator) |
| **真机部署** | 未提及 | 已验证真实道路部署 |

### 关键优化点对比

**RAD 的贡献：**
- 验证了 3DGS 作为 RL 训练环境的可行性
- 证明 RL+IL 协同可以解决 IL 的因果混淆问题
- 解耦动作空间设计降低探索复杂度

**RAD-2 的贡献（相对于 RAD 的改进）：**
- **Generator-Discriminator 架构**：将轨迹生成与质量评估解耦，支持多模态轨迹候选的联合优化
- **扩散模型替代离散动作**：更自然地建模连续轨迹分布，支持多样性和鲁棒性
- **TCR-GRPO 算法**：利用时序一致性减少信用分配方差
- **On-policy Generator Optimization**：结构化纵向优化避免直接在高维空间反向传播
- **BEV-Warp 替代 3DGS**：大幅提升训练吞吐量，降低计算成本
- **真机部署验证**：仿真到真实的迁移能力

### 技术演进路线

```
RAD (NeurIPS 2025)          RAD-2 (arXiv 2026)
    │                            │
    ▼                            ▼
单策略网络                  Generator-Discriminator
PPO + GAE                   TCR-GRPO + On-policy Opt
3DGS 环境                   BEV-Warp 环境
解耦离散动作                扩散多模态轨迹
IL 正则化                   Generator IL + Discriminator RL
```

### 论文关联

- RAD-2 是 RAD 的续作，延续了 "RL for Autonomous Driving" 的研究主线
- RAD 证明了 RL 在端到端自动驾驶中的可行性
- RAD-2 在此基础上探索了更高效的训练范式（Generator-Discriminator）和更大规模的部署（BEV-Warp + 真机）

---

**References:**
- [arXiv:2604.15308](https://arxiv.org/abs/2604.15308)
- [Project Page: hgao-cv.github.io/RAD-2](https://hgao-cv.github.io/RAD-2/)
- [RAD: 2502.13144](https://arxiv.org/abs/2502.13144)
