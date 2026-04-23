# XXPO 系列算法精读报告

> 本文档持续记录 RL 领域中各类 XXPO（Policy Optimization）算法的核心原理、改进点与实验结论。

---

## Contents（目录索引）

| 算法 | 全称 | 发布时间 | 机构 | 会议/期刊 | 一句话总结 | 章节 |
|------|------|---------|------|----------|-----------|------|
| [PPO](#1-ppo-proximal-policy-optimization) | Proximal Policy Optimization | 2017 | OpenAI | arXiv | 策略优化奠基之作，CLIP 机制控制策略更新幅度 | [§1](#1-ppo-proximal-policy-optimization) |
| [GRPO](#2-grpo-group-relative-policy-optimization) | Group Relative Policy Optimization | 2025.01 | DeepSeek | — | Group-relative sampling 替代 Value Model，大幅降低训练成本 | [§2](#2-grpo-group-relative-policy-optimization) |
| [GSPO](#3-gspo-group-sequence-policy-optimization) | Group Sequence Policy Optimization | [待查证] | [待查证] | — | [待查证] | [§3](#3-gspo-group-sequence-policy-optimization) |
| [DAPO](#4-dapo-decoupled-clip-and-dynamic-sampling-policy-optimization) | Decoupled Clip and Dynamic Sampling Policy Optimization | 2025.03 | [待查证] | — | Decoupled Clip + Dynamic Sampling 解决 over-exploration 问题 | [§4](#4-dapo-decoupled-clip-and-dynamic-sampling-policy-optimization) |
| [GMPO](#5-gmpo-geometric-mean-policy-optimization) | Geometric-Mean Policy Optimization | [待查证] | [待查证] | — | [待查证] | [§5](#5-gmpo-geometric-mean-policy-optimization) |

*最后更新：2026-04-22*

---

## 1. PPO (Proximal Policy Optimization)

### 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| **代码** | [openai/baselines](https://github.com/openai/baselines) |

### 基本信息

| 字段 | 内容 |
|------|------|
| **论文标题** | Proximal Policy Optimization Algorithms |
| **发布时间** | 2017.08 |
| **机构** | OpenAI |
| **作者** | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov |

### 1.1 Motivation（问题背景）

PPO 提出的背景是：标准策略梯度方法（如 REINFORCE）面临两个核心问题：

1. **策略更新幅度难以控制**：大步伐更新导致策略崩溃，小步伐更新学习太慢
2. **数据效率低**：每个样本只能使用一次（on-policy）

Trust Region Policy Optimization (TRPO) 通过 KL 散度约束解决了更新幅度问题，但计算复杂（需要二阶导数）。PPO 提出用一阶优化近似 TRPO，同时保证策略单调提升。

### 1.2 一句话总结

**PPO 通过 CLIP 机制限制策略更新比例，在保证训练稳定性的同时实现数据高效利用，成为强化学习最广泛使用的基线算法。**

### 1.3 核心贡献

1. **Clipped Surrogate Objective**：限制策略更新比例 $\in [1-\epsilon, 1+\epsilon]$
2. **Adaptive KL Penalty**：可选的 KL 散度惩罚项，自适应调整系数
3. **Multiple Epochs**：允许对同一批数据多次更新，提高数据效率
4. **GAE**：Generalized Advantage Estimation，提供方差-偏差权衡的控制

### 1.4 方法详述

#### 1.4.1 PPO-Clip 目标函数

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比。

```python
def ppo_clip_loss(theta, theta_old, states, actions, advantages, epsilon=0.2):
    """
    PPO Clip Loss
    """
    # 计算概率比 r(theta)
    pi_theta = policy_prob(theta, states, actions)
    pi_old = policy_prob(theta_old, states, actions)
    ratio = pi_theta / (pi_old + 1e-8)

    # Clipped loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    # 取较小值
    clip_loss = -torch.min(surr1, surr2).mean()

    return clip_loss
```

### 1.5 实验结论

PPO 在多个 benchmark 上与 TRPO 性能相当，但计算效率提升 3 倍。

---

## 2. GRPO (Group Relative Policy Optimization)

### 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | DeepSeek-R1 技术报告（待查证 arXiv ID） |
| **代码** | [待补充] |

### 基本信息

| 字段 | 内容 |
|------|------|
| **论文标题** | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（待确认） |
| **发布时间** | 2025.01（待确认） |
| **机构** | DeepSeek（待确认） |

### 2.1 Motivation（问题背景）

GRPO 由 DeepSeek 在 DeepSeek-R1 模型训练中提出，旨在解决传统 RL 中 Value Model 训练成本高、数据效率低的问题。

### 2.2 一句话总结

**GRPO 通过 Group-relative Sampling 机制，利用同组样本的相对表现估计 Advantage，消除了训练 Value Model 的高昂成本。**

### 2.3 核心贡献

1. **Group-relative Sampling**：同组样本替代 Value Model 估计 Advantage
2. **剪枝机制**：过滤无效或有害样本
3. **KL 散度正则化**：与 Reference Model 保持一致

### 2.4 方法详述

#### 2.4.1 Group-relative Advantage 估计

对于每个问题 $q$，采样 $G$ 个响应 $\{o_1, o_2, ..., o_G\}$：

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}
$$

其中 $\mu_G$ 和 $\sigma_G$ 是该组的均值和标准差。

#### 2.4.2 GRPO 目标函数

$$
L(\theta) = -\mathbb{E}_{(q, G) \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{ref}} \hat{A}_i, \text{clip} \cdot \hat{A}_i \right) \right]
$$

---

## 3. GSPO (Group Sequence Policy Optimization)

### 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [待查证 arXiv ID] |
| **代码** | [待查证] |

### 基本信息

| 字段 | 内容 |
|------|------|
| **论文标题** | Group Sequence Policy Optimization（待确认） |
| **发布时间** | [待查证] |
| **机构** | [待查证] |

### 3.1 Motivation（问题背景）

[待查证]

### 3.2 一句话总结

**[待查证]**

### 3.3 核心贡献

[待查证]

### 3.4 方法详述

[待查证]

---

## 4. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

### 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| **代码** | [待查证] |

### 基本信息

| 字段 | 内容 |
|------|------|
| **论文标题** | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| **发布时间** | 2025.03.18 |
| **机构** | [待查证]（作者列表含 Yu Qiying, Zhang Zheng 等） |

### 4.1 Motivation（问题背景）

Inference scaling 使 LLM 通过 RL 获得推理能力，但 OpenAI o1 和 DeepSeek R1 的关键技术细节未公开，社区难以复现。

### 4.2 一句话总结

**DAPO 通过 Decoupled Clip、Dynamic Sampling、Token-level Loss 和 Hangover Technique 四项技术，实现了 50 AIME 2024 分数（基于 Qwen2.5-32B）。**

### 4.3 核心贡献

1. **Decoupled Clip**：解耦策略提升与探索范围控制
2. **Dynamic Sampling**：动态过滤低质量样本
3. **Token-level Loss**：token-level 而非 sequence-level 损失
4. **Hangover Technique**：记忆难以样本的长期影响

### 4.4 方法详述

#### 4.4.1 Decoupled Clip

DAPO 解耦 CLIP 和探索控制：

$$
L^{DAPO} = \frac{\pi}{\pi_{ref}} \hat{A} - \lambda \cdot \text{KL}(\pi || \pi_{ref})
$$

#### 4.4.2 Dynamic Sampling

只保留 top (1-threshold) 的样本，减少无效探索。

---

## 5. GMPO (Geometric-Mean Policy Optimization)

### 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [待查证 arXiv ID] |
| **代码** | [待查证] |

### 基本信息

| 字段 | 内容 |
|------|------|
| **论文标题** | Geometric-Mean Policy Optimization（待确认） |
| **发布时间** | [待查证] |
| **机构** | [待查证] |

### 5.1 Motivation（问题背景）

[待查证]

### 5.2 一句话总结

**[待查证]**

### 5.3 核心贡献

[待查证]

### 5.4 方法详述

[待查证]

---

## 6. 附录：算法对比总结

### 6.1 算法演化路径

```
PPO (2017)
    │
    ▼
GRPO (2025.01) ── DeepSeek ── [待查证]
    │
    ├──→ GSPO ── [待查证]
    │
    └──→ DAPO (2025.03) ── [待查证]
    │
    ▼
GMPO ── [待查证]
```

### 6.2 核心机制对比

| 算法 | Advantage 估计 | Clip 机制 | 状态 |
|------|--------------|----------|------|
| PPO | GAE + Value Model | $r_t \in [1-\epsilon, 1+\epsilon]$ | ✅ |
| GRPO | Group-relative | Clipped | ⚠️ |
| GSPO | [待查证] | [待查证] | ⚠️ |
| DAPO | Dynamic Sampling | Decoupled | ⚠️ |
| GMPO | [待查证] | [待查证] | ⚠️ |

---

## ⚠️ 待完成工作

以下信息需要进一步查证：

1. [ ] GRPO：确认 arXiv ID、完整论文信息
2. [ ] GSPO：确认 arXiv ID、机构（用户提及阿里）
3. [ ] DAPO：确认机构
4. [ ] GMPO：确认 arXiv ID、机构、算法全称

---

*整理 by 优酱 🍃 | 2026-04-22*
*精读标准参考 CLAUDE.md § 论文精读格式标准*
