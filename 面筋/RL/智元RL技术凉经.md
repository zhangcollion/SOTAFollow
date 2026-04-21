# 智元RL技术凉经

**Date**: 2026-04-20
**Topic**: 强化学习算法（PPO/SAC/CQL）与端到端自动驾驶
**Source**: Gemini Chat 问答整理

---

## 1. PPO 中 logp 的计算

### 离散动作空间
策略网络输出 logits，通过 Softmax 得到概率分布：
$$\log \pi(a|s) = z_{a_t} - \log \sum_{j=1}^N \exp(z_j)$$

### 连续动作空间
策略网络输出均值 $\mu$ 和标准差 $\sigma$，动作为高斯分布采样：
$$\log \pi(a_t|s_t) = -\frac{1}{2} \left( \frac{(a_t - \mu)^2}{\sigma^2} + \log(2\pi\sigma^2) \right)$$

### 关键细节
- **数值稳定性**：概率取对数避免浮点下溢
- **两类 logp**：`old_logp`（采样时保存用于重要性采样）vs `current_logp`（训练时实时计算）

---

## 2. RL 算法分类

| 类别 | 代表算法 | 特点 | 适用场景 |
|------|---------|------|---------|
| 策略梯度 | PPO, TRPO | 稳定，易实现 | 复杂连续/离散任务 |
| 价值基础 | DQN, Rainbow | 采样效率高 | 离散动作空间 |
| Actor-Critic | SAC, TD3 | 兼顾稳定性与估计能力 | 连续控制、机器人 |

### 重点算法补充
- **SAC**：最大熵强化学习，结合随机策略与价值函数，离线特性强
- **TD3**：DDPG 改进版，解决 Q 值过估计问题
- **CQL**：保守 Q 学习，用于离线 RL 防过估计

---

## 3. SAC 基本原理与实现

### 核心思想：最大熵目标函数
$$J(\pi) = \sum_{t} \mathbb{E}[r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$

其中 $H$ 为动作分布的熵，$\alpha$ 控制探索-利用权衡。

### 三大核心机制
1. **Twin Q-Networks**：维护两个 Q 网络，取最小值缓解过估计
2. **重参数化技巧**：$a = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$
3. **自适应温度系数**：$\alpha$ 作为可学习参数自动调节

### PyTorch 核心实现
```python
dist = Normal(mu, std)
u = dist.rsample()  # 重参数化采样
action = torch.tanh(u)

# Tanh 雅可比校正（关键！）
log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
```

---

## 4. SAC Actor 更新详解

### 目标：最小化 KL 散度
$$L_{actor}(\phi) = \mathbb{E}[\alpha \log \pi_\phi(a|s) - Q_\theta(s, a)]$$

### 重参数化技巧
- 网络输出确定性统计量 $\mu, \sigma$
- 噪声 $\epsilon$ 从外部注入，梯度可正常回传
- 采样操作被推到计算图边缘

### Tanh + 雅可比校正
有界动作空间需 Tanh 挤压，概率需变量代换校正：
$$\log \pi_\phi(a|s) = \log p(u|s) - \sum_{i=1}^D \log(1 - \tanh^2(u_i))$$

---

## 5. Online/Offline 混合训练中的失败数据

### 失败数据的价值
- 为 Critic 提供真实负向反馈，建立"安全护栏"
- 防止网络对未见过动作盲目乐观

### 关键机制：Terminal 状态处理
$$y = r + \gamma (1 - d) \max_{a'} Q_{target}(s', a')$$

失败状态 $d=1$ 时，切断对未来价值估计的传播。

### 工程策略
- **采样比例**：初期 70% Offline / 30% Online，逐步增加 Online
- **PER（优先经验回放）**：用 TD-Error 加权
- **Reward 截断**：对极端负反馈 Clipping 到 [-10, 10]
- **CQL**：防止分布外动作过估计

---

## 6. CQL 详解

### 损失函数
$$L_{CQL} = L_{TD} + \alpha \cdot (\log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a\sim D}[Q(s,a)])$$

### 物理含义
- **Push Down**：压低所有动作的 Q 值（尤其是 OOD 极值）
- **Pull Up**：抬高数据集中真实动作的 Q 值

### 连续动作空间的 OOD 动作构造
```python
# 均匀随机动作
random_actions = torch.rand(batch_size, action_dim) * 2 - 1
# 当前策略动作
current_actions, _ = actor(states)
# 下一状态动作
next_actions, _ = actor(next_states)

cat_q = torch.cat([q_random, q_current, q_next], dim=1)
ood_q_penalty = torch.logsumexp(cat_q, dim=1).mean()
```

---

## 7. PPO 对动作块的处理

### Macro-MDP
动作块长度 $K$：一次性预测未来 $K$ 步，宏观化 MDP。

### logp 计算
**独立高斯假设**：
$$\log \pi_\theta(A_t|s_t) = \sum_{i=0}^{K-1} \sum_{j=1}^{D} \log \mathcal{N}(a_{t+i,j}|\mu_{t+i,j},\sigma_{t+i,j})$$

**自回归**：
$$\log \pi_\theta(A_t|s_t) = \sum_{i=0}^{K-1} \log \pi_\theta(a_{t+i}|s_t, a_{t:t+i-1})$$

### 优势与代价
- **优势**：降低高频方差，避免动作抖动
- **代价**：高维探索灾难，需大量模仿学习预训练

---

## 8. 仿真中为什么不做人在环 RL

### 三大致命阻碍

1. **仿真加速比冲突**
   - RL 需 10x~100x 超实时并行训练
   - 人类在环强制 1x 实时，速度不可接受

2. **人类反应延迟破坏 MDP**
   - 生理延迟 200~300ms
   - 与 Action Chunking 冲突，时序对齐崩溃

3. **密集奖励下的人类疲劳**
   - 连续控制需密集奖励
   - 人类无法精确持续打分，方差巨大

### 工业界解法
- **离线接管数据化**：影子模式收集人类修正数据
- **轨迹级偏好优化**：DPO/RLHF 应用于序列决策

---

## 面试重点速记

| 问题 | 核心回答 |
|------|---------|
| PPO vs SAC | PPO on-policy 稳定，SAC off-policy 效率高、最大熵探索 |
| CQL 解决什么问题 | 离线 RL 中 OOD 动作过估计 |
| 动作块好处 | 降低方差、避免抖动、宏观化信用分配 |
| 为什么不用人在环 | 实时速度限制、延迟破坏 MDP、人类疲劳 |
