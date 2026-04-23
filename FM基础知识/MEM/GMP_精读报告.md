# Gated Memory Policy (GMP) 精读报告

> **论文**：Gated Memory Policy: Learning When and What to Recall for Visuomotor Manipulation
> **arXiv**：2604.18933
> **作者**：Yihuai Gao, Jinyun Liu, Shuang Li, Shuran Song
> **机构**：Stanford University, Carnegie Mellon University
> **顶会/顶刊**：arXiv 2026 (cs.RO)
> **发布日期**：2026-04-21
> **代码/项目**：[gated-memory-policy.github.io](https://gated-memory-policy.github.io/)

---

## 一句话总结

GMP 通过**学习何时召回记忆**（Memory Gate）+ **学习召回什么**（Cross-Attention History Conditioning）+ **扩散噪声增强鲁棒性**三重设计，让视觉运动策略在无需人工设计规则的情况下，自适应地在 Markov 任务和 Non-Markov 任务之间切换，在 MemMimic 基准上相对长历史基线提升 **30.1%** 成功率。

---

## 核心贡献

1. **Gated Memory Policy**：首个同时解决"何时召回"和"召回什么"的视觉运动策略，在 Markov 和 Non-Markov 任务上均保持 SOTA 性能
2. **自监督 Gate 校准**：通过对比"有/无记忆"两策略的动作预测误差生成二值监督信号，避免端到端训练的梯度冲突，无需手工调参
3. **Cross-Attention 时序融合**：替代自注意力，复杂度从 \(\mathcal{O}(H^{2})\) 降至 \(\mathcal{O}(h \times H)\)，配合 KV Cache 实现线性历史检索
4. **扩散噪声一致性**：训练/推理均对历史动作加噪，避免过度依赖干净历史，显著提升鲁棒性
5. **MemMimic 基准**：首个覆盖 In-Trial 和 Cross-Trial 记忆评估的机器人操作基准

---

## 方法详述

### 问题定义

机器人操作任务按记忆需求分为三层：

| 任务类型 | 记忆跨度 | 示例 |
|---------|---------|------|
| Markov 任务 | 无需历史 | 简单抓取，单帧视觉+本体感受即可 |
| In-Trial Memory | 单次执行内 | Match Color 需记住箱子初始颜色 |
| Cross-Trial Memory | 跨试次总结 | Iterative Pushing 从推距推断摩擦力 |

**核心挑战**：一个好的记忆增强策略必须同时回答两个问题：
- **何时召回？**（When to recall）：在大多数时刻，记忆是噪音，策略应忽略它
- **召回什么？**（What to recall）：无法靠人工规则指定，必须端到端学习

### 整体 Pipeline

```
当前时间步 t
    │
    ├──→ ViT Encoder ─→ 聚合图像 token
    │
    ├──→ Memory Gate MLP(φ) ─→ μt ∈ {0,1}
    │
    └──→ DiT Hidden States (Query)
              │
              ▼
         ┌─────────────────────────┐
         │  Cross-Attention        │
         │  Query ← 历史 KV Cache  │
         └─────────────────────────┘
              │
              ▼
         历史上下文 ht:t+h
              │
              ▼
         z̄t:t+h = μt · ht:t+h + zt:t+h  (门控融合)
              │
              ▼
         DiT 去噪预测
```

### 核心数学公式

**1. Diffusion Policy 损失函数**（基础）：

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{A^0_{t:t+h}, \epsilon, k} \left[ \left\| A^0_{t:t+h} - \varphi_\theta\left(A^k_{t:t+h}, I_t, P_t, k\right) \right\|_2^2 \right]$$

**2. Memory Gate 二值门控**：

$$\mu_t = \mathbf{1}\{ \sigma(\phi(I_t, P_t)) > 0.5 \} \in \{0, 1\}$$

**3. 门控融合输出**：

$$\bar{\mathbf{z}}_{t:t+h} = \mu_t \cdot \mathbf{h}_{t:t+h} + \mathbf{z}_{t:t+h}$$

**4. Gate 校准阈值判断**：

$$\mu_t = \begin{cases} 1 & \text{if } \delta_t \geq \theta \cdot \delta_t^{\text{mem}} \\ 0 & \text{otherwise} \end{cases}$$

> **图 1：机器人操作任务的三层记忆需求**（对应论文 Figure 1）
>
> ![GMP 任务概览](https://arxiv.org/html/2604.18933v1/figures/jpg/teaser.jpg)
>
> - **(a) Markov 任务**：如简单抓取，无需历史，当前感知即可完成
> - **(b) In-Trial Memory**：单次执行内需要记忆上下文，例如 Match Color 任务需记住箱子初始颜色
> - **(c) Cross-Trial Memory**：跨 Trial 总结物理属性（如摩擦力、质量），通过试错迭代调整动作

> **图 2：GMP 网络架构**（对应论文 Figure 2）
>
> ![GMP 方法架构](https://arxiv.org/html/2604.18933v1/figures/jpg/network.jpg)
>
> - **左 (a)**：基于 DiT 的完整 GMP 框架，额外增加了 Gated Attention 模块
> - **右 (b)**：Gated Attention 模块三要素：① 二值门控 \(\mu_{t}\) 决定是否执行历史交叉注意力；② 加噪历史动作条件提升鲁棒性；③ KV Cache 缓存历史 token 降低计算成本

---

## 训练与推理伪代码

### 阶段一：Memory Gate 校准（Calibration）

```python
# ============================================================
# 阶段一：Memory Gate 校准
# 输入：数据集 D_split = D_train ∪ D_val
# ============================================================

# Step 1: 划分数据集
D_train, D_val = split_dataset(D, ratio=0.5)

# Step 2: 在 D_train 上分别训练两个策略
pi_always_off = train_policy(D_train, gate_mode="off")     # π, μt=0 始终
pi_always_on  = train_policy(D_train, gate_mode="on")     # π_mem, μt=1 始终

# Step 3: 在 D_val 上采样，计算各时刻动作预测误差
trajectories_off = sample(pi_always_off, D_val, N_rounds=N)
trajectories_on  = sample(pi_always_on, D_val, N_rounds=N)

errors_off = compute_errors(trajectories_off)   # δ_t: 无记忆策略在时刻 t 的误差
errors_on  = compute_errors(trajectories_on)    # δ_t^mem: 有记忆策略在时刻 t 的误差

# Step 4: 生成 Gate 标签
theta = 1.5  # 比率阈值（人工设定）
gate_labels = (errors_off >= theta * errors_on).float()  # 1=需要记忆, 0=不需要

# Step 5: 单独训练 Memory Gate MLP
gate_mlp = MLP(input_dim=dim(I_t)+dim(P_t), hidden=64, output=1, activation="sigmoid")
optimizer = Adam(gate_mlp.parameters(), lr=1e-3)

for epoch in range(num_gate_epochs):
    for batch in D_train:
        I_t, P_t = batch["image"], batch["proprio"]
        mu_pred  = gate_mlp(I_t, P_t)           # 预测概率
        mu_label = gate_labels[t]                # 真实标签
        loss_gate = BCE(mu_pred, mu_label)
        optimizer.zero_grad()
        loss_gate.backward()
        optimizer.step()

# 校准完成：冻结 Gate 权重
freeze(gate_mlp)
```

### 阶段二：完整策略微调（Final Policy Training）

```python
# ============================================================
# 阶段二：Gated Memory Policy 完整训练
# 输入：完整数据集 D = D_train ∪ D_val，标定好的 Gate MLP
# ============================================================

pi_gated = train_policy(D_full, gate_mode="calibrated", gate_mlp=gate_mlp)

# 训练目标
for batch in D_full:
    I_t, P_t, actions, k = batch["image"], batch["proprio"], batch["actions"], batch["diffusion_step"]

    # Memory Gate 前向
    mu_t = gate_mlp(I_t, P_t)   # 二值门: 0 或 1

    # 历史 KV Cache（滑动窗口）
    history_kvcache = build_kvcache(batch["history"], window_size=n)

    # 加噪历史动作条件（扩散调度）
    A_history_noised = add_diffusion_noise(A_history, k-1)

    # Cross-Attention 历史查询
    h_t = cross_attention(query=z_t, kv=history_kvcache)   # 历史上下文

    # 门控融合
    if mu_t == 1:
        z_bar = h_t + z_t   # 融合历史
    else:
        z_bar = z_t          # 跳过历史

    # 去噪预测
    A_pred = denoiser(z_bar, A_noisy, k)
    loss = MSE(A_pred, A_gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 输出：π_gated（既能在 Markov 任务上关闭记忆，又能在 Non-Markov 任务上按需召回）
```

---

## 实验结论

### 5.1 主实验结果：MemMimic 基准

GMP 提出了 MemMimic，一个专门评估视觉运动策略记忆能力的基准，包含 6 个任务：

| 任务 | 类型 | 描述 |
|------|------|------|
| T1: Match Color | In-Trial | 记住初始箱子颜色，放置立方体 |
| T2: Discrete Place Back | In-Trial | 记住原位置，悬空 2 秒后放回 |
| T3: Continuous Place Back | In-Trial (Real) | 真实机器人，杯子放回原位 |
| T4: Iterative Pushing | Cross-Trial | 从推距推断摩擦力，动态调整推力 |
| T5: Iterative Flinging | Cross-Trial | 从甩布结果推断质量，调整甩力 |
| T6: Iterative Casting | Cross-Trial (Real) | 从滑行距离推断摩擦系数 |

**关键结果**：GMP 相对长历史基线提升 **30.1%** 平均成功率。

### 5.2 消融实验

| 消融项 | 发现 |
|--------|------|
| **Calibration vs STE** | Calibration 显著优于 Straight-Through Estimator；STE 倾向于让门值趋向 1（过度使用记忆） |
| **Calibration vs 正则项** | 正则项方法对权重敏感，难以在 Markov/Non-Markov 两类任务上同时 work |
| **扩散噪声方案** | No Noise / Random Level / Diffusion Forcing 三种方案均不如 GMP 的 Diffusion Noising 方案 |

### 5.3 鲁棒性分析

- **计算效率**：120 步历史自注意力需要 ~0.5s/步，GMP (Gate Off) 仅需 ~0.05s/步，与无历史策略相当
- **长记忆测试**：T1' Match Color with Random Delay（5-600 秒随机延迟），GMP 在 6000 帧记忆 buffer 下达到 99.0%±1.0% 成功率，推理仅需 0.16 秒

### 5.4 定性分析

> **图 3：Cross-Attention 注意力可视化**（对应论文 Figure 4）
>
> ![GMP 注意力可视化](https://arxiv.org/html/2604.18933v1/figures/jpg/exp1_match_color.jpg)
>
> 上图展示了 Match Color 任务（t=80 时刻放置立方体，注意力落在 t=48 首次观察箱子颜色）和 Iterative Pushing 任务（第 4 次推动注意力集中在第 2、3 次推结果）。蓝色 = \(\mu_{t}=1\)（门开），灰色 = \(\mu_{t}=0\)（门关）

**关键发现**：
- Match Color 任务中，t=80 时刻放置立方体时，注意力权重最高点落在 t=48（首次观察箱子颜色的时刻）
- Iterative Pushing 中，第 4 次推的时刻，注意力集中在第 2、3 次推的结果（过推/欠推）

---

## KnowHow（核心洞察）

1. **Memory Gate 校准的 insight**：用"动作预测误差对比"这一自监督信号替代端到端训练中的正则项，避免了手工调参，优雅地解决了"何时用记忆"的问题。核心洞察是：**让数据本身告诉我们哪些时刻真正需要记忆**。

2. **Cross-Attention 替代自注意力的 insight**：历史 token 不需要与当前 token 做完整自注意力——历史信息只需要被"查询"，而不需要彼此交互。这将复杂度从 \(\mathcal{O}(H^{2})\) 降为 \(\mathcal{O}(h \times H)\)，对 \(H\) 是**线性**而非二次。

3. **门控的运行特点**：即使在 Non-Markov 任务上，门也**大部分时间关闭**（Match Color 中 73% 时刻关闭，Iterative Pushing 中 58% 关闭），仅在关键时刻（如需要回忆初始颜色、回顾上次推力结果时）才打开。这说明记忆应当是"稀缺资源"而非"默认选项"。

4. **扩散噪声一致性的 insight**：Diffusion Forcing 在训练时加随机噪声，但推理时不加噪声，导致训练-推理不一致。GMP 在训练和推理**都使用扩散噪声**，保证了两者的一致性。

5. **为什么端到端训练门控不可行？**：如果把门控和策略一起端到端训练，无正则时策略倾向于尽可能多地使用历史（Markov 任务严重过拟合）；加正则时如果权重过大，门值趋向于始终为 0（在 Non-Markov 任务上完全丧失记忆能力）。很难找到一个正则权重在两类任务上都 work。

6. **KV Cache 的关键优势**：推理时，如果 Gate 关闭，则完全跳过历史注意力，推理时间与无历史策略相同。零额外计算负担使得 GMP 可以处理任意长度历史而不影响推理延迟。

7. **视觉特征压缩的设计**：用 Multi-Head Attention Pooling (MAP) 将所有 patch token 聚合为**单一 token**，大幅压缩视觉 token 数量，每个历史时间点只对应一个聚合图像 token + \(h\) 个动作 token。

8. **校准流程的两阶段设计**：先固定 Gate 训练两个极端策略（常开/常闭），再基于误差对比生成监督信号训练 Gate，最后冻结 Gate 重新训练策略。这个流程避免了梯度冲突，让每个模块各自优化到最优。

---

## arXiv Appendix 关键点总结

**A. 超长记忆测试（T1' Match Color with Random Delay）**
- 5-600 秒随机延迟测试记忆长度
- GMP 在 6000 帧记忆 buffer 下达到 99.0%±1.0% 成功率
- 推理仅需 0.16 秒（8 步去噪，5090 GPU）

**B. Gate 校准消融（Finding 4）**
- 验证了 Calibration 策略优于 STE（Straight-Through Estimator）和正则项两种替代方案
- 正则项方法对权重敏感，难以找到通用权重
- STE 倾向于让门值趋向 1（过度使用记忆）

**C. 推理时间对比（Finding 5）**
- 在 RTX 3080 上，120 步历史自注意力需要 ~0.5s/步
- GMP (Gate Off) 仅需 ~0.05s/步，与无历史策略相当
- Gate 关闭时完全跳过历史注意力，零额外计算

**D. 噪声注入消融（Finding 6）**
- No Noise / Random Level / Diffusion Forcing 三种方案对比
- Diffusion Noising（GMP 方案）在 Iterative Pushing 上显著优于所有基线
- 验证了训练-推理噪声一致性的重要性

**E. 更多实验任务**
- exp2_discrete_place_back: 悬空 2 秒后放回原位
- exp3_continuous_place_back: 真实机器人实验，杯子放回原位
- exp7_robomimic: Markov 任务验证，GMP 与无历史策略性能相当

**F. 补充材料图**
- supp_in_the_wild_data.jpg: 真实场景数据采集
- supp_casting_pos_control.jpg: 位置控制可视化
- supp_gate_label_statistics.jpg: Gate 标签统计（73%/58% 门关闭比例验证）

---

## 总结

**3 大核心贡献**：
1. **Gated Memory Policy 架构**：首次同时解决"何时召回"和"召回什么"两个子问题，在 Markov 和 Non-Markov 任务上均保持 SOTA 性能
2. **自监督 Gate 校准**：通过两阶段误差对比生成监督信号，避免端到端训练的梯度冲突，无需手工调参
3. **Cross-Attention + 扩散噪声一致性**：复杂度从 \(\mathcal{O}(H^{2})\) 降至 \(\mathcal{O}(h \times H)\)，训练/推理噪声一致保证鲁棒性

**最重要洞察**：
GMP 的核心贡献不是某个新架构，而是**对"记忆"这件事的精准问题分解**：When/What/How Robust 三个子问题分别用 Gate/Cross-Attention/Diffusion Noise 解决，干净利落。尤其是 Gate 校准流程，用两个固定策略的误差对比来生成监督信号，避免了端到端训练的梯度冲突——这个思路非常值得借鉴到其他需要"选择性使用信息"的场景（如 attention 机制、skip connection、 gating network 等）。

---

## 参考链接

- **论文**：https://arxiv.org/abs/2604.18933
- **arXiv HTML**：https://arxiv.org/html/2604.18933v1
- **项目主页**：https://gated-memory-policy.github.io/
- **HuggingFace 模型**：https://huggingface.co/yihuai-gao/gated-memory-policy
- **HuggingFace 数据集**：https://huggingface.co/datasets/yihuai-gao/gated-memory-policy