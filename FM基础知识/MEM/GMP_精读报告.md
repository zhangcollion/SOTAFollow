# Gated Memory Policy (GMP) 精读报告

> **论文**：Gated Memory Policy: Learning When and What to Recall for Visuomotor Manipulation
> **arXiv**：2604.18933
> **作者**：Yihuai Gao et al.
> **机构**：Stanford University, Carnegie Mellon University, 及其他
> **顶会/顶刊**：arXiv 2026 (cs.RO)
> **发布日期**：2026-04-21
> **代码/项目**：[gated-memory-policy.github.io](https://gated-memory-policy.github.io/)

---

## 一句话总结

GMP 通过**学习何时召回记忆**（Memory Gate）+ **学习召回什么**（Cross-Attention History Conditioning）+ **扩散噪声增强鲁棒性**三重设计，让视觉运动策略在无需人工设计规则的情况下，自适应地在 Markov 任务和 Non-Markov 任务之间切换，在 MemMimic 基准上相对长历史基线提升 **30.1%** 成功率。

---

## 拟人化开篇

想象你让机器人去完成一个任务：把杯子放回原处。

如果你只看它**现在**看到的样子——杯子在桌上——你不知道它之前有没有被移动过、它原本在哪里。这时候"记忆"就有用了：你需要记住之前看到的一切，才能把杯子准确放回去。

但问题是：记忆也会帮倒忙。

对于一个简单任务（比如抓起一个杯子），如果你脑子里堆满了过去 100 个时间步的所有历史画面，反而会让你反应变慢、容易分心。更糟糕的是，这些历史画面里可能夹杂着噪声或错误信息，会把你的判断带偏。

人类是怎么解决这个问题的？我们会**本能地判断"现在需不需要调用记忆"**：放杯子的时候，我们会快速回忆一下"杯子原来在哪儿"；抓起杯子的瞬间，这类简单动作根本不需要回忆历史。

**GMP 就是在让机器人学会这个能力。**

---

## 背景与问题动机

### 机器人操作任务的三层记忆需求

并非所有任务都同样需要记忆。按记忆跨度从短到长，可分为三层：

1. **Markov 任务**：如简单抓取，几乎不需要历史。单帧视觉 + 当前本体感受就能完成任务。
2. **Trial 内记忆（In-Trial Memory）**：单次执行中需要回顾上下文。例如，把立方体放进颜色匹配的箱子里——你需要记住"拿起立方体之前，箱子们各自是什么颜色"，但立方体被拿起后箱子颜色会打乱，你必须记住初始颜色。
3. **跨 Trial 记忆（Cross-Trial Memory）**：需要从多次交互中总结物理属性。例如，反复推物体感知摩擦力，或反复甩布感知质量——你需要从过去几次尝试的**结果**中推断物理参数，再调整下一次动作。

### 朴素方案及其代价

最直接的做法是：**扩展历史窗口**——把更多的过去动作和视觉观测一起输入给策略。

但这会带来两个严重问题：

- **过拟合**：历史越长，输入维度越大，训练数据不变的情况下模型越容易过拟合，尤其在 Markov 任务上性能断崖下跌。
- **计算成本爆炸**：在 Transformer 架构中，对 H 步历史做自注意力的复杂度是 O(H²)，随着历史长度增加，训练和推理成本急剧上升。

### 核心挑战

一个好的记忆增强策略必须同时回答两个问题：

> **何时召回？**（When to recall）—— 在大多数时刻，记忆是噪音，策略应该忽略它。  
> **召回什么？**（What to recall）—— 复杂操作任务无法靠人工规则指定"存储什么、检索什么"，必须端到端学习。

---

## 方法详解

### 3.1 基础：Transformer-based Diffusion Policy

GMP 以 **Diffusion Transformer (DiT)** 为骨干网络。在标准 Diffusion Policy 中，给定当前图像 I_t 和机器人本体感受 P_t，策略预测未来 h 步的动作轨迹：

A_{t:t+h} = {A_t, A_{t+1}, ..., A_{t+h-1}}

训练时，对真实动作 A^0_{t:t+h} 逐步加高斯噪声 epsilon ~ N(0, I)，用去噪网络 phi_theta 预测原始动作，损失函数为：

L_action = E[A^0_{t:t+h}, epsilon, k] [ ||A^0_{t:t+h} - phi_theta(A^k_{t:t+h}, I_t, P_t, k)||² ]

其中 k 是扩散步数。推理时用 DDIM 调度器去噪 K 步。

**问题**：标准 Diffusion Policy 仅利用最近 1-2 帧观测，无法处理需要长历史的任务。

---

### 3.2 时序融合模块：Gated Cross-Attention（核心重点）

这是 GMP 最关键的技术创新，也是本次精读重点展开的部分。

#### 3.2.1 历史轨迹的表示与存储

对于 n 个历史动作块，每个块包含 h 个连续动作：

{A_{t-nh:t-(n-1)h}, ..., A_{t-h:t}}

每个动作块对应一个采样的图像观测（高频帧存在大量冗余，因此每块只采样一帧）：

I_{t-nh:h:t} = {I_{t-nh}, I_{t-(n-1)h}, ..., I_{t-h}}

**视觉特征提取**：用预训练 **ViT encoder**（SigLIP-B/16）提取每个图像的特征，然后通过 **Multi-Head Attention Pooling (MAP)** 将所有 patch token 聚合为**单一 token**，大幅压缩视觉 token 数量。

**编码后**：每个历史时间点对应一个聚合图像 token + h 个动作 token，组成该时间点的 memory chunk。

#### 3.2.2 Cross-Attention 机制（而非自注意力）

如果直接把历史 token 与当前 token 在序列维度上拼接，用标准**自注意力**处理，计算复杂度为 O(H²)，H 是历史 token 总数。

GMP 的核心洞察是：**历史 token 不需要与当前 token 做完整自注意力**。历史信息只需要被"查询"，而不需要彼此交互。

因此，GMP 引入一个**轻量级 Cross-Attention 模块**：

```
当前输入（Query）← Cross-Attention ← 历史 Memory KV Cache
```

- **Query** 来自 DiT 的当前 hidden states：z_{t:t+h} ∈ R^{h×d}
- **Key / Value** 来自缓存的历史 chunk（每个 chunk = 1 个聚合图像 token + h 个动作 token）

通过交叉注意力，Query attend 到历史中最相关的信息，输出历史上下文向量 h_{t:t+h} ∈ R^{h×d}。

**关键优势**：Cross-Attention 的复杂度为 O(h×H)，对 H 是**线性**而非二次。推理时，如果 Gate 关闭，则完全跳过历史注意力，推理时间与无历史策略相同。

#### 3.2.3 KV Cache：滑动窗口缓存

推理时，历史 token 无法一次性全部输入，必须逐步构建。

GMP 参考因果 Transformer 中的 **KV Cache** 机制，使用**滑动窗口**缓存 n 个最近的历史 chunk：

- 每个 chunk = 1 个聚合图像特征 + h 个动作 token
- 新时间步到来时：将最新 chunk 追加到窗口，窗口满时移除最旧的 chunk
- **零额外计算**：KV Cache 使得每次推理时无需重新对历史做前向编码，直接从缓存读取

#### 3.2.4 完整时序融合流程

整个时序融合模块的工作流程如下（对应论文 Figure 2）：

```
Step 1. 当前时间步 t：
  输入 → 当前图像 It + 本体感受 Pt

Step 2. 视觉编码：
  It → ViT encoder → 聚合图像 token

Step 3. Memory Gate 计算（MLP + Sigmoid）：
  μt = 1{ σ( φ(It, Pt) ) > 0.5 } ∈ {0, 1}
  如果 μt = 0：跳过历史注意力，直接使用无记忆输出
  如果 μt = 1：执行下一步

Step 4. Cross-Attention 历史查询：
  当前 Query (DiT hidden states)
      ↕ Cross-Attention
  历史 KV Cache (n 个 chunk × (1 图像 token + h 动作 token))

Step 5. 扩散噪声调节：
  历史动作：A_{t-nh:t}^{k-1}（噪声步 k-1，比预测目标的噪声步 k 低一级）
  早期步（k 大）→ 强噪声注入
  晚期步（k 小）→ 弱噪声

Step 6. 门控融合：
  z̄_{t:t+h} = μt · h_{t:t+h} + z_{t:t+h}
  结果送入 DiT 前馈层
```

---

### 3.3 Memory Gate：学习何时召回记忆

#### 3.3.1 门控机制

给定当前观测 I_t 和本体感受 P_t，用一个 MLP phi（含 Sigmoid 激活）生成二进制门值 μt：

μt = 1{ σ( φ(I_t, P_t) ) > 0.5 } ∈ {0, 1}

门值 μt 作用在历史 Cross-Attention 的输出 h_{t:t+h} 上，决定是否将历史信息融入当前输出：

z̄_{t:t+h} = μt · h_{t:t+h} + z_{t:t+h}

- μt = 0：历史信息被完全阻断，等同于无记忆策略
- μt = 1：历史信息被融合进策略输出

#### 3.3.2 为什么端到端训练门控不可行？

如果把门控和策略一起端到端训练（加 BCE Loss 或 Straight-Through Estimator），会发现：

- **无正则时**：策略倾向于尽可能多地使用历史，因为这样通常能降低训练损失。结果是在 Markov 任务上严重过拟合。
- **加正则时**：如果正则权重过大，门值趋向于始终为 0，在 Non-Markov 任务上又完全丧失记忆能力。

**困境**：很难找到一个正则权重在两类任务上都 work。

#### 3.3.3 自监督校准流程（Calibration）

GMP 提出了一个巧妙的**两阶段自监督校准**方法，核心思想是：**让数据本身告诉我们哪些时刻真正需要记忆**。

**Step 1**：将数据集划分为 D_train 和 D_val。

**Step 2**：在 D_train 上分别训练两个策略：
- π：Memory Gate 始终关闭（μt = 0，无记忆）
- π_mem：Memory Gate 始终打开（μt = 1，强制记忆）

**Step 3**：在 D_val 上对两个策略各采样 N 条轨迹，计算每个时间步 t 的动作预测误差：
- δt：无记忆策略 π 的误差
- δt^mem：有记忆策略 π_mem 的误差

**Step 4**：引入比率阈值 θ，生成门标签：

μt = 1（记忆必要） if δt ≥ θ · δt^mem
μt = 0 otherwise

即：如果某个时间步**无记忆时的误差明显大于有记忆时的误差**，说明记忆在此刻是必要的。

**Step 5**：用 Binary Cross Entropy Loss 单独训练 Memory Gate MLP φ。

**Step 6**：冻结 Memory Gate 参数，在完整数据集上重新训练策略，得到最终 **Gated Memory Policy** π_gated。

#### 3.3.4 门控的运行特点

实验发现：
- 在 Markov 任务上，门几乎始终关闭（μ = 0），因为记忆不影响性能
- 即使在 Non-Markov 任务上，门也**大部分时间关闭**（Match Color 中 73% 时刻关闭，Iterative Pushing 中 58% 关闭），仅在关键时刻（如需要回忆初始颜色、回顾上次推力结果时）才打开

---

### 3.4 扩散噪声增强（Diffusion Noise Augmentation）

#### 3.4.1 问题

直接使用干净的历史动作做训练会导致过拟合——模型过度依赖"完美的历史"，对噪声或错误的历史敏感度极高。

#### 3.4.2 方法

在扩散步 k 时，策略预测噪声水平的未来动作 A^k_{t:t+h}，同时以**低一级噪声水平** k-1 的历史动作 A^{k-1}_{t-nh:t} 作为条件：

```
去噪初期（k = K, K-1, ...）：历史动作加强噪声，强制模型不过度依赖干净历史
去噪末期（k = ..., 2, 1）：历史动作仅加微弱噪声，保留细粒度信息
```

这一设计创造了一个**噪声调度**：随去噪进度推进，历史信号逐渐清晰。

**与 Diffusion Forcing 的区别**：Diffusion Forcing 在训练时加随机噪声，但推理时不加噪声，导致训练-推理不一致。GMP 在训练和推理**都使用扩散噪声**，保证了两者的一致性。

---

## 实验结果

### 4.1 MemMimic 基准

GMP 提出了 MemMimic，一个专门评估视觉运动策略记忆能力的基准，包含：

| 任务 | 类型 | 描述 |
|------|------|------|
| T1: Match Color | In-Trial | 记住初始箱子颜色，放置立方体 |
| T2: Discrete Place Back | In-Trial | 记住原位置，悬空 2 秒后放回 |
| T3: Continuous Place Back | In-Trial (Real) | 真实机器人，杯子放回原位 |
| T4: Iterative Pushing | Cross-Trial | 从推距推断摩擦力，动态调整推力 |
| T5: Iterative Flinging | Cross-Trial | 从甩布结果推断质量，调整甩力 |
| T6: Iterative Casting | Cross-Trial (Real) | 从滑行距离推断摩擦系数 |

**关键结果**：GMP 相对长历史基线提升 **30.1%** 平均成功率。

### 4.2 Markov 任务（RoboMimic）

在没有记忆需求的 Markov 任务上，GMP 保持与无历史 Diffusion Policy 相当的性能，而其他长历史方法出现明显下降。

### 4.3 关键发现（Findings）

**Finding 2：Cross-Attention 自适应选择正确模态**

- Match Color 任务中，t=80 时刻放置立方体时，注意力权重最高点落在 t=48（首次观察箱子颜色的时刻）
- Iterative Pushing 中，第 4 次推的时刻，注意力集中在第 2、3 次推的结果（过推/欠推）

**Finding 5：计算效率**

- 自注意力的推理时间随历史长度二次增长
- GMP 的 Cross-Attention 对历史长度是线性增长
- 当 Gate 关闭时，完全跳过历史注意力，推理时间与无历史策略相同

---

## KnowHow + 总结评价

### 核心创新点

1. **Memory Gate 校准**：用"动作预测误差对比"这一自监督信号替代端到端训练中的正则项，避免了手工调参，优雅地解决了"何时用记忆"的问题。
2. **Cross-Attention 替代自注意力**：将历史建模从 O(H²) 降为 O(h×H)，配合 KV Cache 实现高效历史检索。
3. **扩散噪声一致性**：训练和推理都加噪声，避免了 Diffusion Forcing 的 train-test mismatch。
4. **无需人工设计**：直接从 raw image + action 学习记忆召回策略，可泛化到新任务。

### 局限性

- 记忆窗口仍是有限的（滑动窗口策略），无法真正做到"无限记忆"。论文提到未来可探索基于重要性的 token 替换策略。
- Memory Gate 的校准依赖动作预测误差作为代理信号，对内在模糊的任务可能不可靠。

### 个人点评

GMP 的核心贡献不是某个新架构，而是**对"记忆"这件事的精准问题分解**：When/What/How Robust 三个子问题分别用 Gate/Cross-Attention/Diffusion Noise 解决，干净利落。尤其是 Gate 校准流程，用两个固定策略的误差对比来生成监督信号，避免了端到端训练的梯度冲突——这个思路非常值得借鉴到其他需要"选择性使用信息"的场景。

---

## Appendix：论文 Appendix 补充要点

1. **T1' Match Color with Random Delay**：5-600 秒随机延迟测试记忆长度，GMP 在 6000 帧记忆 buffer 下达到 99.0%±1.0% 成功率，推理仅需 0.16 秒（8 步去噪，5090 GPU）。

2. **消融实验（Finding 4）**：验证了 calibration 策略优于 STE（Straight-Through Estimator）和正则项两种替代方案——正则项方法对权重敏感，STE 倾向于让门值趋向 1（过度使用记忆）。

3. **推理时间对比（Finding 5）**：在 RTX 3080 上，120 步历史自注意力需要 ~0.5s/步，GMP (Gate Off) 仅需 ~0.05s/步，与无历史策略相当。

4. **噪声注入消融（Finding 6）**：No Noise / Random Level / Diffusion Forcing 三种方案对比，Diffusion Noising（GMP 方案）在 Iterative Pushing 上显著优于所有基线，验证了训练-推理噪声一致性的重要性。

---

## 参考链接

- **论文**：https://arxiv.org/abs/2604.18933
- **arXiv HTML**：https://arxiv.org/html/2604.18933v1
- **项目主页**：https://gated-memory-policy.github.io/
