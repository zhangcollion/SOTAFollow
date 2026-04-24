# DeepSeek-V4 论文精读

> **论文信息**
> - **arXiv**: [2503.24978](https://arxiv.org/abs/2503.24978) (2025)
> - **机构**: DeepSeek AI
> - **代码**: https://github.com/deepseek-ai/DeepSeek-V4
> - **模型**: https://huggingface.co/collections/deepseek-ai/deepseek-v4

---

## 1 Motivation（问题背景）

### 1.1 当前 LLM 的核心瓶颈

| 瓶颈 | 具体表现 |
|------|---------|
| **注意力计算复杂度** | 标准注意力机制呈二次方增长，极长上下文场景计算成本极高 |
| **测试时 Scaling 受限** | 推理模型的计算资源需求限制了 Test-time Scaling 的发展 |
| **KV Cache 爆炸** | 极长序列的 KV Cache 存储成为内存瓶颈 |
| **长程场景需求** | 复杂 Agent 工作流、跨文档分析等场景对超长上下文有强烈需求 |

### 1.2 现有开源模型的局限

- **DeepSeek-V3.2** 等模型虽在通用能力上取得进展，但在处理超长序列时存在核心架构效率问题
- 现有稀疏注意力方案（如 StreamingLLM）在压缩率与信息保留之间难以平衡

### 1.3 本文动机

**核心问题**：如何打破超长上下文场景下的效率壁垒，实现真正实用的百万 token 上下文支持？

---

## 2 一句话总结

DeepSeek-V4 通过 CSA+HCA 混合稀疏注意力架构和 mHC 残差连接优化，在仅 27% 单 token 推理 FLOPs 和 10% KV Cache 的条件下实现百万 token 上下文高效处理，重新定义开源模型 SOTA。

---

## 3 核心贡献

1. **CSA+HCA 混合稀疏注意力**：通过压缩稀疏注意力（CSA）和重度压缩注意力（HCA）的混合架构，实现超长上下文的高效计算

2. **mHC（流形约束超连接）**：将残差映射矩阵约束到双随机矩阵流形，增强深层堆叠的数值稳定性

3. **Muon 优化器**：采用 Newton-Schulz 正交化迭代，实现更快的收敛速度和更高的训练稳定性

4. **FP4 量化感知训练**：对 MoE 专家权重应用 MXFP4 量化，显著降低推理内存和计算成本

5. **两阶段后训练范式**：专家独立训练 + 多教师 OPD 蒸馏，高效合并多领域能力

---

## 4 方法详述

### 4.1 问题定义

**输入**：最长 1M token 的文本序列
**输出**：文本补全

**核心目标**：
1. 在超长上下文场景下显著降低计算成本（FLOPs）
2. 显著降低 KV Cache 存储需求
3. 保持模型在各类任务上的 SOTA 性能

### 4.2 算法框架

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DeepSeek-V4 架构                                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Input Tokens (最长 1M)                                                │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │           Embedding Layer                                     │      │
│  └──────────────────────────────────────────────────────────────┘      │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Transformer Block × L                                          │      │
│  │                                                               │      │
│  │   ┌────────────────────────────────────────────────────┐   │      │
│  │   │  CSA / HCA (交替使用)                                │   │      │
│  │   │  Compressed Sparse Attention                       │   │      │
│  │   │  Heavily Compressed Attention                     │   │      │
│  │   └────────────────────────────────────────────────────┘   │      │
│  │                         │                                    │      │
│  │                         ▼                                    │      │
│  │   ┌────────────────────────────────────────────────────┐   │      │
│  │   │  DeepSeekMoE (Feed-Forward)                        │   │      │
│  │   │  • Fine-grained Routed Experts                     │   │      │
│  │   │  • Shared Experts                                 │   │      │
│  │   │  • Hash Routing                                   │   │      │
│  │   └────────────────────────────────────────────────────┘   │      │
│  │                         │                                    │      │
│  │                         ▼                                    │      │
│  │   ┌────────────────────────────────────────────────────┐   │      │
│  │   │  mHC (Manifold-Constrained Hyper-Connections)       │   │      │
│  │   │  • 双随机矩阵约束                                   │   │      │
│  │   │  • 谱范数有界 (≤1)                                │   │      │
│  │   └────────────────────────────────────────────────────┘   │      │
│  └──────────────────────────────────────────────────────────────┘      │
│       │                                                                │
│       ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  MTP Modules (Multi-Token Prediction)                         │      │
│  └──────────────────────────────────────────────────────────────┘      │
│       │                                                                │
│       ▼                                                                │
│  Output                                                                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.3 核心组件详解

#### 4.3.1 CSA (Compressed Sparse Attention)

**核心思想**：
1. 将每 m 个 token 的 KV Cache 压缩为 1 个条目
2. 使用 Lightning Indexer 稀疏选择 top-k 个相关压缩块
3. 在压缩后的 KV 条目上执行 MQA（Multi-Query Attention）

**KV 压缩公式**：
$$
[ S_a^{mi:m(i+1)-1}; S_b^{m(i-1):mi} ] = \text{Softmax}_{row}([Z_a^{mi:m(i+1)-1} + B_a; Z_b^{m(i-1):mi} + B_b] )
$$

$$
C_{Comp_i} = \sum_{j=mi}^{m(i+1)-1} S_{a_j} \odot C_{a_j} + \sum_{j=m(i-1)}^{mi-1} S_{b_j} \odot C_{b_j}
$$

**Lightning Indexer 稀疏选择**：
- 将 query 压缩到低维空间
- 计算与压缩 KV 块之间的索引分数
- 选择 top-k 个最高分块进行注意力计算

#### 4.3.2 HCA (Heavily Compressed Attention)

**与 CSA 的区别**：
- 压缩率更高：m' >> m
- 不使用稀疏选择，直接在所有压缩块上执行注意力
- 保留滑动窗口 KV 条目以增强局部依赖

#### 4.3.3 mHC (Manifold-Constrained Hyper-Connections)

**核心创新**：将残差映射矩阵 $B_l$ 约束到双随机矩阵流形（Birkhoff 多面体）

**数学约束**：
$$
B_l \in \mathcal{M} := \{ M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n, \mathbf{1}_n^T M = \mathbf{1}_n^T, M \succeq 0 \}
$$

**三大优势**：
1. 谱范数有界：$\parallel B_l \parallel_2 \leq 1$，残差变换非扩展
2. 流形封闭：$\mathcal{M}$ 对乘法封闭，保证深层堆叠的数值稳定
3. 信号保真：输入/输出变换通过 Sigmoid 约束为非负有界

#### 4.3.4 MoE 与 Hash 路由

**与 DeepSeek-V3 的关键差异**：

| 特性 | DeepSeek-V3 | DeepSeek-V4 |
|------|-------------|-------------|
| **激活函数** | Sigmoid | **Sqrt(Softplus)** |
| **路由** | Top-K 路由 | **Hash 路由** |
| **负载均衡** | 辅助损失 | **无辅助损失 + 序列级平衡** |

**SqrtSoftplus**：
$$
\text{SqrtSoftplus}(x) = \sqrt{\log(1 + e^x)}
$$

#### 4.3.5 Muon 优化器

**核心原理**：
1. 使用动量累积梯度
2. Nesterov 动量加速
3. Hybrid Newton-Schulz 正交化将更新矩阵正交化

**算法流程**：
```python
def muon_step(W, G, M, lr, momentum, weight_decay, gamma):
    M = momentum * M + G                    # 动量更新
    G_nesterov = momentum * M + G            # Nesterov 动量
    O = hybrid_newton_schulz(G_nesterov)    # 正交化
    O = O * sqrt(max(n, m)) * gamma         # 缩放更新
    W = W * (1 - lr * weight_decay) - lr * O # 权重更新
    return W, M
```

**Hybrid Newton-Schulz 迭代**：
- 前 8 步：系数 $(a,b,c) = (3.4445, -4.7750, 2.0315)$ 快速收敛
- 后 2 步：系数 $(a,b,c) = (2, -1.5, 0.5)$ 稳定奇异值

---

---

## 5 Pre-Training（预训练）

### 5.1 数据构建

DeepSeek-V4 系列在超过 **32T** 多样化高质量 tokens 上进行预训练：

| 模型 | 训练 Tokens |
|------|------------|
| DeepSeek-V4-Flash | 32T |
| DeepSeek-V4-Pro | 33T |

### 5.2 模型配置

| 配置项 | DeepSeek-V4-Pro | DeepSeek-V4-Flash |
|--------|-----------------|-------------------|
| **总参数** | 1.6T | 284B |
| **激活参数** | 49B | 13B |
| **上下文长度** | 1M | 1M |
| **注意力头数** | 128 | 128 |
| **KV 头数** | 16 (MQA) | 16 (MQA) |
| **MTP 模块数** | - | - |

### 5.3 训练稳定性

为缓解训练不稳定性，采用以下策略：
- **混合精度训练**：FP8 前向/反向，FP32 优化器状态
- **渐进式学习率调度**：warmup → constant → cosine decay
- **梯度裁剪**：全局梯度范数裁剪

---

## 6 Post-Training Pipeline（后训练流程）

### 6.1 Specialist Training（专家独立培养）

DeepSeek-V4 采用**两阶段后训练范式**：专家独立培养 + 多教师 OPD 蒸馏。

**目标领域**：数学、代码、Agent、指令遵循

**训练流程**：
1. **SFT（监督微调）**：在高质量领域特定数据上建立基础能力
2. **GRPO（群相对策略优化）**：使用领域特定提示和奖励信号进一步优化

### 6.2 三种推理模式

| 推理模式 | 特性 | 典型用例 | 响应格式 |
|---------|------|---------|---------|
| **Non-think** | 快速、直觉响应 | 日常任务、紧急反应、低风险决策 | 总结 |
| **Think High** | 深度逻辑分析 | 复杂问题解决、规划、中等风险决策 | <think> thinking </think> 总结 |
| **Think Max** | 极致推理深度 | 探索模型推理边界 | 特殊系统提示 + 深度思考链 |

**Think Max 模式注入指令**：
```
Reasoning Effort: Absolute maximum with no shortcuts permitted.
You MUST be very thorough in your thinking and comprehensively decompose the
problem to resolve the root cause, rigorously stress-testing your logic against
all potential paths, edge cases, and adversarial scenarios.
```

### 6.3 Generative Reward Model (GRM)

传统 RLHF 需要人工标注训练 Reward Model。DeepSeek-V4 采用 **Generative Reward Model (GRM)**：
- 基于 rubric 构建 RL 数据
- 使用 GRM 评估策略轨迹
- **关键创新**：GRM 本身也通过 RL 优化

### 6.4 工具调用架构

**XML 格式工具调用**：
```xml
<|DSML|tool_calls>
<|DSML|invoke name="$TOOL_NAME">
<|DSML|parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE
</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>
```

**关键优势**：XML 格式有效减少转义失败和工具调用错误

### 6.5 交错式思考（Interleaved Thinking）

| 场景 | DeepSeek-V3.2 | DeepSeek-V4 |
|------|---------------|-------------|
| **工具调用场景** | 每轮用户消息丢弃思考链 | 完整保留跨轮思考历史 |
| **通用对话场景** | 新消息到达时丢弃旧思考 | 保持简洁上下文 |

### 6.6 Quick Instruction

解决辅助任务（小模型判断是否搜索、意图识别等）需要冗余 prefilling 的问题：

| Special Token | 描述 |
|--------------|------|
| `<|action|>` | 判断是否需要网络搜索 |
| `<|title|>` | 生成对话标题 |
| `<|query|>` | 生成搜索查询 |
| `<|authority|>` | 判断权威性需求 |
| `<|domain|>` | 识别问题领域 |
| `<|read_url|>` | 判断 URL 是否需要读取 |

### 6.7 On-Policy Distillation (OPD) 细节

**OPD 目标函数**：
$$
\mathcal{L}_{OPD}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{KL}(\pi_\theta \parallel \pi_{E_i})
$$

**关键设计**：
1. **超过 10 个教师模型**覆盖不同领域
2. **全词汇表蒸馏**：保留完整 logit 分布
3. **反向 KL 散度**：$D_{KL}(\pi_\theta \parallel \pi_{E_i})$

**高效工程实现**：
- Teacher 权重卸载到分布式存储，ZeRO-like 分片
- 只缓存最后层 hidden states，训练时重建 logits
- 按 teacher 顺序调度数据，最小化 head 加载

---

## 7 RL & OPD 基础设施

### 7.1 FP4 量化集成

- **Rollout 和推理阶段**：直接使用 FP4 权重
- **训练阶段**：FP4 → FP8 去量化，模拟量化感知训练
- 复用现有 FP8 混合精度框架，无需修改反向传播管道

### 7.2 可抢占式 Rollout 服务

支持 KV cache 持久化和 WAL（Write-Ahead Log）机制：
- 任务抢占时保留 KV cache
- 恢复时从持久化 WAL 和 KV cache 继续
- **关键**：从 checkpoint 重新生成会产生长度偏差（短响应更容易在中断中存活）

### 7.3 百万 Token 上下文 RL 优化

- 轻量级元数据 + 重型 per-token 字段分离
- 通过共享内存数据加载器消除节点内数据冗余
- 动态 batch 大小，平衡计算吞吐量和 I/O 重叠

### 7.4 DSec 沙箱基础设施

DeepSeek Elastic Compute (DSec) 是生产级沙箱平台：

| 组件 | 功能 |
|------|------|
| **Apiserver** | API 网关 |
| **Edge** | 每主机代理 |
| **Watcher** | 集群监控 |

**四大核心设计**：
1. **异构工作负载**：支持轻量级函数调用到完整软件工程管道
2. **快速镜像加载**：分层存储 + 3FS 分布式文件系统
3. **高密度部署**：内存复用 + spinlock 优化
4. **轨迹记录与抢占恢复**：支持快速转发和确定性回放

---

## 8 基础设施关键技术

### 8.1 细粒度 EP 通信-计算重叠

**MoE 层四阶段**：Dispatch → Linear-1 → Linear-2 → Combine

**优化策略**：
- 将 experts 分组调度成多个 wave
- 每个 wave 完成后立即开始计算，无需等待所有 experts
- 理论加速比：1.92×（相比 Naive 1.42×）

### 8.2 TileLang 内核开发

**Host Codegen**：将 host 端逻辑移入生成的代码，CPU 验证开销从数十微秒降至 <1 微秒

**SMT 求解器辅助形式化整数分析**：用于布局推理、内存 hazard 检测、边界分析

### 8.3 批不变性核

**挑战**：实现输出与 batch 位置无关

**解决方案**：
- 注意力：开发双核策略处理 wave 量化问题
- MoE 反向：token 顺序预处理 + buffer 隔离

### 8.4 确定性与可复现性

**非确定性来源**：
- 注意力反向：atomicAdd 累积梯度
- MoE 反向：多 SM 并发写入
- mHC 矩阵乘法：split-k 算法

**解决方案**：
- 分离累积 buffer → 全局确定性求和
- token 顺序预处理 + buffer 隔离
- 分块输出 → 确定性归约

---

## 9 训练与推理伪代码

### 9.1 CSA 前向传播

```python
def csa_forward(x, m=8, k=8):
    """
    x: [seq_len, batch, hidden_dim]
    m: 压缩率
    k: top-k 选择数
    """
    seq_len, batch, d = x.shape
    
    # 1. 计算 KV 和压缩权重
    C_a = x @ W_a_KV  # [n, c]
    C_b = x @ W_b_KV
    Z_a = x @ W_a_Z
    Z_b = x @ W_b_Z
    
    # 2. KV 压缩
    C_comp = []
    for i in range(0, n, m):
        segment_C = torch.cat([C_a[i:i+m], C_b[max(0,i-m):i]], dim=0)
        segment_Z = torch.cat([Z_a[i:i+m], Z_b[max(0,i-m):i]], dim=0)
        weight = F.softmax(segment_Z + positional_bias, dim=0)
        compressed = (weight * segment_C).sum(dim=0)
        C_comp.append(compressed)
    C_comp = torch.stack(C_comp)  # [n/m, c]
    
    # 3. Lightning Indexer 稀疏选择
    c_Q = x @ W_DQ  # 压缩 query
    index_scores = (c_Q @ W_w) @ C_comp.transpose(-2, -1)
    index_scores = F.relu(index_scores)
    top_k_idx = torch.topk(index_scores, k=k, dim=-1).indices
    selected_C = C_comp[top_k_idx]  # [n, k, c]
    
    # 4. Shared KV MQA
    query = c_Q @ W_UQ  # [n, n_h, c]
    output = core_mqa(query, selected_C, selected_C)
    
    # 5. Grouped Output Projection
    output = grouped_projection(output)
    
    return output
```

### 9.2 mHC 前向传播

```python
def mhc_forward(x, W_pre, W_res, W_post, S_pre, S_res, S_post, 
                 alpha_pre, alpha_res, alpha_post):
    """
    x: [batch, seq, d]
    """
    batch, seq, d = x.shape
    n_hc = W_pre.shape[0] // d
    
    # 展平并归一化
    x_flat = x.view(-1, d)
    x_norm = F.rms_norm(x_flat, (d,))
    x_hat = x_norm.flatten()
    
    # 动态生成参数
    A_tilde = alpha_pre * (x_hat @ W_pre) + S_pre
    B_tilde = alpha_res * (x_hat @ W_res) + S_res
    C_tilde = alpha_post * (x_hat @ W_post) + S_post
    
    # Sinkhorn 投影到双随机矩阵
    B_constrained = doubly_stochastic_projection(B_tilde.view(n_hc, n_hc))
    
    # Sigmoid 约束
    A = torch.sigmoid(A_tilde.unsqueeze(0))
    C = torch.sigmoid(C_tilde.unsqueeze(0))
    
    # HC 变换
    residual = B_constrained @ (C * pre_output)
    output = A * x + residual
    
    return output.view(batch, seq, -1)


def doubly_stochastic_projection(B, max_iter=100, eps=1e-4):
    """Sinkhorn-Knopp 算法"""
    B = F.relu(B)
    for _ in range(max_iter):
        B = B / (B.sum(dim=-1, keepdim=True) + eps)  # 行归一化
        B = B / (B.sum(dim=-2, keepdim=True) + eps)  # 列归一化
        if torch.allclose(B.sum(dim=-1), torch.ones_like(B.sum(dim=-1)), atol=eps):
            break
    return B
```

### 9.3 Muon 优化器

```python
class MuonOptimizer:
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.01, gamma=0.01):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.state = {p: {'momentum': torch.zeros_like(p)} for p in params}
        
    def hybrid_newton_schulz(self, G, steps=10):
        """混合 Newton-Schulz 正交化"""
        G = G / (torch.norm(G, dim=(-2, -1), keepdim=True) + 1e-8)  # 归一化
        
        # 前 8 步：快速收敛
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(8):
            GGT = G @ G.transpose(-2, -1)
            G = a * G + b * GGT @ G + c * (GGT @ GGT) @ G
            
        # 后 2 步：稳定奇异值
        a, b, c = 2, -1.5, 0.5
        for _ in range(2):
            GGT = G @ G.transpose(-2, -1)
            G = a * G + b * GGT @ G + c * (GGT @ GGT) @ G
        
        return G
    
    def step(self, params, grads):
        for p in params:
            if p not in self.state:
                self.state[p] = {'momentum': torch.zeros_like(p)}
            
            G = grads[p]
            M = self.state[p]['momentum']
            
            # 累积动量
            M = self.momentum * M + G
            
            # Nesterov 动量
            G_nesterov = self.momentum * M + G
            
            # 正交化
            O = self.hybrid_newton_schulz(G_nesterov + self.momentum * M)
            
            # 缩放更新
            n, m = O.shape[-2], O.shape[-1]
            O = O * torch.sqrt(max(n, m)) * self.gamma
            
            # 权重更新
            p.data = p.data * (1 - self.lr * self.weight_decay) - self.lr * O
```

---

## 10 实验结论

### 10.1 效率对比

#### 10.1.1 推理 FLOPs 对比（1M Token 上下文）

| 模型 | 单 Token FLOPs (T) | 相对 DeepSeek-V3.2 |
|------|-------------------|-------------------|
| DeepSeek-V3.2 | 基准 | 100% |
| DeepSeek-V4-Pro | 3.7× lower | 27% |
| DeepSeek-V4-Flash | 9.8× lower | 10% |

#### 10.1.2 KV Cache 对比（1M Token 上下文）

| 模型 | KV Cache (GB) | 相对 DeepSeek-V3.2 |
|------|--------------|-------------------|
| DeepSeek-V3.2 | 基准 | 100% |
| DeepSeek-V4-Pro | 9.5× smaller | 10% |
| DeepSeek-V4-Flash | 13.7× smaller | 7% |

### 10.2 基准测试结果

#### 10.2.1 DeepSeek-V4-Pro-Max vs 闭源/开源模型

| Benchmark | Opus-4.6 | GPT-5.4 | Gemini-3.1-Pro | K2.6 | GLM-5.1 | **DS-V4-Pro-Max** |
|-----------|----------|---------|---------------|------|---------|-------------------|
| **MMLU-Pro (EM)** | 89.1 | 87.5 | 91.0 | 87.1 | 86.0 | **87.5** |
| **SimpleQA-Verified** | 46.2 | 45.3 | 75.6 | 36.9 | 38.1 | **57.9** |
| **Chinese-SimpleQA** | 76.4 | 76.8 | 85.9 | 75.9 | 75.0 | **84.4** |
| **GPQA Diamond** | 91.3 | 93.0 | 94.3 | 90.5 | 86.2 | **90.1** |
| **HLE (Pass@1)** | 40.0 | 39.8 | 44.4 | 36.4 | 34.7 | **37.7** |
| **LiveCodeBench** | 88.8 | - | 91.7 | 89.6 | - | **93.5** |
| **CodeForces (Rating)** | - | 3168 | 3052 | - | - | **3206** |
| **HMMT 2026 Feb** | 96.2 | 97.7 | 94.7 | 92.7 | 89.4 | **95.2** |
| **IMOAnswerBench** | 75.3 | 91.4 | 81.0 | 86.0 | 83.8 | **89.8** |
| **Apex Shortlist** | 85.9 | 78.1 | 89.1 | 75.5 | 72.4 | **90.2** |
| **Terminal Bench 2.0** | 65.4 | 75.1 | 68.5 | 66.7 | 63.5 | **67.9** |
| **SWE Verified** | 80.8 | - | 80.6 | 80.2 | - | **80.6** |

#### 10.2.2 DeepSeek-V4-Flash vs DeepSeek-V4-Pro

| Benchmark | DS-V4-Flash Max | DS-V4-Pro Max |
|-----------|-----------------|----------------|
| **MMLU-Pro** | 86.2 | 87.5 |
| **SimpleQA-Verified** | 34.1 | 57.9 |
| **LiveCodeBench** | 91.6 | 93.5 |
| **CodeForces Rating** | 3052 | 3206 |
| **Terminal Bench 2.0** | 56.9 | 67.9 |

### 10.3 1M Token 上下文性能

在 MRCR 任务上：
- **128K 以内**：检索性能高度稳定
- **128K~512K**：性能开始下降但仍保持较强水平
- **1M token**：DeepSeek-V4-Pro 仍显著超越 Gemini-3.1-Pro

### 10.4 真实场景任务

#### 10.4.1 中文写作

- DeepSeek-V4-Pro vs Gemini-3.1-Pro 功能写作胜率：**62.7%** vs 34.1%
- 创意写作指令遵循胜率：60.0%

#### 10.4.2 代码 Agent

| 模型 | 通过率 |
|------|--------|
| Haiku 4.5 | 13% |
| Sonnet 4.5 | 47% |
| **DeepSeek-V4-Pro-Max** | **67%** |
| Opus 4.5 | 70% |
| Opus 4.6 Thinking | 80% |

---

## 11 KnowHow（核心洞察）

1. **CSA+HCA 混合架构的价值**：通过交替使用不同压缩率的注意力机制，在计算效率和模型性能之间取得平衡

2. **mHC 流形约束的稳定性**：将残差矩阵约束到双随机矩阵流形，确保谱范数有界，是深层堆叠数值稳定的关键

3. **Muon 正交化的收敛优势**：Newton-Schulz 正交化使梯度更新沿谱范数方向，比标准动量更高效

4. **Hash 路由的确定性**：无需辅助损失即可实现负载均衡，通过预定义哈希函数确定 expert 分配

5. **SqrtSoftplus 的平滑性**：相比 Sigmoid，提供更平滑的激活值变化

6. **FP4 量化的实用性**：训练时模拟量化，推理时直接使用真实 FP4 权重，实现无损加速

7. **交错式思考的 Agent 价值**：在工具调用场景保留完整思考链，对复杂多轮任务至关重要

8. **Quick Instruction 的工程优化**：复用 KV cache 避免冗余 prefilling，显著降低 TTFT

9. **全词汇表 OPD 的稳定性**：保留完整 logit 分布，比 token 级 KL 估计梯度方差更低

10. **KV Cache 层级管理的必要性**：异构 KV 条目需要不同的缓存和驱逐策略

11. **细粒度 EP 通信优化**：将 experts 分组调度，实现计算-通信流水线重叠

12. **批不变性核的心理健康价值**：确保输出与 batch 位置无关，对训练可复现性至关重要

---

## 12 核心架构对比

| 维度 | DeepSeek-V4 | DeepSeek-V3.2 | Gemini-3.1-Pro |
|------|-------------|---------------|---------------|
| **总参数** | 1.6T / 284B | - | 闭源 |
| **激活参数** | 49B / 13B | - | 闭源 |
| **上下文长度** | 1M | 128K | 1M |
| **注意力架构** | CSA + HCA 混合 | 滑动窗口 | 标准注意力 |
| **KV Cache (1M)** | 10% / 7% | 100% | - |
| **优化器** | Muon | AdamW | - |
| **量化** | FP4 MoE | - | - |
| **路由** | Hash | Top-K | - |

---

## 13 总结

### 三大核心贡献

1. **CSA+HCA 混合稀疏注意力**：通过分层压缩策略，实现 1M token 上下文下仅 27% FLOPs 和 10% KV Cache

2. **mHC 流形约束超连接**：将残差映射约束到双随机矩阵流形，确保深层堆叠的数值稳定性

3. **Muon + FP4 QAT 优化**：Newton-Schulz 正交化加速收敛，FP4 量化降低推理成本

### 后训练两阶段范式

1. **专家独立培养**：SFT + GRPO 针对数学、代码、Agent 等领域独立训练专家
2. **多教师 OPD 蒸馏**：超过 10 个教师模型通过全词汇表蒸馏合并到统一学生模型

### 局限性与未来方向

1. **架构复杂性**：为追求极致长上下文效率采用了相对复杂的架构设计
2. **新维度稀疏**：探索更稀疏的 embedding 模块
3. **低延迟架构**：优化长上下文部署和交互响应性
4. **在线学习**：为未来在线学习范式奠定基础

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2503.24978](https://arxiv.org/abs/2503.24978) |
| **代码** | [GitHub: deepseek-ai/DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V4) |
| **模型** | [HuggingFace: DeepSeek-V4](https://huggingface.co/collections/deepseek-ai/deepseek-v4) |
