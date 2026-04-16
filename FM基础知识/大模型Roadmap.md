# 大模型基础知识 Roadmap

> 来源：小红书「其实大模型也就就这点东西，轻松吃透 LLM基础知识」
> 链接：https://www.xiaohongshu.com/discovery/item/69df95f00000000022025aaf
> 整理时间：2026-04-16

---

## 📐 一、Transformer 结构

### 核心组件

| 组件 | 作用 | 关键参数 |
|------|------|----------|
| **Multi-Head Self-Attention** | 捕捉序列内长距离依赖 | $h$（头数）、$d_{model}$（维度） |
| **Feed-Forward Network (FFN)** | 逐位置非线性变换 | $d_{ff}$（隐藏层维度） |
| **Positional Encoding** | 注入序列位置信息 | 绝对位置编码 / 旋转位置编码（RoPE） |
| **Layer Norm / RMS Norm** | 稳定训练 | $\gamma, \beta$（可学习） |

### Attention 计算公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 关键变体

- **Flash Attention**：IO-aware 精确注意力，减少 HBM 访问（$O(N^2)$ → $O(N)$ 显存）
- **Grouped Query Attention (GQA)**：多头注意力与KV缓存的高效折中
- **Sparse Attention**：BigBird、Longformer 等长上下文建模

### 架构演进

```
Original Transformer (2017) 
  → BERT (MLM预训练, Encoder-only)
  → GPT-2/3 (Decoder-only, In-context Learning)
  → T5 (Encoder-Decoder, Seq2Seq)
  → LLaMA / GPT-4 (Scale + RLHF)
```

---

## 🏛️ 二、主流大模型

### 时间线与代表模型

| 年份 | 模型 | 机构 | 关键创新 |
|------|------|------|----------|
| 2020 | GPT-3 | OpenAI | In-context Learning, 175B 参数 |
| 2022 | ChatGPT | OpenAI | RLHF 对话能力 |
| 2023 | GPT-4 | OpenAI | 多模态 + 长上下文 |
| 2023 | LLaMA | Meta | 开源, 羊驼系模型鼻祖 |
| 2023 | LLaMA 2/3 | Meta | 商业友好, 羊驼生态爆发 |
| 2023 | Claude 2/3 | Anthropic | Constitutional AI, 长上下文 |
| 2023 | Mistral | Mistral AI | MoE, 7B 超越 13B |
| 2024 | GPT-4o / o1 / o3 | OpenAI | 原生多模态, 推理能力 |
| 2024 | Claude 3.5 Sonnet | Anthropic | 编码能力 SOTA |
| 2024 | Gemini 1.5/2.0 | Google | 100M token 上下文 |
| 2024 | DeepSeek-V2 | 深度求索 | MoE, 成本屠夫 |
| 2025 | Qwen 2.5 / 3 | 阿里 | 开源旗舰 |
| 2025 | Grok-3 | xAI | 最大规模推理模型 |

### 按能力分类

```
文字对话 → ChatGPT / Claude / Gemini
编程能力 → GPT-4o / Claude 3.5 / DeepSeek-Coder
数学推理 → o1 / o3 / DeepSeek-Math
开源生态 → LLaMA / Qwen / Mistral / DeepSeek
长上下文 → Gemini 1.5 (10M) / Claude 3 (200K)
多模态  → GPT-4o / Gemini / Claude 3
```

---

## 🔄 三、预训练 Pre-train 过程

### 数据处理流水线

```
Raw Text → 清洗去重 → 安全过滤 → 分词(Tokenizer)
         → 格式化(ChatML/对话格式) → 训练数据
```

### 分词器 (Tokenizer)

| 分词器 | 特点 | 词表大小 |
|--------|------|----------|
| BPE (GPT-2) | 字节级, 压缩友好 | ~50K |
| WordPiece (BERT) | 词根词缀分离 | ~30K |
| SentencePiece |  언어无关, 无空格假设 | 可调 |
| Tiktoken (GPT-4) | 快速 BPE 实现 | ~100K |

### 预训练目标

#### 1. **CLS (Centered Language Modeling)** — GPT 系
只预测下一个 token，损失函数：
$$
\mathcal{L}_{\text{CLM}} = -\sum_t \log P(x_t \mid x_{<t}; \theta)
$$

#### 2. **MLM (Masked Language Modeling)** — BERT 系
随机遮盖 15% token 并预测：
$$
\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{m \sim \mathcal{M}} \sum_{i \in m} \log P(x_i \mid x_{\backslash m}; \theta)
$$

#### 3. **UL2 (Uni-Loan Language)** 
混合降噪器目标（Span corruption + Next token prediction）

###  Scaling Laws

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{S_c}{S}\right)^{\alpha_S}
$$

其中 $N$ 是参数量，$D$ 是训练 token 数，$S$ 是计算量。代表工作：Chinchilla (DeepMind, 2022) 指出 $N$ 与 $D$ 应同比例 scaling。

### 分布式训练策略

| 策略 | 显存占用 | 通信量 | 备注 |
|------|----------|--------|------|
| **Data Parallel (DP)** | $O(N)$ | AllReduce | 最常用 |
| **Tensor Parallel (TP)** | $O(1/N)$ | AllReduce (行/列切分) | Megatron-LM |
| **Pipeline Parallel (PP)** | $O(1/N)$ | P2P 通信 | 微批次填充 |
| **ZeRO-1/2/3** | $O(1/N)$ | AllReduce | DeepSpeed |
| **FSDP** | $O(1/N)$ | AllGather | 显存优化版 DP |

---

## 🎯 四、后训练 Post-train 过程

### 4.1 SFT (Supervised Fine-Tuning)

在指令数据上做全参数或 LoRA 微调：

```python
# 伪代码：SFT 训练循环
for batch in dataloader:
    inputs = tokenize(instruction + response)
    logits = model(inputs.input_ids)
    loss = cross_entropy(logits[:, :-1], inputs.labels[:, 1:])
    loss.backward()
    optimizer.step()
```

**高质量指令数据是关键**：数据质量 >> 数据量

### 4.2 RLHF (Reinforcement Learning from Human Feedback)

Pipeline (InstructGPT / ChatGPT)：

```
SFT Model → Reward Model → RL Fine-tuning (PPO)
```

#### Reward Modeling
$$
\mathcal{L}_R = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\left( r(x, y_w) - r(x, y_l) \right)
$$

#### PPO 目标
$$
\mathcal{L}^{\text{PPO}} = -\mathbb{E}_{(x,y) \sim \pi_{\theta}^{\text{RL}}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\text{SFT}}(y|x)} A(x,y) \right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta \parallel \pi_{\text{SFT}}\right)
$$

### 4.3 DPO (Direct Preference Optimization)

绕过 Reward Model，直接用偏好数据优化：
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(l|x)} \right)
$$

### 4.4 RLAIF & Constitutional AI

- **RLAIF**：用 AI 反馈替代人工反馈
- **Constitutional AI (Anthropic)**：基于一组原则的自我批评与修订

### 4.5 LoRA & Adapter 系列

| 方法 | 原理 | 显存节省 |
|------|------|----------|
| **LoRA** | 低秩分解 $W = W_0 + BA$, $A,B$ 可训练 | ~70-80% |
| **QLoRA** | 4-bit NF4 量化 + LoRA | ~65% |
| **AdaLoRA** | 自适应 rank 分配 | 动态调整 |
| **LoRA+ / DoRA** | 改进的学习率与权重分解 | 收敛更快 |

---

## ⚖️ 五、模型压缩与量化

### 5.1 量化 (Quantization)

将 FP32/FP16 权重映射到低位宽整数：

| 精度 | 格式 | 显存/权重 | 适用场景 |
|------|------|-----------|----------|
| FP32 | 32-bit 浮点 | 4 bytes | 训练基准 |
| FP16 | 16-bit 浮点 | 2 bytes | 推理标配 |
| BF16 | Brain Float 16 | 2 bytes | 训练更稳定 |
| INT8 | 8-bit 整数 | 1 byte | 推理加速 |
| INT4 | 4-bit 整数 | 0.5 bytes | 端侧部署 |
| **NF4** | 4-bit Normal Float | ~0.5 bytes | QLoRA 最优 |

**GPTQ**: Post-training 量化，逐层校准，1-2 GPU 可量化 70B 模型
**AWQ**: Activation-aware 权重量化，对 LLMs 更友好
**GGUF/GGML**: llama.cpp 格式，CPU+GPU 混合推理

### 5.2 剪枝 (Pruning)

| 类型 | 原理 | 效果 |
|------|------|------|
| **Magnitude Pruning** | 删除最小绝对值权重 | 简单有效 |
| **SparseGPT** | 一次性稀疏化 GPT 模型 | 50% sparsity |
| **Wanda** | 基于权重×激活的剪枝 | 无需重训练 |

### 5.3 知识蒸馏 (Distillation)

- **LLaMA.cpp / Qwen.cpp**: 整数化+反量化
- **MiniGPT-4 / Vicuna**: 蒸馏大模型对话能力
- **MoE 专家合并**: 专家权重平均

---

## 🧩 六、专家模型 MoE (Mixture of Experts)

### 核心思想

每个 token 只激活部分专家网络：
$$
y = \sum_{i=1}^{N} G_i \cdot E_i(x), \quad G = \text{TopK}(\text{Softmax}(W_g x))
$$

### 代表架构

| 模型 | 参数量 | 激活参数 | 专家数 | TopK |
|------|--------|----------|--------|------|
| **Mixtral 8x7B** | 46.7B | 12.9B | 8 | 2 |
| **DeepSeek-V2** | 236B | 21B | 128 | 8 |
| **Grok-1** | 314B | - | 8 | 2 |
| **Qwen2-MoE** | 57B | 14.3B | 64 | 8 |
| **LLaMA 4** | - | - | - | - |

### 关键技术

- **细粒度专家**：将 FFN 拆分更多小专家
- **共享专家**：部分专家始终激活（DeepSeek-V2）
- **Expert Choice Routing**：token 选择专家，避免负载不均
- **MoE 负载均衡 loss**：防止路由崩溃

---

## 🔍 七、RAG & Agent

### 7.1 RAG (Retrieval-Augmented Generation)

```
Query → 向量化(Embedding) → 检索(Retriever) → 生成(LLM) → Response
```

**关键组件**：

| 组件 | 技术选型 |
|------|----------|
| **Embedding** | BGE, M3E, OpenAI text-embedding-3 |
| **向量数据库** | Milvus, Qdrant, Weaviate, FAISS |
| **重排序** | BGE-reranker, Cohere |
| **迭代检索** | HyDE, Self-RAG, FLARE |

### 7.2 Agent

核心范式：**感知(Perception) → 规划(Planning) → 行动(Action)**

| 架构 | 代表工作 | 核心机制 |
|------|----------|----------|
| **ReAct** | 推理+行动交替 | $\text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation}$ |
| **AutoGPT** | 自主任务分解 | 自我批评+迭代规划 |
| **Toolformer** | 工具调用学习 | API 调用作为特殊 token |
| **Reflexion** | 经验反思 | 语义化自我反馈 |
| **Memorizing Agents** | LongMem | 历史经验记忆增强 |

### 7.3 多模态 Agent

- **Visual Agent**: GPT-4V, Gemini, LLaVA
- **Video Agent**: FireRTC, 视频理解+推理
- **Robotic Agent**: RT-2, OpenVLA, Physical Intelligence

---

## 🚀 八、部署 & 分布式训练 & 推理加速

### 8.1 推理框架

| 框架 | 加速技术 | 适用场景 |
|------|----------|----------|
| **vLLM** | PagedAttention, Continuous Batching | 生产级 LLM 推理 |
| **llama.cpp** | 4-bit 量化, CPU/GPU 混合 | 端侧/本地部署 |
| **TensorRT-LLM** | FP8, INT8, Attention优化 | NVIDIA GPU 最优 |
| **SGLang** | RadixAttention, Continuous Batching | 高吞吐长上下文 |
| **Ollama** | 简化部署 | 本地一键运行 |
| **DeepSeek-V2** | MLA + MTP | 自家推理优化 |

### 8.2 KV Cache 优化

- **PagedAttention (vLLM)**: 类 OS 分页管理 KV cache，显存利用率提升 2-4x
- **StreamingLLM**: 保留 Sink token + 局部 Window，保持长生成
- **Flash Decoding**: 长的 Key/Value 分块并行

### 8.3 并行推理

```
Request 1: [T1 T2 T3 ... TN]
Request 2:        [T1 T2 T3 ...]  ← Continuous Batching
Request 3:                [T1 T2 T3 ...]
```

- **Continuous Batching**：动态 batch，最大 GPU 利用率
- **Prefix Caching**：共享 system prompt 的 KV cache

### 8.4 分布式训练框架

| 框架 | 并行策略 | 通信优化 |
|------|----------|----------|
| **Megatron-LM** | TP/PP/SP | 异步通信 |
| **DeepSpeed ZeRO** | ZeRO-1/2/3 | 梯度累积+分片 |
| **Colossal-AI** | 统一并行策略 | 异构训练 |
| **FSDP (Fairscale)** | 完全分片数据并行 | AllGather |

### 8.5 推理加速方法

| 方法 | 原理 | 加速比 |
|------|------|--------|
| **Speculative Decoding** | 小模型草稿+大模型验证 | ~2-3x |
| **Medusa** | 多 token 同时预测 | ~2x |
| **Eagle** | 自回归解码树的早期退出 | ~3x |
| **Flash Attention 3** | 非对称内存访问 | 高吞吐量 |

---

## 📊 九、模型评估

### 9.1 评估维度

```
能力评估
├── 📝 文本理解 / 摘要 / 问答
├── 💻 编程能力 (HumanEval / MBPP / LiveCodeBench)
├── 🧮 数学推理 (GSM8K / MATH / MMLU)
├── 🌐 知识问答 (TriviaQA / Natural Questions)
├── 🎭 对话质量 (MT-Bench / ChatArena)
├── 🔄 推理能力 (BigBench-Hard / ARC-C)
└── 🖼️ 多模态 (MME / MMBench / SEED-Bench)

安全评估
├── 🛡️ 对抗攻击防御
├── 🚫 有害内容过滤
├── 📌 幻觉检测 (TruthfulQA / HaluEval)
└── 🔒 隐私合规

效率评估
├── ⚡ 推理延迟 (Token/s)
├── 💾 显存占用
└── 📉 吞吐量和成本
```

### 9.2 主流 Benchmark

| Benchmark | 侧重点 | 特点 |
|-----------|--------|------|
| **MMLU** | 多学科知识 | 57 学科, 英文 |
| **CEval / CMMLU** | 中文能力 | 中文知识 |
| **HumanEval** | 代码补全 | Pass@1 评估 |
| **GSM8K** | 小学数学 | 8K 题 |
| **MATH** | 竞赛数学 | 12K 题, LaTeX |
| **BigBench-Hard** | 复杂推理 | 拒绝比例高 |
| **AlpacaEval** | 指令遵循 | 胜率对比 |
| **MT-Bench** | 多轮对话 | 8 类问题 |
| **ChatArena** | 众包ELO排名 | 匿名对战 |

### 9.3 大模型榜单

| 榜单 | 链接 | 特点 |
|------|------|------|
| **LMSYS Chatbot Arena** | lmarena.ai | 人类盲测 ELO |
| **Open LLM Leaderboard** | huggingface.co | 开源模型排名 |
| **SuperCLUE** | superclueai.com | 中文能力 |
| **FlagEval** | flageval.org | 国产模型评测 |

---

## 🔬 十、其他重要结构

### 10.1 State Space Model (SSM)

| 模型 | 核心机制 | 优势 |
|------|----------|------|
| **Mamba** | 选择性 SSM + 硬件感知 | 线性时间序列建模 |
| **Mamba-2** | State Space Dual (SSD) | Transformer 并行度 |
| **Jamba** | Mamba + Transformer 混合 | 兼顾效率和长程依赖 |

核心公式（连续形式）：
$$
h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Du(t)
$$

离散化后：
$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad \bar{A} = e^{AT}, \quad \bar{B} = (e^{AT}-I)A^{-1}B
$$

### 10.2 Mamba 和 Transformer 对比

| 维度 | Transformer | Mamba (SSM) |
|------|-------------|--------------|
| 复杂度 | $O(N^2)$ 注意力 | $O(N)$ 线性 |
| 长序列 | 受限 (需 Flash Attention) | 更擅长 |
| 并行训练 | 强 (矩阵运算) | 中等 |
| 推理效率 | 较低 (Full Attention) | 高 (线性) |
| 任务泛化 | 通用性强 | 序列建模强 |

### 10.3 混合架构

| 模型 | 架构组合 | 说明 |
|------|----------|------|
| **Transformer-XL** | Transformer + 片段递归 | 解决超长依赖 |
| **GLM** | Prefix-LM + MLM | 中文理解增强 |
| **RWKV** | RNN + Transformer | 兼顾效率和可扩展性 |
| **RetNet** | Retention 机制 | Transformer + RNN |
| **Griffin** | Linear Recurrence + Local Attention | Mamba 团队新工作 |

### 10.4 位置编码进阶

- **RoPE (Rotary Position Embedding)**: 旋转式绝对位置编码，LLaMA/Qwen/Gemma 采用
- **ALiBi (Attention with Linear Biases)**: 线性偏置，BLOOM 采用
- **YaRN**: RoPE 的长度外推微调
- **NTK-Aware Scaling**: 神经切线核视角的位置编码缩放

---

## 🗺️ 学习路径建议

```
入门阶段（1-2月）
├── ✅ Transformer: Attention / FFN / Positional Encoding
├── ✅ 分词与数据处理
├── ✅ 预训练目标: CLM / MLM
└── ✅ HuggingFace Transformers 快速上手

进阶阶段（2-4月）
├── ✅ RLHF / DPO 原理与代码
├── ✅ 量化原理: GPTQ / AWQ / GGUF
├── ✅ LoRA / QLoRA 微调实战
├── ✅ vLLM / llama.cpp 推理部署
└── ✅ 主流模型架构: LLaMA / Qwen / Mistral

深入阶段（4-6月）
├── ✅ MoE 原理与训练策略
├── ✅ RAG / Agent 架构与工具调用
├── ✅ 长上下文建模
├── ✅ 对齐技术: SFT / RLHF / Constitutional AI
└── ✅ Benchmark 评估与模型优化

前沿方向
├── 🆕 SSM: Mamba / Jamba
├── 🆕 原生多模态: GPT-4o / Gemini
├── 🆕 推理Scaling: o1 / o3 / 强化学习搜索
└── 🆕 World Model / VLA / 具身智能
```

---

## 📚 参考资源

### 官方文档 / 论文
- Attention: "Attention Is All You Need" (Vaswani et al., 2017)
- LLaMA: "LLaMA: Open and Efficient Foundation Language Models" (Meta, 2023)
- RLHF: "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT, OpenAI 2022)
- DPO: "Direct Preference Optimization" (Stanford, 2023)
- LoRA: "LoRA: Low-Rank Adaptation of Large Language Models" (Microsoft, 2021)
- GPTQ: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

### 开源工具
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- vLLM: https://docs.vllm.ai/
- llama.cpp: https://github.com/ggerganov/llama.cpp
- DeepSpeed: https://www.deepspeed.ai/
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM

### 学习社区
- paperswithcode.com — 论文+代码对照
- huggingface.co/models — 模型库
- lmarena.ai — Chatbot Arena 榜单
- arxiv.org — 最新论文

---

*整理 by 优酱 🍃 | 如有补充欢迎交流*
