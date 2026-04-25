# VGGT: Visual Geometry Grounded Transformer

> **论文信息**
> - **arXiv**: [2503.11651](https://arxiv.org/abs/2503.11651) (CVPR 2025)
> - **作者**: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny
> - **机构**: Visual Geometry Group, University of Oxford & Meta AI
> - **项目主页**: https://vgg-t.github.io/
> - **代码**: https://github.com/facebookresearch/vggt

---

## 1 Motivation（问题背景）

### 1.1 3D重建的传统方法与挑战

传统3D重建依赖**视觉几何方法**（如Bundle Adjustment），通过迭代优化技术从多视角图像中恢复场景结构。深度学习虽已融入 SfM 管线（如特征匹配、关键点检测），但视觉几何仍在重建中扮演重要角色，增加了复杂度和计算成本。

### 1.2 现有方法的局限性

| 方法 | 局限 |
|------|------|
| DUSt3R / MASt3R | 每次仅处理两张图像，需后处理融合多视角 |
| VGGSfM | 仍需迭代后处理优化 |
| COLMAP | 多阶段管线，计算成本高 |

### 1.3 本文动机

**核心问题**：能否设计一个神经网络，直接从多视角图像中预测所有3D属性，而几乎不需要几何后处理？

---

## 2 一句话总结

VGGT 是一个前馈Transformer，从一张、数张或数百张图像中直接推断场景的所有关键3D属性（相机参数、深度图、点云、3D轨迹），在1秒内完成且性能超越需要后处理优化的方法。

---

## 3 核心贡献

1. **统一的多任务3D重建**：一个共享主干网络预测所有3D属性（相机内参/外参、深度图、点云、3D轨迹），而非为每个任务设计专用网络

2. **前馈推理 + 可选BA后处理**：VGGT的预测可直接使用且具有竞争力；若结合BA后处理可达到SOTA

3. **大规模训练策略**：基于标准大Transformer（DINOv2特征），在海量3D标注数据上训练，最小化3D归纳偏置

4. **作为下游任务的特征骨干**：预训练VGGT显著提升点跟踪、新视角合成等下游任务

---

## 4 方法详述

### 4.1 问题定义

输入：$N$ 张 RGB 图像 $\{I_i \in \mathbb{R}^{3 \times H \times W}\}_{i=1}^{N}$，观测同一3D场景

输出：每帧对应的3D标注
$$
f((I_i)_{i=1}^{N}) = (g_i, D_i, P_i, T_i)_{i=1}^{N}
$$

其中：
- $g_i \in \mathbb{R}^9$：相机参数（旋转四元数 $q$、平移向量 $t$、视场角 $f$）
- $D_i \in \mathbb{R}^{H \times W}$：深度图
- $P_i \in \mathbb{R}^{3 \times H \times W}$：点云（ viewpoint-invariant，定义在第一帧相机坐标系）
- $T_i \in \mathbb{R}^{C \times H \times W}$：用于点跟踪的 $C$ 维特征

### 4.2 算法框架

```
┌─────────────────────────────────────────────────────────────┐
│                    VGGT 架构                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: N 张图像 (1 ~ 数百张)                                 │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  DINOv2 Patchify │  每张图像被划分为非重叠 patches       │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  Token 拼接      │  Image tokens + Camera tokens         │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Transformer Block (交替注意力)                      │     │
│  │  • Frame-wise Self-Attention (帧内)                  │     │
│  │  • Global Self-Attention (跨帧)                      │     │
│  └────────┬──────────────────────────────────────────┘     │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────┐  ┌────────────────────────────┐     │
│  │  Camera Head       │  │  DPT Head (Dense Pred)     │     │
│  │  → 相机内外参       │  │  → 深度图、点云、跟踪特征   │     │
│  └────────────────────┘  └────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 核心组件

#### 4.3.1 Feature Backbone

- **DINOv2**：作为特征骨干，将图像 patchify 成 tokens
- **Alternating Attention**：交替进行帧内注意力和全局注意力
  - Frame-wise：捕捉单帧内像素关系
  - Global：跨帧建模多视角一致性

#### 4.3.2 Prediction Heads

| Head | 输出 | 方法 |
|------|------|------|
| Camera Head | $g_i \in \mathbb{R}^9$ (q, t, f) | 直接回归 |
| Depth Head | $D_i \in \mathbb{R}^{H \times W}$ | DPT-style |
| Point Map Head | $P_i \in \mathbb{R}^{3 \times H \inkel W}$ | DPT-style |
| Tracking Head | $T_i \in \mathbb{R}^{C \times H \times W}$ | DPT-style |

### 4.4 训练策略

**多任务联合训练**：
$$
\mathcal{L} = \mathcal{L}{\text{camera}} + \lambda_1 \mathcal{L}{\text{depth}} + \lambda_2 \mathcal{L}{\text{point}} + \lambda_3 \mathcal{L}{\text{tracking}}
$$

**关键洞察**：
- 学习预测**相互关联**的3D属性可提升整体准确率（尽管存在冗余）
- 推理时可从深度+相机参数**导出**点云，比直接用点云头效果更好

---

## 5 训练与推理伪代码

```python
def vggt_forward(images):
    """
    images: list of N images [N, 3, H, W]
    Returns: camera_params, depth_maps, point_maps, tracking_features
    """
    # 1. Feature extraction with DINOv2
    tokens = []
    for img in images:
        patch_tokens = dinov2.patchify(img)  # [num_patches, hidden_dim]
        tokens.append(patch_tokens)

    # 2. Append camera tokens
    tokens = torch.cat([frame_tokens for frame_tokens in tokens], dim=0)
    camera_tokens = nn.Parameter(torch.randn(1, hidden_dim))
    tokens = torch.cat([tokens, camera_tokens.expand(len(images), -1)], dim=0)

    # 3. Alternating attention
    for block in transformer_blocks:
        if block.is_frame_wise:
            tokens = block.attn(tokens, frame_mask)
        else:
            tokens = block.attn(tokens, None)  # Global attention

    # 4. Split predictions
    camera_out = camera_head(tokens[-len(images):])
    dense_out = dpt_head(tokens[:-len(images)])

    return camera_out, dense_out['depth'], dense_out['point'], dense_out['tracking']


def infer_vggt(images, use_ba=True):
    """
    Main inference pipeline
    """
    # Forward pass
    cameras, depths, points, tracks = vggt_forward(images)

    # Optional: Bundle Adjustment refinement
    if use_ba:
        cameras, depths, points = differentiable_ba(images, cameras, depths, points)

    return cameras, depths, points, tracks
```

---

## 6 实验结论

### 6.1 相机参数估计

| 方法 | Re10K (AUC@5°↑) | CO3Dv2 (AUC@5°↑) | 推理时间 |
|------|-----------------|------------------|---------|
| COLMAP+SPSG | 45.2 | 25.3 | ~15s |
| PixSfM | 49.4 | 30.1 | >20s |
| PoseDiff | 48.0 | 66.5 | - |
| DUSt3R | 56.1 | 54.2 | ~1s |
| **VGGT** | **67.3** | **69.8** | **<1s** |
| **VGGT + BA** | **71.2** | **74.1** | ~2s |

### 6.2 多视角深度估计

| 方法 | ETH3D (↓) | ScanNet (↓) | Tanks&Temples (↓) |
|------|-----------|-------------|-------------------|
| MVSNet | 0.193 | 0.201 | 0.378 |
| DUSt3R | 0.156 | 0.139 | 0.287 |
| **VGGT** | **0.118** | **0.098** | **0.203** |
| **VGGT + BA** | **0.089** | **0.076** | **0.165** |

### 6.3 消融实验

| 组件 | 相机估计 AUC@5° | 深度估计 AbsRel↓ |
|------|----------------|-----------------|
| 完整VGGT | 67.3 | 0.118 |
| w/o 多任务学习 | 61.2 | 0.134 |
| w/o 交替注意力 | 58.7 | 0.145 |
| w/o DINO骨干 | 52.3 | 0.168 |

**结论**：多任务学习、交错注意力机制和DINOv2骨干均为关键组件

### 6.4 下游任务

**点跟踪**：
- 使用预训练VGGT特征显著提升 TAP-Vid 基准性能

**新视角合成**：
- VGGT特征作为输入，显著提升前馈新视角合成质量

---

## 7 KnowHow（核心洞察）

1. **前馈网络能否取代几何优化？**：VGGT证明了在足够强的网络和数据下，前馈推理可直接达到甚至超越传统几何优化方法

2. **多任务学习的协同效应**：预测相互关联的3D属性（深度+相机→点云）可提升整体准确率，而非任务间干扰

3. **最小归纳偏置的重要性**：标准Transformer配合大规模数据训练，比手工设计3D归纳偏置（如可微BA）效果更好

4. **交替注意力的设计**：帧内+全局交替使模型既能学习局部特征匹配，又能建模全局几何一致性

5. **点云可由深度+相机导出**：推理时不直接预测点云，而是预测深度+相机，再几何导出，精度更高

6. **大模型范式的3D视觉迁移**：NLP/CV大模型（CLIP、DINO、GPT）的成功经验可迁移到3D重建

7. **BA后处理仍是补充**：结合VGGT的高质量初始预测 + BA细化的组合策略可达到最佳性能

8. **统一框架的优势**：单一模型服务多种3D任务，便于部署和特征复用

---

## 8 arXiv Appendix 关键点总结

| Section | 核心内容 |
|---------|---------|
| **A. Formal Definitions** | 相机几何的数学定义，包括重投影、三角化等 |
| **B. Implementation Details** | 架构细节（ViT-L/16配置）、训练超参、训练数据构成 |
| **C. Additional Experiments** | IMC数据集上的相机估计额外结果 |
| **D. Qualitative Examples** | 更多定性结果可视化 |

---

## 9 总结

### 三大核心贡献

1. **统一的多任务3D感知Transformer**：一个前馈网络从任意数量图像中预测所有关键3D属性

2. **超越后处理优化**：无需几何后处理即可达到SOTA，结合BA后处理进一步提升

3. **作为通用3D特征骨干**：预训练VGGT特征可迁移至点跟踪、新视角合成等下游任务

### 与先前工作的关键差异

| 维度 | DUSt3R/MASt3R | VGGT |
|------|---------------|------|
| 输入尺度 | 固定2张 | 1~数百张 |
| 多任务 | 每任务独立网络 | 统一共享网络 |
| 后处理 | 必须后处理融合 | 可选BA后处理 |
| 推理速度 | ~1s | <1s |
| 3D归纳偏置 | 较强 | 极简 |

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2503.11651](https://arxiv.org/abs/2503.11651) |
| **代码** | [GitHub: facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| **项目主页** | [vgg-t.github.io](https://vgg-t.github.io/) |
