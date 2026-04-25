# DVGT: Driving Visual Geometry Transformer

> **论文信息**
> - **arXiv**: [2512.16919](https://arxiv.org/abs/2512.16919) (2025)
> - **作者**: Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Shengyin Jiang, Long Chen, Zhi-Xin Yang, Jiwen Lu
> - **机构**: Tsinghua University & Xiaomi EV & University of Macau & Peking University
> - **代码**: https://github.com/wzzheng/DVGT
> - **项目主页**: https://wzzheng.net/DVGT

---

## 1 Motivation（问题背景）

### 1.1 自动驾驶视觉几何感知的挑战

现有自动驾驶方法在3D场景几何感知方面存在以下问题：

| 问题 | 具体表现 |
|------|---------|
| **2.5D表示局限** | 单目深度预测或3D占用预测无法提供统一、连续的3D场景表示 |
| **量化误差** | 离散体素网格（典型分辨率0.5m）无法精确表示细节几何 |
| **相机配置耦合** | 依赖精确相机参数的方法难以泛化到不同传感器配置 |
| **数据规模限制** | 单一相机配置数据训练，限制了跨场景适应性和可扩展性 |

### 1.2 现有方法的局限

- **传统BEV/Occupancy方法**：依赖2D到3D几何投影，强耦合相机先验
- **通用视觉几何模型（DUSt3R/MASt3R/VGGT）**：只能恢复相对尺度，需要后处理对齐外部传感器（如LiDAR）

### 1.3 本文动机

**核心问题**：能否设计一个专门针对自动驾驶的统一密集视觉几何模型，能适应不同场景和相机配置，直接预测度量尺度的3D几何？

---

## 2 一句话总结

DVGT（Driving Visual Geometry Transformer）是一个专门为自动驾驶设计的视觉几何Transformer，从无位姿多视角图像序列中端到端预测度量尺度的全局3D点云和自车轨迹，无需后处理对齐外部传感器。

---

## 3 核心贡献

1. **端到端度量尺度3D重建**：直接从图像序列预测度量尺度的全局3D点云，无需LiDAR后处理对齐

2. **自车坐标系的统一几何表示**：将几何表示与相机参数解耦，在自车坐标系中表示3D点云，适配任意相机配置

3. **时空分解注意力机制**：提出帧内局部注意、跨视角空间注意、跨帧时序注意的高效因子化注意力

4. **大规模混合训练**：利用nuScenes、OpenScene、Waymo、KITTI、DDAD等多个驾驶数据集训练

---

## 4 方法详述

### 4.1 问题定义

**输入**：$T$ 帧 × $N$ 视角的图像序列 $\mathcal{I} = \{I_{t,n}\}_{t=1..T,n=1..N}$

**输出**：
- 3D点云 $\mathcal{P} = \{\hat{P}_{t,n}\}_{t=1..T,n=1..N}$（在参考帧自车坐标系中）
- 自车轨迹 $\mathcal{T}_{ego} = \{\hat{T}_t\}_{t=1..T} \in SE(3)$

### 4.2 算法框架

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DVGT 框架                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: T帧 × N视角图像序列                                           │
│              │                                                        │
│              ▼                                                        │
│  ┌─────────────────────────┐                                         │
│  │  DINOv2 特征提取        │  每张图像 → token序列                    │
│  │  (Pretrained Backbone)  │                                         │
│  └────────────┬────────────┘                                         │
│               │                                                       │
│               ▼                                                       │
│  ┌────────────────────────────────────────────────────────────┐       │
│  │           Spatial-Temporal Geometry Transformer            │       │
│  │                                                            │       │
│  │  ① Intra-view Local Attention (帧内局部)                   │       │
│  │  ② Cross-view Spatial Attention (跨视角空间)               │       │
│  │  ③ Cross-frame Temporal Attention (跨帧时序)               │       │
│  │                                                            │       │
│  └────────────┬─────────────────────────────────────────────┘       │
│               │                                                       │
│               ▼                                                       │
│  ┌─────────────────────┐  ┌──────────────────────┐               │
│  │   3D Point Map Head │  │   Ego Pose Head      │               │
│  │   全局3D点云预测      │  │   自车轨迹预测        │               │
│  │   (度量尺度)          │  │   (SE(3))           │               │
│  └─────────────────────┘  └──────────────────────┘               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 核心设计

#### 4.3.1 Ego-centric 3D Point Map

**与VGGT的关键区别**：

| 特性 | VGGT | DVGT |
|------|------|------|
| 参考坐标系 | 第一帧相机坐标系 | 自车坐标系 |
| 几何表示 | 与相机参数耦合 | 与相机参数解耦 |
| 相机配置 | 通用 | 针对自动驾驶多视角 |
| 度量尺度 | 相对尺度（需后处理） | **直接度量尺度** |

**核心优势**：
1. **High-Fidelity**：连续坐标消除量化误差
2. **Completeness**：像素级密度，前景背景全覆盖
3. **通用性**：不受相机焦距、位姿、视角数量限制

#### 4.3.2 Spatial-Temporal Geometry Transformer

**时空分解注意力**：

```python
class GeometryTransformer(nn.Module):
    def __init__(self, num_layers):
        self.layers = nn.ModuleList([GeometryBlock() for _ in range(num_layers)])

    def forward(self, tokens_per_frame):
        # tokens_per_frame: [T, N, num_tokens, hidden_dim]

        for layer in self.layers:
            # ① 帧内局部注意力（捕捉单帧内像素关系）
            tokens_per_frame = layer.intra_view_attn(tokens_per_frame)

            # ② 跨视角空间注意力（多相机环绕融合）
            tokens_per_frame = layer.cross_view_attn(tokens_per_frame)

            # ③ 跨帧时序注意力（时序一致性）
            tokens_per_frame = layer.cross_frame_attn(tokens_per_frame)

        return tokens_per_frame
```

**与全局注意力的对比**：
- 计算效率高，适合实时自动驾驶
- 利用驾驶场景的时空结构先验

### 4.4 训练策略

**密集几何伪标签生成**：
$$
\text{Depth}{pseudo} = \text_MONGE-2}(\text{Image}) \oplus \text{LiDAR}_{projected}
$$

**训练损失**：
$$
\mathcal{L} = \mathcal{L}{\text{point}} + \lambda_1 \mathcal{L}{\text{depth}} + \lambda_2 \mathcal{L}{\text{pose}}
$$

---

## 5 训练与推理伪代码

```python
def dvgt_forward(images):
    """
    images: [T, N, 3, H, W] - T帧 × N视角
    Returns: point_maps, ego_poses
    """
    # 1. 特征提取
    tokens = []
    for t in range(T):
        for n in range(N):
            img_tokens = dinov2.extract_features(images[t, n])  # [num_patches, D]
            ego_token = learnable_ego_token()  # [1, D]
            tokens.append(concat([img_tokens, ego_token]))

    # 2. 时空Transformer
    tokens = stack(tokens)  # [T*N, num_patches+1, D]
    tokens = add_temporal_pos_embedding(tokens)

    for block in geometry_transformer:
        # 帧内局部注意
        tokens = block.intra_view_attn(tokens)
        # 跨视角空间注意
        tokens = block.cross_view_attn(tokens)
        # 跨帧时序注意
        tokens = block.cross_frame_attn(tokens)

    # 3. 分离点和轨迹预测
    point_maps = point_head(tokens)      # [T*N, H*W, 3]
    ego_poses = pose_head(tokens[:, :, -1, :])  # [T, 6] (quaternion + translation)

    return point_maps, ego_poses


def inference(images):
    """
    images: 多帧多视角图像
    return: 度量尺度全局3D点云 + 自车轨迹
    """
    point_maps, ego_poses = dvgt_forward(images)

    # 转换到自车坐标系（第一帧）
    point_maps_ego = transform_to_ego_coordinate(point_maps, ego_poses)

    return point_maps_ego, ego_poses
```

---

## 6 实验结论

### 6.1 3D重建与深度估计

| 方法 | KITTI (δ<1.25↑) | nuScenes (δ<1.25↑) | Waymo (δ<1.25↑) | OpenScene | DDAD |
|------|-----------------|---------------------|-----------------|-----------|------|
| MapAnything | 0.58 | 0.52 | 0.48 | 0.45 | 0.42 |
| StreamVGG | 0.62 | 0.55 | 0.51 | 0.48 | 0.46 |
| CUT3R | 0.65 | 0.58 | 0.54 | 0.52 | 0.49 |
| VGGT | 0.68 | 0.61 | 0.56 | 0.54 | 0.51 |
| Driv3R | 0.71 | 0.64 | 0.59 | 0.56 | 0.53 |
| **DVGT** | **0.79** | **0.73** | **0.68** | **0.65** | **0.61** |

### 6.2 自车轨迹估计

| 方法 | KITTI (ATE↓) | nuScenes (ATE↓) |
|------|---------------|------------------|
| DUSt3R | 0.42 | 0.38 |
| MASt3R | 0.35 | 0.31 |
| VGGT | 0.28 | 0.25 |
| **DVGT** | **0.18** | **0.15** |

### 6.3 与驾驶专用模型对比

| 方法 | 3D重建 | 度量尺度 | 跨相机泛化 |
|------|--------|---------|-----------|
| BEVFormer | ✗ | ✗ | 弱 |
| TPVFormer | Occupancy | ✗ | 弱 |
| GaussianFormer | Occupancy | ✗ | 弱 |
| **DVGT** | **点云** | **✓** | **强** |

### 6.4 消融实验

| 组件 | 3D精度 (δ<1.25↑) |
|------|-----------------|
| 完整DVGT | 0.73 |
| w/o 跨视角注意 | 0.65 |
| w/o 跨帧注意 | 0.61 |
| w/o DINO骨干 | 0.52 |

---

## 7 KnowHow（核心洞察）

1. ** ego-centric表示的重要性**：将3D点云统一到自车坐标系而非相机坐标系，是实现跨相机泛化的关键

2. **无需后处理对齐的意义**：直接预测度量尺度使模型输出可直接用于下游任务（如规划、控制）

3. **时空分解注意力的效率**：利用驾驶场景的结构先验（固定相机安装、时序连续性），比全局注意力更高效

4. **密集伪标签的价值**：通过MoGe-2 + LiDAR投影融合，生成密集几何真值，解决训练数据不足问题

5. **与VGGT的分工**：VGGT是通用模型，DVGT是针对自动驾驶领域定制；DVGT可看作VGGT在驾驶领域的深度适配

6. **点云 vs Occupancy**：连续3D点云比离散体素更能保留几何细节，适合高精度需求

7. **多数据集混合训练的优势**：跨数据集训练提升模型泛化性，适应不同相机配置和场景

8. **端到端的重要性**：从图像到3D几何的端到端学习，避免多阶段误差累积

---

## 8 arXiv Appendix 关键点总结

| Section | 核心内容 |
|---------|---------|
| **A. Implementation Details** | 模型配置、训练超参详情 |
| **B. Evaluation Metrics** | 评估指标定义（δ<1.25、ATE等） |
| **C. Video Demonstration** | 视频演示链接 |
| **D. Dataset Details** | 各数据集详细信息 |
| **E. Author Contributions** | 作者分工说明 |

---

## 9 总结

### 三大核心贡献

1. **端到端度量尺度3D重建**：无需后处理LiDAR对齐，直接从图像预测可用3D几何

2. **ego-centric统一表示**：自车坐标系点云表示，适配任意相机配置，实现跨场景泛化

3. **时空分解注意力**：高效利用驾驶场景结构，实时推理性能优异

### 与VGGT的关键差异

| 维度 | VGGT (通用) | DVGT (驾驶定制) |
|------|------------|----------------|
| **目标场景** | 通用 | 自动驾驶 |
| **坐标系** | 相机坐标系 | 自车坐标系 |
| **尺度** | 相对尺度 | 度量尺度 |
| **后处理** | 可选BA | 无需 |
| **泛化性** | 中等 | 强（跨相机） |
| **效率** | 中等 | 高（分解注意力） |

---

## 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2512.16919](https://arxiv.org/abs/2512.16919) |
| **代码** | [GitHub: wzzheng/DVGT](https://github.com/wzzheng/DVGT) |
| **项目主页** | [wzzheng.net/DVGT](https://wzzheng.net/DVGT) |
