# Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills

## 引用信息

| 字段 | 内容 |
|------|------|
| **标题** | Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills |
| **arXiv** | [2503.12533](https://arxiv.org/abs/2503.12533) [cs.RO] |
| **作者** | Haoqi Yuan, Yu Bai, Yuhui Fu, Bohan Zhou, Yicheng Feng, Xinrun Xu, Yi Zhan, Börje F. Karlsson, Zongqing Lu |
| **机构** | 北京大学（PKU）· 北京人工智能研究院（BAAI）· BeingBeyond |
| **顶会/顶刊** | ICML 2025 / arXiv 2025 |
| **发布日期** | 2025-03-16（v1），2025-05-11（v2） |
| **项目主页** | [beingbeyond.github.io/Being-0](https://beingbeyond.github.io/Being-0) |

---

## 1. Motivation（问题背景）

### 1.1 人形机器人的终极目标

构建能够在现实世界中达到人类水平的具身智能体，是人形机器人研究的终极目标。近年来，Foundation Models（FMs，大模型）在高层认知（如指令理解、任务规划、推理）取得突破，同时人形机器人的底层技能（运动、灵巧操作）也在快速发展。然而，将这两者直接组合会遇到严重的**复合误差**和**延迟不匹配**问题。

### 1.2 直接组合 FM + 机器技能的三大问题

| 问题 | 描述 |
|------|------|
| **复合误差** | 长时域任务中，每步技能的微小误差会沿时间步累积，导致最终任务失败 |
| **延迟不匹配** | FM（通常在云端）推理速度慢，而低层技能需要高频率实时控制，两者难以协同 |
| **双足特殊性** | 轮式机器人可以精确跟踪导航轨迹后停止再操作；人形机器人双足站立天然不稳定，需要持续调整重心 |

### 1.3 现有方案的局限

现有 FM-based Agent 框架（如基于 GPT-4o 的方案）存在两大局限：
- **推理效率低**：FM 参数量大、延迟高，无法对环境变化做出快速反应
- **具身场景理解弱**：FM 的视觉语言理解泛化能力强，但直接用于控制人形机器人时，对导航-操作交替阶段的场景理解不足

---

## 2. 一句话总结

**Being-0 是一套分层 Agent 框架，通过 VLM-powered Connector 模块在高层 FM 规划与底层模块化技能库之间架起桥梁，使人型机器人在仅依赖低成本本地部署的情况下，以 4.2 倍效率完成 84.4% 平均完成率的长时域具身任务。**

---

## 拟人化开篇

让一个人形机器人"像人一样"完成任务，你需要什么？

不是简单的「看到→规划→执行」流水线——因为人类的决策是**分层递进**的：先想"我要泡一杯咖啡"，再想"去厨房，找咖啡机，拿杯子……"，最后才调用早已练习过千百遍的"伸手抓取"这个基础运动技能。

现有的 FM+机器人方案，恰恰是在最后这一步出了岔子：让一个几百亿参数的大模型去实时控制机器人关节，就像让一位象棋大师同时用显微镜观察棋盘——层次混乱，延迟极高。

Being-0 的答案是：**层次不能乱，但层次之间的"翻译官"要选对。** 他们造了一个轻量级的 VLM 当 Connector，把大模型的语言规划"翻译"成机器人能听懂的具体技能指令，同时协调下肢运动和上肢操作，终于让人形机器人能够稳定地完成"做一杯咖啡"这样的长时域任务。

---

## 3. 核心贡献

1. **分层 Agent 框架**：FM（云端）+ Connector（本地 VLM）+ 模块化技能库（本地），每层最优部署
2. **VLM-powered Connector**：轻量级 VLM 连接高层语言规划和底层技能执行，增强具身决策
3. **全身人形控制**：支持多自由度（41 DoF）人形机器人，包含灵巧手和主动视觉
4. **实时高效**：除 FM 外所有模块部署在低成本本地设备，导航效率较纯 FM Agent 提升 4.2 倍

---

## 4. 方法详述

### 4.1 系统架构总览

Being-0 由三大组件构成：

```
┌─────────────────────────────────────────────────────┐
│  Foundation Model (FM)      [云端]                  │
│  · 指令理解、任务分解、推理                          │
│  · 输出：语言计划 (language plan)                    │
└────────────────┬────────────────────────────────────┘
                 │ 语言计划
                 ▼
┌─────────────────────────────────────────────────────┐
│  Connector (VLM, 轻量级)    [本地边缘设备]           │
│  · 视觉观察 + 语言计划 → 技能命令                    │
│  · 协调下肢运动 + 上肢操作                          │
│  · 输出：技能ID + 实时关节指令                       │
└────────────────┬────────────────────────────────────┘
                 │ 技能调用
                 ▼
┌─────────────────────────────────────────────────────┐
│  Modular Skill Library      [本地实时]               │
│  · Locomotion Skill（joystick 命令）               │
│  · Manipulation Skills（灵巧手操作）                 │
└─────────────────────────────────────────────────────┘
```

**关键设计洞察**：不同于轮式机器人，人形机器人的下身（运动）和上身（操作）功能天然解耦——这一观察使得分别开发独立的运动技能和操作技能成为可能，大幅降低了技能库构建的复杂度。

### 4.2 Modular Skill Library（模块化技能库）

**下身稳定运动**（Locomotion）：基于摇杆命令的 RL 训练策略，控制下身关节实现多方向导航和站立平衡。

**上身灵巧操作**（Manipulation）：通过远程操作（teleoperation）和模仿学习（imitation learning）获取的操控技能集合，每个技能配有语言描述（如"抓取杯子"、"打开抽屉"）。

技能库特点：
- 每个技能独立，专注单一操作任务
- 技能有语言标签，支持 Connector 通过语言调用
- 下身固定，上身操作——避免全身策略同时控制运动和操作的高维复杂性

### 4.3 Connector 模块（核心创新）

Connector 是一个 VLM，其输入为：
- FM 输出的语言计划（text）
- 当前视觉观察（双目 RGB 图像）

输出为：
- 技能调用指令（skill command + 参数）
- 下肢运动调整（locomotion adjustment）

**训练数据**：第一人称室内导航视角图像，标注语言指令、物体标签和边界框。

**核心能力**：
1. **技能规划**：将高层语言计划翻译为具体技能调用
2. **实时协调**：在运动和操作之间动态切换，调整人形机器人姿态为操作提供更好的初始状态
3. **高频控制**：比 FM 更高的控制频率（本地 VLM vs 云端 FM）

### 4.4 伪代码

```python
import torch
import torch.nn as nn

class Being0Connector(nn.Module):
    """
    Connector: VLM-based bridge between FM and Skill Library
    输入: 图像 tokens + 语言计划 tokens
    输出: 技能ID + 实时运动调整
    """
    def __init__(self, vision_encoder, vlm_backbone, num_skills, skill_embed_dim=128):
        super().__init__()
        self.vision_encoder = vision_encoder      # 视觉编码器（如 ResNet/EfficientNet）
        self.vlm = vlm_backbone                    # 轻量级 VLM（如 LLaVA-7B 级别）
        self.skill_head = nn.Linear(vlm_embed_dim, num_skills)  # 技能分类头
        self.locomotion_head = nn.Linear(vlm_embed_dim, 6)       # 下肢运动调整 (dx, dy, dtheta, ...)
        self.parameter_head = nn.Linear(vlm_embed_dim, max_skill_params)  # 技能参数

    def forward(self, images_left, images_right, language_plan, proprioception):
        # Step 1: 视觉编码
        vis_feat = self.vision_encoder(images_left, images_right)  # (B, vis_dim)

        # Step 2: VLM 融合视觉 + 语言
        vlm_input = torch.cat([vis_feat, language_plan_tokens], dim=-1)
        vlm_out = self.vlm(vlm_input)  # (B, vlm_embed_dim)

        # Step 3: 解码三层输出
        skill_logits = self.skill_head(vlm_out)          # (B, num_skills)
        skill_id = torch.argmax(skill_logits, dim=-1)    # 选最相关技能
        skill_params = self.parameter_head(vlm_out)      # 技能参数
        locomotion_adjust = self.locomotion_head(vlm_out)  # 运动调整

        return skill_id, skill_params, locomotion_adjust


def being0_control_loop(connector, skill_library, fm, max_steps=100):
    """
    Being-0 主控制循环
    """
    observations = initialize_observations()

    for step in range(max_steps):
        # 1. FM 高层规划（云端，低频）
        if step % FM_UPDATE_INTERVAL == 0:
            language_plan = fm(observations.task_description, observations.history)

        # 2. Connector 本地决策（边缘，高频）
        skill_id, skill_params, loco_adj = connector(
            observations.images_left,
            observations.images_right,
            language_plan,
            observations.proprioception
        )

        # 3. 技能库执行低层控制（实时）
        if skill_id.is_locomotion:
            action = skill_library.execute_locomotion(loco_adj)
        else:
            action = skill_library.execute_manipulation(skill_id, skill_params)

        observations = robot.step(action)

        # 4. 成功检测（如需要，FM 可介入重规划）
        if skill_library.is_skill_done(skill_id):
            continue  # 进入下一技能
```

### 4.5 系统部署策略

| 组件 | 部署位置 | 原因 |
|------|---------|------|
| Foundation Model | 云端 | 参数量大、延迟高，但不需要实时性 |
| Connector (VLM) | 本地边缘设备 | 需要高频决策，低延迟 |
| Skill Library | 本地实时控制器 | 毫秒级控制要求 |

---

## 5. 实验结论

### 5.1 主要结果

Being-0 在大型室内环境中进行测试，任务包含导航、操作及两者组合的长时域任务。

| 指标 | 数值 |
|------|------|
| **长时域任务平均完成率** | **84.4%** |
| **导航效率提升**（对比纯 FM Agent） | **4.2×** |
| 机器人配置 | 全尺寸人形，41 DoF（下身13 + 上肢14 + 灵巧手12 + 颈部2） |

### 5.2 消融实验关键发现

**Connector 的贡献**：移除 Connector 直接用 FM 控制 → 完成率显著下降。说明 FM 无法胜任高频率具身控制。

**主动视觉的贡献**：使用主动视觉（双目 RGB + 颈部 2-DoF）对比固定相机 → 完成率提升明显，尤其在操作类任务中。

**模块化 vs 全身策略**：模块化技能库（分离运动+操作）明显优于单一全身策略——因为后者同时学习所有技能复杂度过高。

---

## 6. KnowHow（核心洞察）

### 洞察 1：下身/上身功能解耦是关键设计洞察

大多数任务中，下身负责导航，上身负责操作——这个看似简单的观察，使得分别开发独立技能成为可能，大幅降低了技能库构建的复杂度。轮式机器人无法借鉴这一设计，因为轮式机器人不需要"站立平衡"这一本能需求。

### 洞察 2：Connector 本质是"翻译层"而非"规划层"

Connector 不做高层规划（那是 FM 的职责），而是做**视觉-语言对齐 + 技能选择 + 实时调整**。这使得 Connector 可以用小得多的模型（轻量级 VLM）实现，降低了部署成本。

### 洞察 3：FM 的延迟不匹配是具身 Agent 的核心瓶颈

FM（通常云端）推理延迟在秒级，而机器人控制需要毫秒级响应。Being-0 的解法是让 FM 专注高层低频决策（任务级），Connector 专注低层高频控制（关节级）——层次化缓解延迟问题。

### 洞察 4：远程操作 + 模仿学习是获取操作技能的有效路径

对于上身操作技能，Being-0 采用遥控操作演示 + 模仿学习，避免了从头训练强化学习策略的高样本复杂度。

### 洞察 5：人形机器人的"视觉焦点"需要主动控制

固定相机视角会限制操作精度；主动视觉（颈部 2-DoF 调节相机角度）让人形机器人可以在操作前先"看一眼"目标物体，显著提升抓取和操作成功率。

### 洞察 6：实时性不等于低智能

Connector 以高频运行（本地 VLM），但并不做"笨拙"的直接映射——它接收 FM 的语言计划作为条件输入，因此既享有 FM 的语义理解能力，又具备实时反应速度。

---

## 7. Appendix 关键点

### A.1 硬件配置详情

- **下身 13 DoF**：每条腿 6 DoF（髋、膝、踝）+ 躯干 1 DoF
- **上肢 14 DoF**：每条手臂 7 DoF（肩×2、肘、腕×3）
- **灵巧手 12 DoF**：每只手 6 DoF
- **颈部 2 DoF**：俯仰 + 偏航
- **相机**：双目 RGB，分辨率 640×480，视角约 90°

### A.2 FM 模型选择

论文使用通用 FM-based Agent 框架（Tan et al., 2024），实际调用了 GPT-4o 或同等能力的 VLM。FM 仅在云端运行，不参与本地实时控制。

### A.3 Connector 训练数据

第一人称室内导航图像，标注内容：
- 语言指令（navigation commands）
- 物体标签（object categories）
- 边界框（bounding boxes）

### A.4 技能库规模

- Locomotion skills：1 个（基于 RL 的通用行走策略，支持多方向）
- Manipulation skills：多个独立技能，每个技能对应一种操作（如抓取、放置、推拉等），通过模仿学习从远程操作演示中获取

---

## 8. 与 Latent-WAM 的本质对比

> **重要说明**：Being-0 与 Latent-WAM 虽然名称中都有 "WAM"（World Action Model），但实际上是两个完全不同方向的工作：
> - **Latent-WAM** 是**端到端自动驾驶**领域，聚焦于如何用隐世界模型做轨迹规划
> - **Being-0** 是**人形机器人具身智能**领域，聚焦于 FM+模块化技能的分层 Agent 框架

| 维度 | **Being-0** | **Latent-WAM** |
|------|-------------|----------------|
| **领域** | 人形机器人具身智能 | 端到端自动驾驶 |
| **核心问题** | FM+技能如何协同控制人形 | 世界模型如何压缩感知+预测未来 |
| **架构路线** | 分层 Agent：FM(云) → Connector(VLM) → 技能库(本地) | 单阶段端到端：SCWE → DLWM → Trajectory Decoder |
| **规划粒度** | 技能级（Skill-level），离散的技能调用 | 轨迹级（Trajectory-level），连续轨迹点 |
| **控制频率** | Connector 高频（本地），FM 低频（云端） | 全流程在车端，SCWE+DLWM 联合优化 |
| **训练范式** | 模仿学习（操作）+ RL（运动）+ 有监督 VLM（Connector） | 自监督视频预测 + 有监督轨迹回归 |
| **世界模型** | **无**显式世界模型；依赖 FM 的隐式世界知识 | **有**显式隐世界模型（DLWM，causal Transformer） |
| **感知方式** | 双目 RGB 图像（主动视觉） | 多视角相机图像 |
| **规划目标** | 完成自然语言描述的长时域任务（如"做咖啡"） | 预测 4 秒未来轨迹（EPDMS/HD-Score） |
| **代表性指标** | 任务完成率 84.4% | NAVSIM EPDMS 89.3 |
| **参数量** | FM（云端，巨大）+ Connector（轻量 VLM）+ 技能（专用小模型） | 紧凑 104M（感知+世界模型+规划器联合） |
| **部署方式** | FM 云端，其他本地 | 全流程车端本地部署 |
| **核心创新** | VLM-powered Connector 填补语言规划与技能执行之间的粒度鸿沟 | SCWE 几何蒸馏 + DLWM 因果时序预测，用隐表征同时解决空间压缩和动态预测 |
| **典型任务** | "去厨房给我拿一个苹果" | "前方车辆减速，保持车道，4 秒后汇入匝道" |

### 核心本质区别

**Being-0 是"层次化的语言-动作"框架**，核心问题是**粒度对齐**：如何把大模型的高层语言规划，变成机器人关节的低层控制指令。答案是一个 VLM 当"翻译官"（Connector），不需要世界模型。

**Latent-WAM 是"感知压缩+时序预测"框架**，核心问题是**压缩效率**：如何在保留3D几何信息和时序动态的前提下，把海量相机图像压成几KB的隐表征用来轨迹规划。答案是 SCWE（几何蒸馏压缩）+ DLWM（因果 Transformer 做未来预测）。

**一个向下（控制），一个向内（表征）**——两者从问题定义到技术路线都完全不同，但都是各自领域将 VLM/世界模型落地的有益探索。

---

## 9. 参考链接

| 资源 | 链接 |
|------|------|
| **论文** | [arXiv:2503.12533](https://arxiv.org/abs/2503.12533) |
| **项目主页** | [beingbeyond.github.io/Being-0](https://beingbeyond.github.io/Being-0) |
| **代码** | 暂未开源（论文未提供） |

---

*最后更新：2026-04-23*
