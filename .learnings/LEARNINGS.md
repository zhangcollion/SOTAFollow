# Learnings Log

## [LRN-20260421-001] best_practice

**Logged**: 2026-04-21T00:00:00Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
论文精读报告标准流程：从 arXiv 获取图片、公式校验、proofreader 校对

### Details
在撰写 FlowGRPO 和 π0.7 精读报告过程中，总结出完整流程：

**1. arXiv 图片获取**
```bash
# 获取论文 HTML 页面
curl -s "https://arxiv.org/html/{paper_id}v{version}" | grep -E 'figure.*id|x\.png|x\.jpg'

# 图片 URL 格式：https://arxiv.org/html/{paper_id}v{version}/{xN}.png
```

**2. 公式校验**
- 从 arXiv HTML 提取公式原文（grep 'equation|formula|nabla|sigma|DKL'）
- 对照论文 PDF 核实公式编号和内容
- 特别注意：时间步索引方向、符号下标

**3. 图片插入规则**
- 图片必须分散插入到对应章节
- 禁止建立独立的"论文原图解析"章节
- 对应关系：算法框架图→方法详述、实验图→实验结论、消融图→消融小节

**4. Proofreader 校对要点**
- 高优先级：公式错误、时间步索引、缺失章节（伪代码、KnowHow）
- 中优先级：数据准确性、公式编号
- 低优先级：格式问题、拼写

### Suggested Action
撰写新精读报告时，按照以下顺序：
1. 获取 arXiv HTML，提取图片 URL 和公式
2. 按照 8 个必含章节组织内容（一句话总结→核心贡献→方法详述→伪代码→实验结论→KnowHow→Appendix→总结）
3. 插入图片到对应章节
4. 调用 proofreader agent 校对
5. 根据校对意见修正后提交

### Metadata
- Source: conversation
- Related Files: CLAUDE.md, RL/FlowGRPO_精读报告.md, VLA/π0.7_精读报告.md
- Tags: 精读报告, arXiv, proofreader
- Pattern-Key: docs.reading-report-standard流程
- Recurrence-Count: 2

---
