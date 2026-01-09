
# Zero-shot Learning（零样本学习） - GPT2
是一种让模型在从未见过某个类别训练样本的情况下，也能正确识别或推断该类别的机器学习方法。

传统机器学习是：

见过猫 → 学会猫 → 识别猫
没见过斑马 → 识别不了斑马

**Zero-shot Learning（ZSL）**是：

虽然没见过斑马
但知道：斑马 = 像马 + 有黑白条纹
如果看到一个“像马、有黑白条纹”的东西 → 推断是斑马

👉 核心思想：
用“语义知识”来弥补“数据缺失”

### 正式定义

Zero-shot Learning：
模型在训练阶段 完全没有见过目标类别的样本，
但在测试阶段 仍能识别这些新类别。

通常通过：

文本描述

属性（attributes）

向量嵌入（embedding）

来连接“已知类别”和“未知类别”。

# Few-shot Learning（零样本学习） - GPT3

### Few-shot learning 指的是：

模型在推理阶段，仅通过提示中提供的少量示例（通常 1–10 个），就能完成一个新任务，而无需重新训练或微调模型参数。

在 GPT-3 中，这种能力更准确的名字是：In-Context Learning（上下文学习）

也就是说：示例不是训练数据， Prompt 本身就是“任务定义 + 示例 + 输入”

### GPT-3 的 Few-shot 是怎么“学”的？

关键点先给结论

GPT-3 的 Few-shot learning 不涉及任何参数更新，它只是通过 Transformer 的注意力机制，在上下文中识别模式并延续该模式

示例（经典 Few-shot Prompt）
English: happy → French: heureux
English: sad → French: triste
English: fast → French:


模型会输出：rapide


模型在做的事情是：

识别输入输出的对应结构

归纳映射规则

在新输入上套用该规则

👉 像人在“看例题，做新题”

### 机制层面：为什么 Transformer 能做到 Few-shot？

1️⃣ 自注意力 = 模式对齐器

Transformer 会：

同时关注 prompt 中的多个示例

学到「A → B」的映射结构

在新输入上复制这种结构

从计算角度看：

模型在推理时，在隐空间中“模拟”了一个小学习器

2️⃣ 大模型 = 隐式 Meta-Learner

GPT-3 在预训练中见过：

无数任务格式

无数“示例 → 规律 → 预测”的文本结构

结果是：

模型学会了 “如何从上下文中学规则”

但不是通过显式的 meta-learning loss，而是 规模驱动的涌现能力（emergence）

### GPT-3 的 Few-shot 为什么比以前模型强很多？
核心原因只有一个：规模

当模型足够大时：它可以在 forward pass 中，临时表示一个规则、一个映射、甚至一个小算法

这是 GPT-3 论文的核心发现之一：模型规模越大，Few-shot 性能提升越明显

这也是为什么：GPT-2 几乎不行，GPT-3 开始可用，GPT-4 变得稳定

### Few-shot 的关键局限（非常重要）

GPT-3 的 Few-shot 不是万能的：

❌ 对示例顺序敏感

❌ 对 phrasing 极其敏感

❌ 示例过多会“挤掉”有用信息

❌ 不会真正泛化出新概念

因此它更像是：

上下文条件化的模式延续器

而不是人类意义上的学习系统。

### 工业视角：为什么 Few-shot 改变了 AI 系统设计？

Few-shot 让：

Prompt 成为 接口（Interface）

模型成为 通用推理引擎

任务逻辑从代码 → 文本

这直接影响了：

Prompt engineering

Agent system

Tool calling

RAG + Few-shot 组合设计

### 一句话技术总结：

GPT-3 的 Few-shot Learning 是一种基于 Transformer 注意力机制的上下文适配能力，模型在不更新参数的情况下，通过对少量示例进行模式归纳，在推理阶段动态执行新任务。

