# Self-attention 在干什么？

对句子里的每一个词，都做同一件事：

“我是谁？我该关注句子里的谁？我该从他们那里拿什么信息？”

这就是 Q / K / V:

对每个词，模型都会生成三样东西：

1️⃣ Query（Q）——「我现在关心什么？」

代表：当前词想找什么信息

2️⃣ Key（K）——「我能提供什么？」

代表：每个词的标签 / 特征

3️⃣ Value（V）——「真正的信息内容」

代表：这个词能贡献的语义信息

你可以把它想成：

用 Q 去匹配所有 K，再把匹配到的 V 加权读出来。

# Self-attention 的完整流程（一步一步）

以句子为例：

I love natural language processing

### Step 1️⃣ 每个词都变成 Q / K / V

所有词同时做（这一步能并行）：

I      → Q1, K1, V1
love   → Q2, K2, V2
...

### Step 2️⃣ 计算“相关性分数”

比如我们看 love：

用 Q(love)

去和 所有词的 K 计算相似度

得到一堆分数：

I        : 0.2
love     : 0.9
natural  : 0.4
language : 0.5
processing:0.3


👉 表示 “love” 和谁更相关

### Step 3️⃣ 用 softmax 变成权重

把分数变成 加起来等于 1 的权重：

I        : 0.10
love     : 0.40
natural  : 0.15
language : 0.20
processing:0.15

### Step 4️⃣ 加权求和 Value

把所有 V 按权重加起来：

new_representation(love)
= 0.10*V(I) + 0.40*V(love) + ...


👉 得到一个 “融合了上下文的新 love”

### Step 5️⃣ 对每个词都做一遍

所以：

I 看所有词

love 看所有词

processing 看所有词

👉 这就是 self-attention

### Self-attention 的“self”是什么意思？

Q、K、V 都来自同一句话（同一序列）

这就是 self-attention。
（如果 Q 来自别的句子，那就是 cross-attention）

### Self-attention vs RNN（关键差别）

| 对比    | RNN  | Self-Attention |
| ----- | ---- | -------------- |
| 信息传递  | 一步一步 | **直接任意两词**     |
| 并行    | ❌    | ✅              |
| 长距离依赖 | 难    | **非常强**        |
| 训练速度  | 慢    | **快**          |

### 🚨 self attention的问题

1️⃣ 计算复杂度太高（O(n²)）——最大问题
人话解释

句子里有 n 个 token

每个 token 都要“看”所有 token

所以一共要算 n × n 次相关性

👉 这就是 O(n²)。

2️⃣ 显存占用巨大（Memory bottleneck）

Self-attention 不只算得多，还存得多。

你需要存：

Attention scores（n×n）

Softmax 中间结果

反向传播用的中间 tensor

推理阶段的 KV cache

人话

模型不是被算慢的，是被显存撑死的。

📌 在 inference 中：

KV cache 大小 ∝ 层数 × hidden size × 序列长度

长对话 → 显存越来越满

3️⃣ 对“顺序”本身不敏感（需要额外补救）-- L8讲的Sequence Order

Self-attention 本身不知道谁在前、谁在后。

如果你把句子打乱顺序：

I love NLP

NLP love I


Self-attention 无法区分，除非你告诉它顺序。

👉 所以必须加：

Positional Encoding

Rotary / ALiBi / RoPE

📌 这也是 infra / architecture 的一个设计点。

其他问题：

| Self-attention 问题 | 系统 / 架构解决                       |
| ----------------- | ------------------------------- |
| O(n²) 计算          | Sparse / Linear Attention       |
| 显存爆炸              | FlashAttention / KV paging      |
| 顺序缺失              | RoPE / ALiBi                    |
| 推理慢               | KV cache / speculative decoding |
| 噪声干扰              | Attention masking / retrieval   |


# 什么是 sequence order？（一句话）

Sequence order 就是：词出现的先后顺序。

比如：

“我打了他”

“他打了我”

用的词一样，但顺序一换，意思完全变了。

不同模型怎么处理顺序？
1️⃣ RNN / LSTM（天然有顺序）

按顺序一个词一个词读

后面的词一定知道前面的词

顺序是“内建”的

人话：

RNN 天生知道谁先谁后。

2️⃣ CNN（部分顺序）

看连续 n 个词（n-gram）

知道局部顺序

不关心整体位置

人话：

CNN 知道“相邻顺序”，不知道“全局顺序”。

3️⃣ Self-Attention / Transformer（本身不知道顺序）

⚠️ 重点

Self-attention：

每个词同时看所有词

本身不知道哪个在前、哪个在后

👉 如果不加额外信息：

I love NLP
NLP love I

对模型是一样的。

### 🧠 Transformer 是怎么补顺序的？

通过 Positional Encoding（位置编码）：

给每个 token 一个“位置标签”。

例如：

I      + pos(1)
love   + pos(2)
NLP    + pos(3)


这样模型才能区分：

第一个词

第二个词

第三个词

🔑 常见的顺序编码方式（你只需知道名字）

Sinusoidal positional encoding

Learned positional embedding

RoPE（Rotary Positional Embedding）⭐

ALiBi（Attention Linear Bias）
