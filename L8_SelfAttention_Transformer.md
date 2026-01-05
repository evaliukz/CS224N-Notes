# Self-attention 在干什么？（核心直觉）

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
