# Transformer Model 

是一种完全基于 attention 的序列模型，它让序列中的任意位置可以直接交互，并且可以高度并行训练，是现代大模型（GPT、BERT、LLM）的基础。

### 🌱 为什么会有 Transformer？（动机）

| 模型              | 特点                |
| --------------- | ----------------- |
| RNN             | 有顺序但慢             |
| LSTM            | 能记忆但难并行           |
| CNN             | 快但局部              |
| **Transformer** | **快 + 长依赖 + 可扩展** |


### 🧱 Transformer 的整体结构（鸟瞰）

以最经典的 Encoder–Decoder Transformer 为例

<pre>
Encoder (N 层)        Decoder (N 层)
----------------     ----------------
Self-Attention       Masked Self-Attention
Feed Forward         Cross-Attention
                     Feed Forward
</pre>

现在的 GPT 类模型只保留 Decoder 部分（自回归生成）。

### 🧠 Transformer 的核心组件

1️⃣ Embedding + Positional Encoding（先解决“顺序”）

Transformer 本身 不知道顺序，所以必须补充位置信息。

做法：

Token → embedding

加上 position embedding（如 RoPE）

人话：

每个词不仅知道“我是谁”，还知道“我在第几位”。

2️⃣ Self-Attention（Transformer 的灵魂）
它在干什么？

对序列中 每一个 token：

根据相关性，汇总整个序列中对自己最重要的信息。

机制：

每个 token 生成 Q / K / V

Q 和所有 K 算相似度

softmax 得权重

加权 V → 新表示

📌 关键能力：

任意两词 一步到位

长距离依赖强

完全并行

3️⃣ Multi-Head Attention（为什么要多头）

不是只做一次 attention，而是：

用多个“视角”同时做 attention

比如：

一个 head 看语法

一个看语义

一个看位置关系

结果：

信息更丰富

表达能力更强

4️⃣ Feed-Forward Network（FFN）

Attention 负责 信息交流
FFN 负责 非线性变换

结构很简单：

Linear → ReLU/GELU → Linear


人话：

attention 是“讨论”，FFN 是“各自消化吸收”。

5️⃣ Residual + LayerNorm（稳定训练）

每个子模块都有：

x → SubLayer → + x → LayerNorm


作用：

防止梯度消失

让深层网络好训练

🔁 Encoder vs Decoder（一定要分清）
Encoder（BERT 类）

Self-attention 看 全句

用于理解（classification / embedding）

Decoder（GPT 类）

Masked self-attention

只能看前面的 token

用于生成（language model）

🧠 Transformer 是怎么训练的？

以 GPT 为例：

任务：预测下一个 token

Loss：cross-entropy

使用 teacher forcing

大规模数据 + 大模型

模型学到的不是规则，而是：

语言的统计结构 + 世界知识
