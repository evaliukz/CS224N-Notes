# The Transformer Model

是一种完全基于 attention 的序列模型，它让序列中的任意位置可以直接交互，并且可以高度并行训练，是现代大模型（GPT、BERT、LLM）的基础。

## 🌱 为什么会有 Transformer？（动机）

| 模型              | 特点                |
| --------------- | ----------------- |
| RNN             | 有顺序但慢             |
| LSTM            | 能记忆但难并行           |
| CNN             | 快但局部              |
| **Transformer** | **快 + 长依赖 + 可扩展** |


## 🧱 Transformer 的整体结构（鸟瞰）

以最经典的 Encoder–Decoder Transformer 为例

<pre>
Encoder (N 层)        Decoder (N 层)
----------------     ----------------
Self-Attention       Masked Self-Attention
Feed Forward         Cross-Attention
                     Feed Forward
</pre>

现在的 GPT 类模型只保留 Decoder 部分（自回归生成）。

## 🧠 Transformer 的核心组件

#### 1️⃣ Embedding + Positional Encoding（先解决“顺序”）

Transformer 本身 不知道顺序，所以必须补充位置信息。

做法：

Token → embedding

加上 position embedding（如 RoPE）

人话：

每个词不仅知道“我是谁”，还知道“我在第几位”。

#### 2️⃣ Self-Attention（Transformer 的灵魂）
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

#### 3️⃣ Multi-Head Attention（为什么要多头）面试题！！！

回忆一下 单头 self-attention 在做什么：

一个 token 用 一个 Q 去看所有 K，得到 一套权重，汇总出 一个上下文表示。

问题是：一句话里，关系是多种多样的

语法关系

语义关系

指代关系

位置关系

一个 attention 很难同时学好所有关系

👉 就像：
你让一个人同时当语文老师、逻辑老师、翻译官，很吃力。

** Multi-Head Attention 的核心思想：

与其让一个 attention “啥都看”，
不如让多个 attention“各看各的”。

每个 head：

有自己的一套 Q / K / V

关注点不同

学到不同的关系

比如：

一个 head 看语法

一个看语义

一个看位置关系

结果：

信息更丰富

表达能力更强

Step 1️⃣：把 embedding “拆成多份”

Step 2️⃣：每个 head 独立做 self-attention
对同一句话：

Head 1：可能学语法（主谓关系）

Head 2：可能学语义（相关词）

Head 3：可能学指代（this / that）

Head 4：可能学位置邻近性

…

⚠️ 这些不是人手写的，是模型自己学出来的

Step 3️⃣：把所有 head 的结果拼起来：

head1 output ⊕ head2 output ⊕ ... ⊕ head8 output → 再过一个线性层

👉 得到一个 融合多种视角的信息表示

|      | 单头 Attention | Multi-Head Attention |
| ---- | ------------ | -------------------- |
| 关注视角 | 单一           | **多个并行**             |
| 表达能力 | 有限           | **更强**               |
| 信息冲突 | 容易           | **被分散**              |
| 是否更贵 | 较省           | **更耗算力 & 显存**        |

问题：计算和显存更大

head 越多：Q/K/V projection 越多，attention 计算越多，KV cache 也要 按 head 存

👉 这就是为什么：head 数是重要的系统参数，inference 显存和 latency 会随 head 增长

head 不是越多越好，经验事实（很重要）：太多 head → 有些 head 学不到有用东西，后期模型甚至出现 head redundancy

所以：GPT / LLaMA 系列 head 数是精心平衡的

** Scaled Dot Product

用 Query 和 Key 的两个vector的dot product点积来衡量相关性，当ventor的dimensions很大时，这个product很大，所以我们把attention score除以 √d/h 防止数值过大，再用 softmax 得到注意力权重。

#### 4️⃣ Feed-Forward Network（FFN）

Attention 负责 信息交流
FFN 负责 非线性变换

结构很简单：

Linear → ReLU/GELU → Linear

人话：

attention 是“讨论”，FFN 是“各自消化吸收”。

#### 5️⃣ Residual + LayerNorm（稳定训练）

Residual connection（残差连接）就是：把输入直接“绕一圈”加到输出上，让网络在学新东西的同时，不丢原来的信息。

为什么要 residual connection？深层网络的问题是：层数一多 → 信息传不动，梯度容易消失，模型越深反而越难训练

Residual connection 的想法是：“你要是学不会新东西，至少别把旧的弄丢。”

y = x + F(x)

x：原始输入

F(x)：这一层新学到的变化

输出 = 原来的 + 新的


🤖 在 Transformer 里 residual 在哪？

每一层都有：

Self-attention 后：
x → Attention(x) → + x → LayerNorm

Feed-forward 后：
x → FFN(x) → + x → LayerNorm

👉 保证信息能一层一层顺畅传下去

Layer Normalization 是：在“每个样本内部”，把这一层的数值拉回到稳定范围，让模型训练更稳、更快。“不管你这一层算成什么样，我先帮你整理一下，再交给下一层。”

Layer Normalization 到底在“归一化”什么？

这是很多人最容易混的地方。

✅ LayerNorm 是：

对“同一个 token 的所有特征维度”做归一化

比如：

一个 token 的 hidden vector：
[2.0, 0.5, -1.0, 3.2]


LayerNorm 会：

计算这 4 个数的均值和方差

把它们变成：

均值 ≈ 0

方差 ≈ 1

👉 是“横着”归一化

❌ 它不是：

不是跨 batch

不是跨 token

不是跨样本

这一点和 Batch Normalization 完全不同。

🧩 LayerNorm 的标准流程（你不需要背公式）

对每个 token 的向量：

1️⃣ 算均值（mean）
2️⃣ 算方差（variance）
3️⃣ 标准化（减均值 / 除标准差）
4️⃣ 再加一个 可学习的缩放 γ 和偏移 β

最后一步很重要：

模型可以决定：

要不要真的标准化

标准化到什么程度


## 🔁 Encoder vs Decoder（一定要分清）

上面学的是transformer decoder，transformer encoder的唯一区别是self attension时有mask，就只看过去不看未来。二者的区别就是“能不能看未来”。

### 🧠 Encoder 在干什么？
Encoder 的核心特点：看整句，双向理解，不生成词

流程：输入一句话 -> 每个词 同时看前后所有词 -> 得到每个词的“上下文表示”

📌 典型模型：

BERT

Encoder-only Transformer

👉 Encoder 更像: 阅读理解高手，用于理解（classification / embedding）

### 🧠 Decoder 在干什么？ 
Decoder 的核心特点：一步一步生成，只能看过去，负责输出

流程（生成时）：

已生成：我

预测下一个：喜欢

再预测下一个：NLP

⚠️ 关键限制：

不能提前偷看未来的词

这通过 masked self-attention 实现。

📌 典型模型：

GPT

Decoder-only Transformer

👉 Decoder 更像：一边想一边写的作家，用于生成（language model）

<img width="573" height="523" alt="image" src="https://github.com/user-attachments/assets/0b376644-11ef-4db9-83ea-932cf5f18aa1" />


