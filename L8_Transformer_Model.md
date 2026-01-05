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

#### 3️⃣ Multi-Head Attention（为什么要多头）

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

1️⃣ 计算和显存更大

head 越多：

Q/K/V projection 越多

attention 计算越多

KV cache 也要 按 head 存

👉 这就是为什么：

head 数是重要的系统参数

inference 显存和 latency 会随 head 增长

2️⃣ head 不是越多越好

经验事实（很重要）：

太多 head → 有些 head 学不到有用东西

后期模型甚至出现 head redundancy

所以：

GPT / LLaMA 系列 head 数是精心平衡的

#### 4️⃣ Feed-Forward Network（FFN）

Attention 负责 信息交流
FFN 负责 非线性变换

结构很简单：

Linear → ReLU/GELU → Linear

人话：

attention 是“讨论”，FFN 是“各自消化吸收”。

#### 5️⃣ Residual + LayerNorm（稳定训练）

每个子模块都有：

x → SubLayer → + x → LayerNorm


作用：

防止梯度消失

让深层网络好训练

### 🔁 Encoder vs Decoder（一定要分清）
Encoder（BERT 类）

Self-attention 看 全句

用于理解（classification / embedding）

Decoder（GPT 类）

Masked self-attention

只能看前面的 token

用于生成（language model）

### 🧠 Transformer 是怎么训练的？

以 GPT 为例：

任务：预测下一个 token

Loss：cross-entropy

使用 teacher forcing

大规模数据 + 大模型

模型学到的不是规则，而是：

语言的统计结构 + 世界知识

| 问题              | 含义                          |
| --------------- | --------------------------- |
| O(n²) attention | 长序列算不动                      |
| 显存占用大           | attention matrix + KV cache |
| 推理慢             | context 越长越慢                |
| 数据需求大           | 小数据效果差                      |
