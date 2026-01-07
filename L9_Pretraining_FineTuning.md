# Pretraining

先让模型在海量文本上自学语言规律，再拿去做具体任务。

就像：小孩先大量听书、看故事 → 学会语法和词义

以后再去写作文、做阅读理解。

### Algorithm: Byte-Pair Encoding 

是一种把文本拆成“子词（subword）”的算法，通过不断合并最常出现的字符对，得到一个适合模型使用的词表。

为什么需要 BPE？

传统按“空格分词”的问题：

新词：ophthalmologist, OMM, Entra360
👉 词表里可能没有 → 变成 <UNK>

不同形态：play / played / playing
👉 被当成完全不同的词

BPE 的思路：

别只用“完整单词”，而是用 subword 拼。

例如：

un + interrupt + ed
neuro + ophthalmologist

👉 再罕见的词也能被表示出来。

#### BPE 的优点与问题

| 方面               | 说明        |
| ---------------- | --------- |
| ✅ 解决 OOV         | 不再依赖完整词   |
| ✅ 词表更小           | 几万即可覆盖    |
| ✅ 适合 Transformer | token 化一致 |
| ❌ 语义被切碎          | 过细会增加长度   |
| ❌ 中文里也会拆         | 但影响较小     |

BPE 通过 subword 建立词表，减少 <UNK>，是现代 NMT 与 LLM 的标准 tokenizer。

### Pretraining Overview

1️⃣ 它在做什么？

Language Model 先在海量无标注文本上自监督学习：

给定上文 → 预测下一个 token（GPT）

Mask 一些词 → 预测被遮住的词

损失函数通常是：cross-entropy
输出是：每个 token 的通用 embedding + 参数。

2️⃣ 数据

不需要人工标签

用：书，网页，代码，对话记录，这一步最像“狂读全世界的书”。

3️⃣ 结果

预训练后模型会拥有：词义知识，语法结构，世界常识，低 perplexity

👉 因此随机初始化被彻底取代。

### Pretraining encoder

Pretraining Encoders = 用 Transformer Encoder 在双向上下文里自监督学习，得到“理解型语言表示”。

👉 典型代表：BERT、RoBERTa、DistilBERT 等。

1️⃣ Masked Language Modeling

做法：

随机遮住 15% 的 (sub)words token, 用 Encoder 的双向 self-attention 去猜这些(sub)words

例子：

Input:  The [MASK] is expensive.
Label:  book


Encoder 能同时看到：

左边 the

右边 is expensive

👉 因此学到的是：

token → contextual embedding

2️⃣ 早期还有 NSP / 句间目标

判断两句是否本来相邻

现在很多模型已去掉

#### BERT（Bidirectional Encoder Representations from Transformers）

input = token embeddings + segment embeddings + position embeddings

1️⃣ Masked Language Modeling

做法很像给句子挖洞：

原句： I love natural language processing  
输入： I love [MASK] language processing  
标签： natural


Encoder 会同时看到：

左边词：I, love

右边词：processing

用这些信息去猜 mask 的词

👉 因此学到的是：

上下文相关的向量表示（contextual embedding）

Bert有动态表示：同一个词在不同句子里向量不同

“bank” ：river bank & national bank 👉 BERT 能区分。

2️⃣ Next Sentence Prediction（早期目标）

给两句：

sent1: I bought a book

sent2: It is expensive

让模型判断：

这两句是不是本来连在一起？

👉 让模型具备句间结构感。


** Encoder 预训练的问题

计算量 O(n²)

长序列显存大

fine-tuning 可能过拟合



# Fine Tuning

1️⃣ 为什么要微调？

预训练只学会：

“这句话像不像人话”

但具体任务需要：

识别人名（NER）

做情感分类

问答

生成风格

👉 这些目标和“预测下一词”不完全一样。

2️⃣ 微调流程
Pretrained Model
→ 用任务数据继续训练
→ 小学习率
→ 新的 loss / head

例子

NER：
每个 token 预测标签：B-PER, I-ORG…

情感分析：
句子级 softmax 分类

Chat 场景：
instruction tuning + RLHF

3️⃣ 数据

需要标注数据

规模远小于 pretraining：

几万~几百万条

微调更像“专业补习”。

#### Pretraining 的问题

训练成本极高

长序列显存爆炸

学到的偏见会保留

#### Fine-tuning 的问题

可能过拟合

会遗忘部分通用能力（catastrophic forgetting）

需要设计 softmax / task head
