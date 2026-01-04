# 什么是 Neural Machine Translation（NMT）？

NMT 是用一个神经网络，把“整句源语言”直接翻译成“整句目标语言”的模型。

它不是一句一句规则翻译，也不是词对词查表，
而是：先理解整句意思，再生成另一种语言的句子。

🌍 传统翻译 vs NMT（为什么 NMT 是突破）
❌ 传统机器翻译

词对词翻译；
手工规则；
容易出现语序混乱、语义错误

✅ 神经机器翻译（NMT）
端到端学习；
自动学语法、语义、对齐；
翻译更自然

### 🧱 NMT 的经典结构：Encoder–Decoder

这是 CS224N 的核心模型。

源语言句子 → Encoder → 语义表示 → Decoder → 目标语言句子

1️⃣ Encoder（编码器）：读懂原句

作用：把源语言句子“压缩成一个语义表示”

比如：I love natural language processing

Encoder（RNN / LSTM / BiLSTM）逐词读取，把整句话的信息编码进隐藏状态。

早期版本：最后一个 hidden state = 整句语义

⚠️ 问题：句子一长 → 信息丢失

2️⃣ Decoder（解码器）：生成翻译

Decoder 做的事：根据 Encoder 给的语义，一步一步生成目标语言

例如翻译成中文：我 → 喜欢 → 自然语言处理

Decoder 每一步：
看之前已经生成的词；
看 encoder 的语义；
预测下一个词

🚨 Encoder–Decoder 的早期致命问题

“把整句话塞进一个向量里，太挤了”

长句翻译质量急剧下降

信息瓶颈严重

👉 这直接导致了 Attention 的出现

🔦 Attention 机制（NMT 的灵魂）

一句话理解：

翻译每个词时，不看整句压缩向量，而是“回头看源句中最相关的词”。

Attention 在干嘛？（人话）

翻译：

I love natural language processing


当 Decoder 要生成：

“我” → 关注 “I”

“喜欢” → 关注 “love”

“自然语言处理” → 关注 “natural language processing”

👉 模型会自动学对齐关系

Attention 带来的巨大提升

解决长句问题

自动完成“词对齐”

翻译质量大幅提高

📌 Attention = 现代 NMT 的分水岭

完整 NMT + Attention 流程（一步到位）
Encoder: h1, h2, h3, ..., hn
Decoder step t:
  → 计算 attention 权重
  → 加权得到 context vector
  → 生成目标词 y_t

### NMT 的进化路线
阶段	模型
早期	RNN Encoder–Decoder
突破	RNN + Attention
现代	Transformer（Self-Attention）
