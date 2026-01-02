# 一、RNN 是怎么训练的？（Training）
## 🔁 1️⃣ RNN 的基本训练方式：BPTT

RNN 的训练叫做 Backpropagation Through Time（时间反向传播）。

直觉理解：

把 RNN 在时间轴上“拉直”，变成一条很长的神经网络，然后从后往前算梯度。

示意：

x1 → h1 → y1

     ↓
     
x2 → h2 → y2

     ↓
     
x3 → h3 → y3

训练步骤

给定一个句子

RNN 逐词读入，预测下一个词

计算 loss（预测 vs 真值）

从最后一个时间步开始，一路反传到最早的时间步

更新参数

## ⚠️ 2️⃣ 实际训练中的技巧

为了让训练可行，通常会用：

Truncated BPTT：
不反传整个句子，只反传最近 N 步（比如 20–50 步）

Gradient Clipping：
防止梯度爆炸

Teacher Forcing：
训练时用真实词作为下一步输入（而不是模型预测）


# 🚀 二、RNN 用来干什么？（Use）
📌 典型应用场景
| 任务              | RNN 在做什么 |
| --------------- | -------- |
| 语言模型            | 预测下一个词   |
| 机器翻译（早期）        | 编码源语言句子  |
| 语音识别            | 处理时间序列音频 |
| 情感分析            | 聚合整句情绪   |
| 序列标注（NER / POS） | 建模上下文    |

一句话总结：

RNN 适合处理“有顺序、长度不固定”的数据。

# 三、RNN 的核心问题（Problems）

这是 CS224N 非常重点 的部分 👇

❌ 1️⃣ 梯度消失（Vanishing Gradient）

当序列很长：

梯度在反向传播中不断乘小于 1 的数

越往前 → 梯度越小 → 前面的信息学不到

结果：

RNN 很难记住很久之前的词

❌ 2️⃣ 梯度爆炸（Exploding Gradient）

梯度不断变大

参数更新失控 → loss 直接炸掉

解决：

Gradient clipping

❌ 3️⃣ 长距离依赖问题

例如：

“The book that I bought last year and really enjoyed was expensive.”

主语和谓语隔得很远
标准 RNN 很难记住 “book” 才是主语

❌ 4️⃣ 训练慢，不能并行

RNN 是 严格时间顺序 的：

h_t 必须等 h_{t-1}


→ 无法像 Transformer 那样并行训练
→ 在大数据上效率很差

❌ 5️⃣ 曝光偏差（Exposure Bias）

训练时：

用真实词作为下一步输入（teacher forcing）

测试时：

用模型自己预测的词

→ 一旦预测错一个词，后面会 一路错下去

# 四、如何改进 RNN？（历史演进）

| 改进          | 解决什么问题         |
| ----------- | -------------- |
| LSTM / GRU  | 记忆长距离依赖        |
| Attention   | 不靠记忆，直接“看”相关位置 |
| Transformer | 完全抛弃 RNN，支持并行  |
