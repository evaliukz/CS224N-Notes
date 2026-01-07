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


### Parameter Efficient Fine-Tuning（PEFT）

PEFT = 只改动“很少一部分参数”就完成微调，而不是把整个大模型都重新训练。

目标：
👉 省钱
👉 省显存
👉 省时间
👉 还能保留 pretrained 的通用能力。

#### 🌱 为什么需要 PEFT？

如果按传统 fine tuning：

模型 7B / 70B 参数

每次微调都要：

全量梯度

optimizer state

checkpoint

问题：

需要巨大 GPU 显存

数据一多就很贵

还可能过拟合

👉 PEFT 的思路：

大部分参数冻结，只加“外挂小模块”。

#### 🧩 PEFT 的几种主流做法

1️⃣ LoRA（最最常用）

Low-Rank Adaptation

人话：

不直接改权重 W

而是学两个小矩阵 A、B

让

W' = W + A×B

特点：

参数只有原来的 <1%

对 multi-head attention 的 Q/K/V 很有效

现在大模型微调标配

👉 可以给每个客户一份 LoRA。

总结：LoRA 通过 W + A×B 的低秩适配，在极少参数下实现 fine tuning。

2️⃣ Prefix / Prompt Tuning

在输入侧加可学习 token

直觉：

[可学习前缀] + 真实句子


不改模型

只改“提示表示”

👉 适合 encoder 和 decoder 两类模型。

3️⃣ Adapter Tuning

每层中间插一个小网络

x → Frozen Transformer → 小adapter → 输出


只训 adapter

主权重冻结

4️⃣ IA3 / BitFit

只改 bias

只改缩放系数

👉 更极端的 PEFT。

🧠 一个例子帮你懂

你要做 NER 或 LLM SFT：

传统：

512 维 × 全量参数都算梯度

PEFT：

只训练：

LoRA weights

adapter layers

prefix tokens

👉 就能让模型适配：

医疗领域

对话风格

syntactic task


#### Fine-tuning 的问题

可能过拟合

会遗忘部分通用能力（catastrophic forgetting）
