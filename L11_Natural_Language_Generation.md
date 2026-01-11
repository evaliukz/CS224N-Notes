# Natural Language Generation

Natural Language Processing = Natural Language Understanding + Natural Language Generation

NLU（Natural Language Understanding）：
👉 人 → 机器（理解）

NLG（Natural Language Generation）：
👉 机器 → 人（表达）

### 如何找到most likely string？

这是NLG / LLM 的数学核心。“Most likely string” = 在给定条件下，联合概率最大的 token 序列。

但真实世界中我们几乎不可能精确找到它，只能用近似搜索（decoding）。

#### 1️⃣ 数学上什么叫 most likely string？

给定输入（prompt / context） 
x，模型定义的是：

P(y∣x)=t=1∏T​P(yt​∣y<t​,x)

Most likely string 定义为：

y∗=argymax​P(y∣x)  在所有可能的字符串中，找到概率乘积最大的那一条

#### 2️⃣ 为什么不能“直接算出来”？

🚫 原因一：搜索空间爆炸

词表大小：~50,000

序列长度：100 tokens

可能字符串数：

50,000
100

👉 比宇宙原子数还大

🚫 原因二：局部最优 ≠ 全局最优

一个 token 概率大，不代表整句概率最大：

"我 喜欢 吃"
vs
"我 非常 喜欢 吃"


某一步 greedy 选错，后面全盘皆输

#### 3️⃣ Greedy decoding 为什么不是 most likely？
Greedy 定义

每一步选：yt​=argmaxP(yt​∣y<t​)

问题

这是在最大化：tmax​P(yt​∣⋅)

❌ 但我们要的是：

maxt∏​P(yt​∣⋅)

📌 局部最大 ≠ 全局最大

#### 4️⃣ 那现实中是怎么“找 most likely 的”？

答案：近似搜索（Decoding Algorithms）

#### 5️⃣ Beam Search：最接近“most likely string”的方法

核心思想

同时保留 K 条最有希望的前缀

每一步扩展，再剪枝

算法直觉

Step 1: 保留 top K tokens

Step 2: 每条扩展 → K×V

Step 3: 按 log probability 排序 → 留 K 条

优化目标： argymax​t∑​logP(yt​∣y<t​)

✅ 是 MAP（Maximum A Posteriori）近似

Beam Search 的问题

容易：重复，平庸，官腔，在开放文本（chat）中效果反而不好

👉 所以 ChatGPT 不用 beam search

#### 6️⃣ Sampling：放弃 most likely，换“像人类”
Top-k Sampling

只在概率最高的 k 个 token 中采样：yt​∼TopK(P)

Top-p（Nucleus Sampling）

选累计概率 ≥ p 的最小集合：i∈S ∑​P(i)≥p

📌 当前 LLM 标配

Temperature（温度）

Temperature（温度）是 NLG / LLM 里最容易被“会用但不真懂”的参数。Temperature 控制的是：模型在“不太可能的词”上，敢不敢冒险。

低温：保守、确定、像考试标准答案

高温：发散、创造、像头脑风暴

1️⃣ 从物理“温度”类比（最好记）

在物理里：

温度低 → 分子运动慢 → 状态稳定

温度高 → 分子运动快 → 状态更随机

在语言生成里：

温度低 → token 选择集中在高概率

温度高 → token 选择更分散

📌 完全是同一个“随机性”的隐喻

Temperature 影响的是“你看到的模型人格”

低温 → 冷静、克制、工程师

高温 → 热情、诗性、发散

❌ Temperature 不会：

让模型更聪明

增加新知识

减少 hallucination

✅ 它只是：

改变“从已知分布里怎么抽样”

现代 LLM 通常三者一起用：

| 参数          | 控制什么      |
| ----------- | --------- |
| Temperature | 概率分布“平不平” |
| Top-k       | 能选的候选个数   |
| Top-p       | 覆盖多少概率质量  |


#### 7️⃣ 关键结论（非常重要）

LLM 实际上“刻意不找 most likely string”

原因：

Most likely 往往：

啰嗦

重复

无聊

人类偏好的是：

合理但不最优

多样但不胡编

📌 ChatGPT = controlled stochastic generator



总结：

| 角度     | 结论                                            |
| ------ | --------------------------------------------- |
| 理论     | Most likely string = argmax joint probability |
| 计算     | NP-hard（指数搜索）                                 |
| 近似     | Beam search                                   |
| 实践     | Sampling > MAP                                |
| LLM 目标 | **人类偏好 ≠ 最大概率**                               |


# Model-based metrics 
用一个“模型”来判断另一个模型生成的文本好不好，而不是靠词面重合度。

### 为什么需要 Model-based metrics？
传统指标的问题

| 指标     | 核心缺陷         |
| ------ | ------------ |
| BLEU   | 只看 n-gram 重合 |
| ROUGE  | 偏向摘要长度       |
| METEOR | 仍是词级匹配       |

📌 同义表达会被误判为“不好”：

Reference:
The revenue dropped significantly.

Generation:
There was a sharp decline in revenue.

👉 BLEU 很低，但人类觉得 完全正确

### Model-based metrics 的核心思想

不用“字像不像”，而是让模型判断：

语义是否一致？

事实是否正确？

是否自然？

是否符合人类偏好？

### 三大类 Model-based metrics

1️⃣ Embedding-based（向量相似度）
代表：BERTScore

做法

用预训练模型（如 BERT）生成 token embeddings

计算 cosine similarity

取最大匹配平均

📌 看语义，不看字面

优点

同义句友好

快

缺点

不判断事实真假

不懂“是否胡编”

2️⃣ Learned Scoring Models（训练出来的打分模型）
代表：BLEURT

做法

用大量 人类打分数据 训练一个回归模型

输入：(reference, generation)

输出：质量分数

📌 直接学“人类怎么打分”

优点

与人工评价相关性高

比 BLEU 稳定

缺点

域外泛化差

训练成本高

3️⃣ LLM-as-a-Judge（当前主流）
核心思想

让大模型当评委

示例 Prompt：

Please rate the following answer from 1 to 5
based on factual correctness and clarity.


📌 ChatGPT / GPT-4 / Claude 都常被用作评估器
