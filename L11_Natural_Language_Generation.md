# Natural Language Generation

Natural Language Processing = Natural Language Understanding + Natural Language Generation

NLU（Natural Language Understanding）：
👉 人 → 机器（理解）

NLG（Natural Language Generation）：
👉 机器 → 人（表达）

### 如何找到most likely string？

这是NLG / LLM 的数学核心。“Most likely string” = 在给定条件下，联合概率最大的 token 序列。

但真实世界中我们几乎不可能精确找到它，只能用近似搜索（decoding）。

1️⃣ 数学上什么叫 most likely string？

给定输入（prompt / context） 
x，模型定义的是：

P(y∣x)=t=1∏T​P(yt​∣y<t​,x)

Most likely string 定义为：

y∗=argymax​P(y∣x)  在所有可能的字符串中，找到概率乘积最大的那一条

2️⃣ 为什么不能“直接算出来”？

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

3️⃣ Greedy decoding 为什么不是 most likely？
Greedy 定义

每一步选：yt​=argmaxP(yt​∣y<t​)

问题

这是在最大化：tmax​P(yt​∣⋅)

❌ 但我们要的是：

maxt∏​P(yt​∣⋅)

📌 局部最大 ≠ 全局最大

4️⃣ 那现实中是怎么“找 most likely 的”？

答案：近似搜索（Decoding Algorithms）

5️⃣ Beam Search：最接近“most likely string”的方法

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

容易：

重复

平庸

官腔

在开放文本（chat）中效果反而不好

👉 所以 ChatGPT 不用 beam search

6️⃣ Sampling：放弃 most likely，换“像人类”
Top-k Sampling

只在概率最高的 k 个 token 中采样：yt​∼TopK(P)

Top-p（Nucleus Sampling）

选累计概率 ≥ p 的最小集合：i∈S ∑​P(i)≥p

📌 当前 LLM 标配

7️⃣ 关键结论（非常重要）

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

8️⃣ 直觉类比（很好记）

找 most likely string 就像：

写一篇“概率最高的作文”

结果通常是：
“众所周知……随着社会的发展……具有重要意义……”

总结：

| 角度     | 结论                                            |
| ------ | --------------------------------------------- |
| 理论     | Most likely string = argmax joint probability |
| 计算     | NP-hard（指数搜索）                                 |
| 近似     | Beam search                                   |
| 实践     | Sampling > MAP                                |
| LLM 目标 | **人类偏好 ≠ 最大概率**                               |
