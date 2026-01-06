# Pretraining

先让模型在海量文本上自学语言规律，再拿去做具体任务。

就像：小孩先大量听书、看故事 → 学会语法和词义

以后再去写作文、做阅读理解。

### Byte-Pair Encoding 

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
