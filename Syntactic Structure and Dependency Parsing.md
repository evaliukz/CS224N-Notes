# Syntactic Structure（句法结构）& Dependency Parsing（依存解析）

The quick brown fox jumps over the lazy dog.
句法结构让我们理解：
“fox” 是主语
“jumps” 是谓语动词
“dog” 是被修饰对象
“over” 表示动作的空间关系

## What's syntactic structure?（什么是句法结构）
句子不是扁平 token list 而具有 层级（constituency）结构 和 依存（dependency）结构
| 类型                          | 描述                                  | 例子                                  |
| --------------------------- | ----------------------------------- | ----------------------------------- |
| Constituency Tree **短语成分树** | NP/VP/PP 层级组成                       | (NP (DT The) (NN fox))              |
| Dependency Tree **依存树**     | 用 directed arc 表示 head ←→ dependent | nsubj(jumps, fox), dobj(eat, apple) |

VP: verb phrase


现代 NLP 更多依赖 dependency tree —— 更贴近语义关系

## Dependency Parsing（依存解析）
目标：给每个词找到 head，并标注关系标签（subject/object/compound ）
She enjoys playing tennis
   │       │     │
nsubj  xcomp   dobj

## Dependency Tree 是怎么构建出来的？
把一句话当成一堆词排成队，我们要做的事情其实只有一件：
找出“谁管谁”“谁依赖谁”，然后把它们连成一棵树。

这就像你在办公室分配任务：
谁是老板？谁是干活的人？谁是给别人帮忙的人？

数据结构
构建树靠 栈 Stack + 队列 Buffer + 收集的连接 Arcs
Stack：已经处理到手里的词（像放在桌子上）
Buffer：还没处理的词（排队等着）
Arcs：你建立的“谁→谁”依赖关系

三种动作（Transitions）
| 动作        | 大白话解释                                        | 例子               |
| --------- | -------------------------------------------- | ---------------- |
| SHIFT     | 把队列里的下一个词抓到手（放进Stack）                        | Buffer头部 → Stack |
| LEFT-ARC  | 说：Stack里上面的词 **是** Buffer词的“下属”，然后丢掉这个下属     | （buffer当老板）      |
| RIGHT-ARC | 说：Stack上面的词 **是** Buffer词的老板，然后丢掉 buffer的这个词 | （stack当老板）       |


最终 goal：得到完整依存树


初始状态：
Stack: [ROOT]
Buffer: [I, love, NLP]
Arcs: []

🧩 Step 1 — SHIFT

抓第一个词到手里

Stack: [ROOT, I]
Buffer: [love, NLP]

🧩 Step 2 — SHIFT

继续抓

Stack: [ROOT, I, love]
Buffer: [NLP]

🧩 Step 3 — RIGHT-ARC

我们知道 love 才是句子里的动词老板
“I” 是主语，属于 love → 所以连：

love → I    (nsubj)


连完 “I” 就没用了 → 从 Stack 弹出 I

Stack: [ROOT, love]
Buffer: [NLP]
Arcs: [(love, I)]

🧩 Step 4 — SHIFT

把 NLP 抓到桌子上来

Stack: [ROOT, love, NLP]
Buffer: []

🧩 Step 5 — LEFT-ARC 或 RIGHT-ARC？

问自己一句“谁管谁？”
👉 love 是动作
👉 NLP 是动作作用的对象（宾语）

所以应该：

love → NLP   (dobj)


于是做：RIGHT-ARC

Stack: [ROOT, love]
Arcs: [(love, I), (love, NLP)]

🎉 Step 6 — 最后 ROOT 当总老板
ROOT → love   (root)

🎉 最终构建的依存树
      love
     /   \
   I     NLP



## Neural Parsing（神经化依存解析）
传统 transition parser 手工设计特征（POS, shape, capitalization …）
CS224N讲述：用 MLP 自动学习
输入 → stack top, buffer top 的词向量
输出 → softmax 预测下一步动作
训练数据：带金标准依存树的 Treebank（如 Penn Treebank）

使用两个指标：
| 指标  | 描述                                                     |
| --- | ------------------------------------------------------ |
| UAS | Unlabeled Attachment Score（预测 head 是否正确）               |
| LAS | Labeled Attachment Score（head + dependency label 是否正确） |



