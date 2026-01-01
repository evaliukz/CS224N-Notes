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
什么是 Neural Dependency Parsing？

用神经网络（神经网络分类器）来决定构建依存树的动作，从而自动解析句法关系。

传统 parsing 依赖 手工特征（POS tag组合、是否大写、句子位置…）
问题：
❌ 特征难设计
❌ 无法泛化
❌ 实际输入噪声大效果差

神经解析 = 不再靠人写规则，而是 网络学会语言结构本身
→ 这是现代 NLP parser 能达到接近人类水平的关键'

### Neural Dependency Parsing 基本结构
无论哪种版本（Arc-Standard, Arc-Eager, Graph-based），它本质上都有 3 个组件：
词向量 / contextual embedding  -->  状态表示  -->  分类器预测下一步动作

🧩 Step 1 — 输入表示（Embedding）

每个词被映射为向量：

传统：GloVe / word2vec

更高级：ELMo、BERT（contextual embedding → 效果更强）

也可能拼接：word embedding + POS embedding + character embedding

🪢 Step 2 — Parser State Representation（解析状态向量化）

Transition parser 需要决定下一步做什么动作 → 它必须“看一眼当前状态”

状态包括：

Stack 顶部几个词

Buffer 队列前几个词

这些词的 embedding 与 POS

tree 部分结构（可选）

举例：

Stack: [ROOT, love]
Buffer: [NLP, today]


模型会取 —— Stack top（love）、Buffer head（NLP）等 embedding 拼成一个特征向量：

h=f([xstack[−1]​,xstack[−2]​,xbuffer[0]​,...])

通常会用：
✔ MLP
✔ 或 BiLSTM 先对整句编码 → 再取索引向量

🤖 Step 3 — 神经网络预测动作

最终模型就是一个分类器：

输入：状态向量 h
输出：概率分布：哪一个 action 应该执行？
| 动作               | 含义                         |
| ---------------- | -------------------------- |
| SHIFT            | 把 Buffer 词移进 Stack         |
| LEFT-ARC(label)  | 建立 stack.top ← buffer.head |
| RIGHT-ARC(label) | 建立 stack.top → buffer.head |

（带 label = nsubj / dobj / compound …）

网络输出示例概率：

p(a∣h)=softmax(Wh+b)

选最大概率的动作 → 执行 → 状态更新
→ 重复直到 Buffer 空且 Stack 只剩 ROOT

🎓 Step 4 — 模型训练

训练数据：带人工依存树的 Treebank（如 Penn Treebank）

把依存树转成 gold transition sequence
（即正确动作序列）

Loss：L=−t∑​logp(at​∣ht​)

SGD / Adam 优化
多 epoch 学习





使用两个指标：
| 指标  | 描述                                                     |
| --- | ------------------------------------------------------ |
| UAS | Unlabeled Attachment Score（预测 head 是否正确）               |
| LAS | Labeled Attachment Score（head + dependency label 是否正确） |



