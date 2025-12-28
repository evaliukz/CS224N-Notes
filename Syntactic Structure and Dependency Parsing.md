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

现代 NLP 更多依赖 dependency tree —— 更贴近语义关系

## Dependency Parsing（依存解析）
目标：给每个词找到 head，并标注关系标签（subject/object/compound ）
She enjoys playing tennis
   │       │     │
nsubj  xcomp   dobj

## Transition-based Dependency Parsing
数据结构
解析使用三个组件：
Stack:   存放已处理部分
Buffer:  待处理词
Arcs:    已建立的依存关系


初始状态：
Stack: [ROOT]
Buffer: [I, love, NLP]
Arcs: []

三种动作（Transitions）
| 动作        | 作用                                   |
| --------- | ------------------------------------ |
| SHIFT     | 将 buffer 头部移入 stack                  |
| LEFT-ARC  | 建立 arc：stack top ← buffer head，并 pop |
| RIGHT-ARC | 建立 arc：stack top → buffer head，并 pop |

最终 goal：得到完整依存树

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


