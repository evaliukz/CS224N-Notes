# Lecture 3: Named Entity Recognition（命名实体识别）

PER（人名）
LOC（地点）
ORG（组织)
MISC（其他）

输出不是简单分类，而是 序列标注任务（sequence tagging）

常用标注格式：BIO / IOB2
（Begin, Inside, Outside）

Steve lives in New York City
B-PER  O    O  B-LOC I-LOC I-LOC


NER 模型方法演进路线
| 时代       | 模型                              | 特点                          |
| -------- | ------------------------------- | --------------------------- |
| 传统 ML    | HMM / CRF / Feature-based SVM   | 依靠人工特征工程                    |
| RNN 模型   | **BiLSTM**                      | 能捕获序列上下文                    |
| 深度 +结构联合 | **BiLSTM + CRF** ⭐（CS224N重点）    | 解决标签依赖 & 全局最优解              |
| 预训练大模型时代 | **BERT + CRF / BERT Fine-tune** | 最强效果，用 contextual embedding |


NER 是一个序列标注任务，目标是识别实体并分类。
经典 SOTA 架构是 BiLSTM-CRF：BiLSTM 抽取上下文 → CRF 做标签依赖建模 → 用 Viterbi 找最优标签序列。
当前最佳实践是 BERT Fine-tuning for token classification（often + CRF）。


---

## 🧬 Why NER?
NER enables downstream NLP tasks:
- Knowledge graph construction
- Relation extraction
- Financial / medical document mining
- Chatbots & QA systems
- Search systems & user intent analysis

---

## 🧱 Modeling Approaches

### 🔹 Traditional ML (Pre-Neural)
- HMM, CRF, SVM
- Heavy feature engineering (regex, gazetteers)

### 🔹 Neural Models (Modern)
| Model | Idea |
|-------|------|
| BiLSTM | captures contextual sequence info |
| **BiLSTM + CRF** ⭐ | adds structured prediction (CS224N重点) |
| **BERT / Large LM Fine-tuning** | contextual embedding → highest accuracy |

---

## 🤖 BiLSTM-CRF Architecture (Core)

Pipeline:
1️⃣ 输入层
token → 词向量 embedding
可用：GloVe / word2vec / ELMo / BERT

2️⃣ BiLSTM 抽取序列上下文特征：
得到每个词的 contextual hidden state（捕获左右语境）
3️⃣ CRF layer enforces global tag dependency
4️⃣ Final prediction uses **Viterbi algorithm** CRF 提供全局评分

Why CRF?
- solves label dependency violation (e.g., I-PER cannot start without B-PER)
如果只用 softmax（逐 token 分类），会出现：
I-PER 出现在没有 B-PER 的前面
I-ORG 跟在人名后面
→ 违反标签依赖规律
