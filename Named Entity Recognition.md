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
1️⃣ Token → Embedding (GloVe / word2vec / BERT embedding)  
2️⃣ BiLSTM encodes context:
\[
h_t = \text{BiLSTM}(x_1 ... x_n)
\]
3️⃣ CRF layer enforces global tag dependency:
\[
s(y|h) = \sum_t (W h_t)_{y_t} + \sum_t A_{y_{t-1}, y_t}
\]
4️⃣ Final prediction uses **Viterbi algorithm** to maximize:
\[
y^* = \arg\max_y s(y|h)
\]

Why CRF?
- solves label dependency violation (e.g., I-PER cannot start without B-PER)

---

## 📊 Evaluation

NER uses **Precision / Recall / F1**  
- Evaluation is **span-level**
- Example: LOC “New York City” must be fully matched → otherwise incorrect

---

## 🧲 End-to-End Example

Input:

