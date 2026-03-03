# FineTuning-Gemma-using-LoRA

Parameter-efficient fine-tuning of **Google Gemma 2B** using **LoRA (Low-Rank Adaptation)** on the **Databricks Dolly 15k** dataset.

This project demonstrates how to efficiently adapt a large language model using **KerasNLP in Google Colab**, without updating all model parameters.

---

## 📌 Project Overview

Large Language Models (LLMs) are expensive to fine-tune due to their massive parameter sizes.

This project implements:

- ✅ Gemma 2B model loading using KerasNLP
- ✅ Dataset preprocessing (Dolly 15k)
- ✅ Prompt formatting for instruction tuning
- ✅ Parameter-efficient fine-tuning using LoRA
- ✅ Colab-compatible implementation

---

## 🧠 What is LoRA?

LoRA (Low-Rank Adaptation) is a **Parameter-Efficient Fine-Tuning (PEFT)** technique.

Instead of updating the full weight matrix \( W \) of a neural network layer:

\[
W \in \mathbb{R}^{d \times k}
\]

LoRA freezes the original weights and learns a **low-rank update**:

\[
W' = W + \Delta W
\]

Where:

\[
\Delta W = A \cdot B
\]

- \( A \in \mathbb{R}^{d \times r} \)
- \( B \in \mathbb{R}^{r \times k} \)
- \( r \ll d, k \)

This drastically reduces the number of trainable parameters.

---

### 📉 Why LoRA?

- 🔹 Reduces trainable parameters by >90%
- 🔹 Lower GPU/TPU memory usage
- 🔹 Faster training
- 🔹 Stores only small adapter weights
- 🔹 Keeps original pretrained model intact

---

### 📊 LoRA Architecture

![LoRA Diagram](images/lora.png)

The pretrained weights remain frozen while only the low-rank matrices are trained.

---

## 📂 Dataset

**Databricks Dolly 15k**

- 15,000 instruction-response pairs
- Used for instruction tuning
- Downloaded as `.jsonl` file in notebook

Prompt format used:

Instruction:
<instruction>

Response:
<response>


---

## ⚙️ Model Used

- `gemma_2b_en`
- Loaded using:
```python
keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
