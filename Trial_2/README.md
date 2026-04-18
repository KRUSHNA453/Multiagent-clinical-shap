# Trial_2 — Multimodal Clinical Intelligence Pipeline
# README.md

# 🏥 Trial_2: Multimodal Clinical Intelligence Pipeline

> **An end-to-end multimodal AI system for medical image + clinical text disease classification with full XAI, RAG retrieval, and multi-agent synthesis.**

---

## 📋 Project Overview

Trial_2 is a significant upgrade over Trial_1, evolving from a **text-only** BioClinicalBERT pipeline to a **true multimodal system** that processes both radiology images and clinical notes simultaneously.

| Feature | Trial_1 | Trial_2 |
|---|---|---|
| Modalities | Text only (PMC-Patients) | **Images + Text** (MedPix) |
| Image Model | ❌ | **DenseNet-121** (ImageNet pretrained) |
| Text Model | BioClinicalBERT | **Bio_ClinicalBERT** (same + better integration) |
| XAI | SHAP only | **Grad-CAM + SHAP + LIME + Integrated Gradients** |
| Evaluation | Basic | **10+ metrics + dashboard** |
| Notebooks | 1 demo notebook | **7 structured experiment notebooks** |

---

## 📦 Dataset: MedPix

- **Source:** `Agent_code_Trial-2/Dataset/MedPix/`
- **Images:** 2,050 PNG files (CT + MRI scans), 224×224 pixels after preprocessing
- **Text:** Radiology captions + clinical case histories
- **Classes:** 10 grouped disease categories (derived from Topic.Category)

| Class ID | Category | Description |
|---|---|---|
| 0 | Neoplasm | All tumor types (benign, carcinoma, glial, metastatic) |
| 1 | Trauma | Traumatic injuries, fractures, sports medicine |
| 2 | Vascular | Vascular pathology, ischemia, infarction |
| 3 | Congenital | Structural and genetic birth anomalies |
| 4 | Infection | Bacterial, fungal, viral infections |
| 5 | Inflammatory | Non-infectious inflammation, autoimmune |
| 6 | Degenerative/Metabolic | Metabolic, endocrine, toxic disorders |
| 7 | Obstruction/Mechanical | Obstruction, stenosis, surgical complications |
| 8 | Idiopathic/Unknown | Unsure, idiopathic, differential diagnosis |
| 9 | Clinical Sign/Other | Clinical findings, cysts, NOS |

---

## 🚀 Installation

```bash
# 1. Clone / navigate to Trial_2/
cd Trial_2

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Configure API keys (see below)
```

---

## ⚠️ API Key Configuration

Open `config.py` and fill in your keys:

```python
# Line ~22 — HuggingFace token (required)
HF_TOKEN = "hf_your_token_here"
```

| Key | Purpose | Get it at |
|---|---|---|
| `HF_TOKEN` | Download BioClinicalBERT weights + Qwen LLM API | https://huggingface.co/settings/tokens |
| `WANDB_API_KEY` | Experiment tracking (optional) | https://wandb.ai/authorize |

---

## 📒 Notebook Execution Order

Run notebooks **in order** from `notebooks/`:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | Profile MedPix dataset, class distribution, image grid |
| 02 | `02_preprocessing.ipynb` | Build MultimodalDataset, CLAHE augmentation, train/val/test splits |
| 03 | `03_model_training.ipynb` | Train DenseNet+BERT fusion model, metrics dashboard |
| 04 | `04_explainability_xai.ipynb` | Grad-CAM, SHAP, LIME, Integrated Gradients |
| 05 | `05_rag_retrieval.ipynb` | Build FAISS index, MAP@K, NDCG retrieval metrics |
| 06 | `06_multiagent_synthesis.ipynb` | LangChain agents: Explanation + Validation + Summary |
| 07 | `07_full_pipeline_demo.ipynb` | End-to-end demo: image + text → full clinical report |

---

## 📊 Evaluation Metrics Produced

### Classification (Notebook 03)
- Accuracy, Precision, Recall, F1-Score (per-class + macro/weighted)
- AUC-ROC (one-vs-rest, plotted)
- Cohen's Kappa, Matthews Correlation Coefficient (MCC)
- Normalized Confusion Matrix

### XAI Quality (Notebook 04)
- SHAP Feature Consistency Score
- Faithfulness Score (prediction drop when top SHAP features masked)
- Modality Contribution Analysis (image % vs text %)
- Cross-modal agreement score

### Retrieval (Notebook 05)
- MAP@K (K=1,3,5)
- Recall@K
- NDCG@K

All metrics → `outputs/evaluation_report.json`

---

## 🗂️ Project Structure

```
Trial_2/
├── config.py              ← ALL API keys + hyperparameters
├── requirements.txt
├── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_explainability_xai.ipynb
│   ├── 05_rag_retrieval.ipynb
│   ├── 06_multiagent_synthesis.ipynb
│   └── 07_full_pipeline_demo.ipynb
├── src/
│   ├── training/multimodal_predictor.py   ← DenseNet + ClinicalBERT model
│   ├── explainability/xai_engine.py       ← Grad-CAM, SHAP, LIME, IG
│   ├── retrieval/faiss_retrieval.py       ← FAISS vector search
│   ├── agents/explanation_agent.py        ← Agent 1 (Qwen LLM)
│   ├── agents/validation_agent.py         ← Agent 2 (Qwen LLM)
│   ├── agents/summary_agent.py            ← Agent 3 (Qwen LLM)
│   └── pipeline/orchestrator.py           ← End-to-end wiring
├── models/                ← Saved checkpoints
├── data/                  ← Processed splits + FAISS index
└── outputs/               ← GradCAM images, SHAP plots, eval reports
    ├── gradcam/
    ├── shap/
    └── lime/
```

---

## 📸 Sample Outputs *(placeholder — populate after running pipelines)*

- Grad-CAM overlays: `outputs/gradcam/`
- SHAP waterfall plots: `outputs/shap/`
- Evaluation dashboard: Last cell of Notebook 07
- Final clinical report: Rendered in Notebook 07

---

## 📚 References

- DenseNet: Huang et al. (2017). "Densely Connected Convolutional Networks." CVPR.
- CheXNet: Rajpurkar et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on CXR."
- BioClinicalBERT: Alsentzer et al. (2019). "Publicly Available Clinical BERT Embeddings." ACL.
- Grad-CAM: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
- SHAP: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NIPS.
- LIME: Ribeiro et al. (2016). "Why Should I Trust You?" KDD.
- MedPix Dataset: Ackerman et al. (2001). "Informatics in Radiology." RadioGraphics.
