# 🏥 Multiagent-Clinical-SHAP: Multimodal Clinical Intelligence Pipeline

> **An end-to-end multimodal AI system for medical image + clinical text disease classification with full XAI, RAG retrieval, and multi-agent synthesis.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Project Overview

This repository hosts **Trial_2** of our Agentic AI course project, evolving from a text-only pipeline to a **true multimodal system** that processes both radiology images and clinical notes simultaneously.

| Feature | Trial_1 | Trial_2 (Current) |
|---|---|---|
| **Modalities** | Text only | **Images + Text** (MedPix) |
| **Image Model**| ❌ | **DenseNet-121** (ImageNet pretrained) |
| **Text Model** | BioClinicalBERT | **Bio_ClinicalBERT** |
| **XAI Methods**| SHAP only | **Grad-CAM + SHAP + LIME + Integrated Gradients** |
| **Synthesis**  | Single LLM Call | **Multi-Agent RAG Synthesis** (Explanation, Validation, Summary) |

---

## 🚀 Quick Start & Installation

```bash
# 1. Clone the repository
git clone https://github.com/KRUSHNA453/Multiagent-clinical-shap.git
cd Multiagent-clinical-shap/Trial_2

# 2. Install dependencies
pip install -r requirements.txt
```

### ⚠️ API Key Configuration

Open `Trial_2/config.py` and fill in your keys:

```python
# Line ~24 — HuggingFace token (required)
HF_TOKEN = "hf_your_token_here"
```

---

## 🗂️ Repository Structure

The core implementation resides in the `Trial_2/` directory:

```
Multiagent-clinical-shap/
├── Dataset/                   ← Local dataset (ignored via .gitignore)
└── Trial_2/                   ← Main code and pipeline
    ├── notebooks/             ← 01 to 07 execution notebooks
    ├── src/                   ← Core Python modules (training, agents, XAI, RAG)
    ├── config.py              ← API keys & Hyperparameters
    └── requirements.txt       ← Dependencies
```

Head over to the [Trial_2 Directory](./Trial_2/) to explore the full pipeline notebooks, or read the detailed [Trial_2 README](./Trial_2/README.md) for deeper technical specifications.

---

## 📒 Pipeline Notebooks (in `Trial_2/notebooks/`)

1. `01_data_exploration.ipynb` - Profile MedPix dataset & distributions
2. `02_preprocessing.ipynb` - MultimodalDataset, augmentations, & splits
3. `03_model_training.ipynb` - Train DenseNet+BERT fusion model
4. `04_explainability_xai.ipynb` - Grad-CAM, SHAP, LIME, Integrated Gradients
5. `05_rag_retrieval.ipynb` - Build FAISS index & run vector search
6. `06_multiagent_synthesis.ipynb` - LangChain reasoning agents
7. `07_full_pipeline_demo.ipynb` - End-to-end execution

---

## 🎓 Academic Context
This project was developed for the **AML-509: Agentic AI and GAN** coursework at SRM. Contains the implementation of multimodal clinical systems utilizing advanced State-of-the-Art Deep Learning architectures and LLM multi-agent orchestrations.
