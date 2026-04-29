# 🏥 Multimodal Clinical Intelligence System with Mission Control UI

> **An end-to-end multimodal AI system for medical image + clinical text disease classification with full XAI, RAG retrieval, and multi-agent synthesis, powered by a futuristic Streamlit dashboard.**

**Project:** Clinical AI Dashboard for Advanced Multimodal Analysis  
**Version:** 2.0 (Streamlit UI + Agents)  
**Last Updated:** April 29, 2026

---

## 🎯 Quick Start

### Run the Streamlit App (Live Interface)
```bash
cd /path/to/Trial_2
pip install -r requirements.txt
python3 -m streamlit run app.py --server.port 8505
# Open: http://localhost:8505
```

### Run Jupyter Notebooks (Experimental)
```bash
jupyter notebook
# Navigate to notebooks/ and open 01_data_exploration.ipynb through 07_full_pipeline_demo.ipynb
```

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
| **UI** | ❌ | **Streamlit with Mission Control theme** |

---

## 🎨 Streamlit Dashboard Features

### Mission Control Interface
The Streamlit app (`app.py`) provides a **futuristic "Mission Control" dashboard** with a professional, presentation-ready design:

✨ **UI Components:**
- **Hero Banner** — Dramatic intro with gradient glows and neon accents
- **Large Interactive Cards** — Oversized metrics, prediction displays, and controls
- **Scan Viewer** — Interactive Plotly image viewer with Grad-CAM opacity slider
- **Explainability Tab** — SHAP importance bars, modality contribution donut chart
- **Retrieval Tab** — Similar case gallery with similarity scores and rankings
- **Clinical Report Tab** — Agent-synthesized report with download options (TXT, MD, PDF)

### Workflow (Single Image + Single Query)
1. **Sidebar:** Upload medical scan (PNG/JPG) + enter clinical notes
2. **Run Analysis:** One-click pipeline execution
3. **Tabs:**
   - **Scan & Grad-CAM:** Original + heatmap overlay with adjustable opacity
   - **Explainability:** SHAP token importance + modality contribution analysis
   - **Retrieval (RAG):** Top-K similar cases from FAISS index
   - **Clinical Report:** Multi-agent synthesized diagnosis report
4. **Download:** Export report as TXT, MD, or PDF

### Technology Stack
- **Framework:** Streamlit 1.50.0
- **Visualizations:** Plotly (interactive charts), Matplotlib (Grad-CAM heatmaps)
- **Theme:** Dark glassmorphism with neon gradients (purple → cyan → green accents)
- **Performance:** @st.cache_resource for pipeline lazy-loading

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

## � Dependencies

**Core Machine Learning:**
- torch==2.8.0 (CPU or GPU)
- torchvision, torchaudio
- transformers==4.57.6 (HuggingFace)
- scikit-learn, pandas, numpy

**XAI & Visualization:**
- plotly==6.7.0
- matplotlib, PIL (Pillow)
- shap==0.49.1 (optional, auto-disabled if missing)
- captum==0.8.0 (for Integrated Gradients)

**Retrieval & Indexing:**
- faiss-cpu==1.13.0

**Agents & LLM Integration:**
- langchain
- requests (for API calls)

**Web Framework:**
- streamlit==1.50.0
- python-dotenv (for environment variables)

See `requirements.txt` for exact versions.

---

## ⚙️ Key Configuration Files

### `config.py`
Centralized settings for:
- API tokens (HuggingFace, Wandb, OpenAI, etc.)
- Model hyperparameters (learning rate, batch size, epochs)
- Data paths and splits
- XAI thresholds and parameters

**Before running anything, fill in your API keys in `config.py` line ~22:**
```python
HF_TOKEN = "hf_your_huggingface_token"
```

### `.env` (Optional)
Alternative to `config.py` for sensitive secrets:
```
HF_TOKEN=hf_xxx
WANDB_API_KEY=wandb_xxx
OPENAI_API_KEY=sk_xxx
```

---

## 🐛 Troubleshooting

### 1. SHAP ImportError
**Problem:** `No module named 'shap'`  
**Solution:** App continues without SHAP; install with `pip install shap` if needed.

### 2. Plotly Deprecation Warnings
**Problem:** "The keyword arguments have been deprecated..."  
**Solution:** Warnings are benign; UI still renders correctly. Uses modern `config={}` internally.

### 3. Port Already in Use
**Problem:** `Port 8505 is already in use`  
**Solution:** 
```bash
# Find & kill old Streamlit processes
pkill -f 'streamlit run'
# Or use a different port
python3 -m streamlit run app.py --server.port 8506
```

### 4. Out of Memory
**Problem:** `CUDA out of memory` or system RAM maxed  
**Solution:** 
- Use CPU only: `torch.device("cpu")`
- Reduce batch size in `config.py`
- Reduce FAISS index size

---

## 📞 Support & Contributing

**Questions?** Open an issue or contact the authors.

**Want to extend?** Key extension points:
- Add new XAI methods in `src/explainability/xai_engine.py`
- Custom agents in `src/agents/`
- New model architectures in `src/training/multimodal_predictor.py`

---
## 🎓 Citation

If you use this project in research or publications, please cite:

```bibtex
@software{trial2_2026,
  author = {AI Research Team},
  title = {Multimodal Clinical Intelligence System with XAI and RAG},
  year = {2026},
  url = {https://github.com/KRUSHNA453/Multiagent-clinical-shap}
}
```

---

## 👥 Authors & Acknowledgments

**Development Team:**  
- Lead Developer: Souta Palliashokkumar
- Advisors: Clinical AI Research Group

**Powered by:**
- HuggingFace (Transformers, Models Hub)
- Meta AI (PyTorch)
- Google (Grad-CAM, XAI research)
- SHAP team (Explainability)
- Streamlit team (Interactive dashboards)

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` file for details.

---
## �📚 References

- DenseNet: Huang et al. (2017). "Densely Connected Convolutional Networks." CVPR.
- CheXNet: Rajpurkar et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on CXR."
- BioClinicalBERT: Alsentzer et al. (2019). "Publicly Available Clinical BERT Embeddings." ACL.
- Grad-CAM: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
- SHAP: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NIPS.
- LIME: Ribeiro et al. (2016). "Why Should I Trust You?" KDD.
- MedPix Dataset: Ackerman et al. (2001). "Informatics in Radiology." RadioGraphics.
