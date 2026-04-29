"""
config.py — Centralized Configuration for Trial_2 Multimodal Clinical Pipeline
================================================================================
ALL API keys, model names, paths, and hyperparameters are defined HERE.
Never hardcode secrets in notebooks. Import from this file instead.

Usage in any notebook or module:
    import sys; sys.path.insert(0, '..')   # or adjust path
    from config import HF_TOKEN, MEDPIX_IMAGE_DIR, ...
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  API KEY REQUIRED
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ Service     : HuggingFace Hub                                            │
# │ Variable    : HF_TOKEN                                                   │
# │ Where       : .env file (HUGGINGFACEHUB_API_TOKEN)                        │
# │ Required for: Downloading Bio_ClinicalBERT weights, LLM Inference API   │
# └──────────────────────────────────────────────────────────────────────────┘
HF_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HUGGINGFACEHUB_API_TOKEN: str = HF_TOKEN   # Alias for compatibility

# LLM for multi-agent synthesis
HF_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"

# ⚠️  API KEY REQUIRED (Optional — for experiment tracking)
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ Service     : Weights & Biases (W&B)                                     │
# │ Variable    : WANDB_API_KEY                                               │
# │ Where       : config.py → line below                                     │
# │ How to get  : https://wandb.ai/authorize                                  │
# │ Required for: Experiment tracking (optional — set to None to disable)    │
# └──────────────────────────────────────────────────────────────────────────┘
WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")   # Leave empty to disable

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Text encoder: Bio_ClinicalBERT from HuggingFace
# Clinical domain-tuned BERT trained on 2M clinical notes (MIMIC-III)
TEXT_ENCODER_MODEL: str = "emilyalsentzer/Bio_ClinicalBERT"

# Image encoder: DenseNet-121 pretrained on ImageNet (torchvision)
# Widely validated on chest X-rays (CheXNet), works generically on all CT/MRI
IMAGE_ENCODER: str = "densenet121"

# LLM for multi-agent synthesis: Qwen 2.5 72B via HF Inference API
# Chosen for strong medical reasoning and free Inference API tier
LLM_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"
LLM_MODE: str = "api"   # 'api' = HF Inference API | 'local' = local pipeline

# Sentence encoder for FAISS retrieval (biomedical domain)
RETRIEVAL_ENCODER: str = "pritamdeka/S-PubMedBert-MS-MARCO"

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PATHS  (relative to Trial_2/ root)
# ─────────────────────────────────────────────────────────────────────────────

# Root of the Trial_2 project (auto-detected from this file's location)
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# MedPix dataset root (one level up from Trial_2/)
MEDPIX_ROOT: Path = PROJECT_ROOT.parent / "Dataset" / "MedPix"

# Specific MedPix files
MEDPIX_IMAGE_DIR: Path    = MEDPIX_ROOT / "images" / "images"   # Actual PNGs live in images/images/ (nested)
MEDPIX_DESCRIPTIONS: Path = MEDPIX_ROOT / "Descriptions.json"
MEDPIX_CASE_TOPIC: Path   = MEDPIX_ROOT / "Case_topic.json"
MEDPIX_SPLITS_DIR: Path   = MEDPIX_ROOT / "splitted_dataset"

# Data outputs within Trial_2/
DATA_DIR: Path          = PROJECT_ROOT / "data"
PROCESSED_DIR: Path     = DATA_DIR / "processed"

# Model checkpoints
MODELS_DIR: Path        = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path   = MODELS_DIR / "best_multimodal_model.pt"
FAISS_INDEX_PATH: Path  = DATA_DIR / "faiss_index.bin"
FAISS_META_PATH: Path   = DATA_DIR / "faiss_meta.pkl"

# Outputs
OUTPUTS_DIR: Path       = PROJECT_ROOT / "outputs"
GRADCAM_DIR: Path       = OUTPUTS_DIR / "gradcam"
SHAP_DIR: Path          = OUTPUTS_DIR / "shap"
LIME_DIR: Path          = OUTPUTS_DIR / "lime"
EVAL_REPORT_PATH: Path  = OUTPUTS_DIR / "evaluation_report.json"
DATASET_SUMMARY_PATH: Path = DATA_DIR / "dataset_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION — 10 GROUPED DISEASE CATEGORIES
# Maps raw Topic.Category strings → integer class IDs
# Clinical rationale: grouped by pathophysiology, not anatomy
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_GROUPS: dict = {
    # 0: Neoplasm
    "Neoplasm, benign":              0,
    "Neoplasm, carcinoma":           0,
    "Neoplasm, glial":               0,
    "Neoplasm, metastatic":          0,
    "Neoplasm, NOS":                 0,
    "Neoplasm, malignant (NOS)":     0,
    "Neoplasm, non-glial":           0,

    # 1: Vascular_Trauma
    "Trauma":                        1,
    "Sports Medicine":               1,
    "Vascular":                      1,
    "Hypoxic or Ischemic":           1,
    "Infarction and/or Necrosis":    1,

    # 2: Infection_Inflammatory
    "Infection, bacteria":           2,
    "Infection, fungi":              2,
    "Infection, virus":              2,
    "Infection, parasite":           2,
    "Infection, NOS":                2,
    "Inflammatory, non-infectious":  2,
    "Inflammatory, NOS":             2,
    "Autoimmune":                    2,
    "Ophthalmology":                 2,

    # 3: Other
    "Congenital, malformation":      3,
    "Congenital, genetic":           3,
    "Degenerative":                  3,
    "Metabolic (see also Toxic)":    3,
    "Toxic":                         3,
    "Endocrine":                     3,
    "Obstruction or Stenosis":       3,
    "Iatrogenic or Surgical (complications)": 3,
    "Physiology":                    3,
    "Diverticulum":                  3,
    "Idiopathic or Unknown":         3,
    "Unsure":                        3,
    "Differential Diagnosis":        3,
    "Miscellaneous":                 3,

    # 4: Clinical Sign
    "Clinical Exam Finding or Sign": 4,
    "Cyst, benign":                  4,
}

# Human-readable class names — used in plots, reports, confusion matrices
CLASS_NAMES: list = [
    "Neoplasm",               # 0
    "Vascular_Trauma",        # 1
    "Infection_Inflammatory", # 2
    "Other",                  # 3
    "Clinical Sign"           # 4
]
NUM_CLASSES: int = len(CLASS_NAMES)

# Default class for categories NOT in CATEGORY_GROUPS above
DEFAULT_CLASS: int = 4   # Clinical Sign

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE: int           = 8       # Reduce to 4 on low-VRAM GPUs
EPOCHS: int               = 25      # Default training epochs
LEARNING_RATE: float      = 2e-5    # AdamW learning rate (standard for fine-tuning BERT)
WEIGHT_DECAY: float       = 0.01    # L2 regularization
DROPOUT_RATE: float       = 0.3     # Dropout in fusion head
VAL_SPLIT: float          = 0.15    # 15% validation split
TEST_SPLIT: float         = 0.15    # 15% test split
MAX_TEXT_LENGTH: int      = 256     # BERT token max length (256 for low VRAM, 512 for high)
IMAGE_SIZE: tuple         = (224, 224)  # DenseNet-121 standard input size
FUSION_HIDDEN_DIM: int    = 512     # Hidden dim of the MLP fusion head
RANDOM_SEED: int          = 42      # Reproducibility seed

# ─────────────────────────────────────────────────────────────────────────────
# XAI CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER: str = "features.norm5"  # Last BatchNorm before classifier in DenseNet-121
NUM_SHAP_SAMPLES: int     = 50       # SHAP background samples (reduce for speed)
NUM_LIME_SUPERPIXELS: int = 50       # LIME superpixel segments
NUM_IG_STEPS: int         = 50        # Integrated Gradients approximation steps

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FAISS_TOP_K: int     = 5    # Top-K similar cases to retrieve
FAISS_MAX_RECORDS: int = 500  # Max records to index (subset for speed)

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION STATS (ImageNet baseline — standard for medical imaging transfer)
# Paper: He et al. (2016) DenseNet for chest X-rays validated with these stats
# ─────────────────────────────────────────────────────────────────────────────
IMG_MEAN: tuple = (0.485, 0.456, 0.406)  # ImageNet RGB mean
IMG_STD: tuple  = (0.229, 0.224, 0.225)  # ImageNet RGB std

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-CREATE OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
for _dir in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, GRADCAM_DIR, SHAP_DIR, LIME_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
