"""
pipeline/orchestrator.py
=========================
End-to-end Multimodal Clinical Intelligence Pipeline Orchestrator.

Wires ALL 5 stages together for single-sample inference:
  Stage 1: Preprocessing (image transform + text tokenization)
  Stage 2: Multimodal Prediction (DenseNet + ClinicalBERT fusion)
  Stage 3: XAI (Grad-CAM + SHAP + Integrated Gradients)
  Stage 4: RAG Retrieval (FAISS top-5 similar cases)
  Stage 5: Multi-Agent Synthesis (Explanation → Validation → Summary)

This is the main entry point for Notebook 07 (Full Pipeline Demo).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class MultimodalClinicalOrchestrator:
    """
    Single-entry-point orchestrator for the multimodal clinical AI pipeline.

    Usage:
        orchestrator = MultimodalClinicalOrchestrator(config_module)
        report = orchestrator.process_case(image_path, clinical_text)
    """

    def __init__(self, config, device: torch.device = None):
        """
        Args:
            config: The config module (import config; pass it here).
            device: Torch device. Auto-detects CUDA if not specified.
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Orchestrator using device: {self.device}")

        logger.info("=== Initializing Multimodal Clinical Pipeline ===")
        self._init_preprocessing()
        self._init_model()
        self._init_xai()
        self._init_retriever()
        self._init_agents()
        logger.info("=== Pipeline Ready ===")

    def _init_preprocessing(self):
        """Set up image transforms and text tokenizer."""
        # Medical image preprocessing pipeline:
        # 1. Resize to 224x224 (DenseNet requirement)
        # 2. Random augmentations (test time: center crop only)
        # 3. Normalize with ImageNet stats (validated for transfer learning)
        self.image_transform = T.Compose([
            T.Resize((self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1])),
            T.ToTensor(),
            T.Normalize(mean=self.config.IMG_MEAN, std=self.config.IMG_STD),
        ])

        logger.info(f"Loading tokenizer: {self.config.TEXT_ENCODER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.TEXT_ENCODER_MODEL,
            token=self.config.HF_TOKEN if self.config.HF_TOKEN != "YOUR_HF_TOKEN_HERE" else None,
        )
        logger.info("Preprocessing initialized.")

    def _init_model(self):
        """Load the multimodal predictor (from checkpoint if available)."""
        # Import here to avoid top-level circular deps
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.training.multimodal_predictor import MultimodalClinicalPredictor

        self.model = MultimodalClinicalPredictor(
            num_classes=self.config.NUM_CLASSES,
            text_model_name=self.config.TEXT_ENCODER_MODEL,
            hidden_dim=self.config.FUSION_HIDDEN_DIM,
            dropout_rate=self.config.DROPOUT_RATE,
        )

        checkpoint = self.config.BEST_MODEL_PATH
        if checkpoint.exists():
            logger.info(f"Loading checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Checkpoint loaded successfully.")
        else:
            logger.warning(
                f"No checkpoint found at {checkpoint}. "
                "Using pretrained ImageNet/BERT weights (untrained fusion head)."
            )

        self.model = self.model.to(self.device).eval()
        logger.info(f"Model loaded on {self.device}.")

    def _init_xai(self):
        """Initialize XAI engines (Grad-CAM, SHAP, IG)."""
        from src.explainability.xai_engine import GradCAMEngine, SHAPEngine, IntegratedGradientsEngine

        self.gradcam_engine = GradCAMEngine(
            self.model,
            target_layer_name=self.config.GRADCAM_TARGET_LAYER,
        )
        self.shap_engine = SHAPEngine(
            self.model,
            self.tokenizer,
            device=self.device,
            max_length=self.config.MAX_TEXT_LENGTH,
        )
        self.ig_engine = IntegratedGradientsEngine(self.model)
        logger.info("XAI engines initialized.")

    def _init_retriever(self):
        """Load FAISS index if available."""
        from src.retrieval.faiss_retrieval import MultimodalFAISSRetriever

        self.retriever = MultimodalFAISSRetriever(embedding_dim=1792)
        try:
            self.retriever.load(
                str(self.config.FAISS_INDEX_PATH),
                str(self.config.FAISS_META_PATH),
            )
            self.has_retriever = True
        except FileNotFoundError:
            logger.warning("FAISS index not found. Stage 4 (RAG) will be skipped.")
            self.has_retriever = False

    def _init_agents(self):
        """Initialize the three LangChain agents."""
        from src.agents.explanation_agent import ExplanationAgent
        from src.agents.validation_agent  import ValidationAgent
        from src.agents.summary_agent     import SummaryAgent

        # Set HuggingFace API token in environment
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.config.HUGGINGFACEHUB_API_TOKEN

        # Agents now pull configuration from config.py directly
        self.exp_agent = ExplanationAgent()
        self.val_agent = ValidationAgent()
        self.sum_agent = SummaryAgent()
        logger.info("Multi-agent system initialized.")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def process_case(
        self,
        image_path: str,
        clinical_text: str,
        image_modality: str = "CT",
        body_region: str = "chest",
        return_intermediates: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full 5-stage pipeline on a single patient case.

        Args:
            image_path:       Path to medical scan image (PNG/JPG).
            clinical_text:    Patient clinical notes / radiology caption.
            image_modality:   'CT' or 'MR'.
            body_region:      Anatomical body region (for agent context).
            return_intermediates: If True, include all intermediate outputs.
        Returns:
            Dict with keys: prediction, confidence, gradcam, shap, ig,
                            retrieved_cases, reasoning, validation, report,
                            final_report (string).
        """
        results: Dict[str, Any] = {}

        # ── STAGE 1: PREPROCESSING ───────────────────────────────────────────
        logger.info("\n=== STAGE 1: PREPROCESSING ===")
        image_tensor, input_ids, attention_mask = self._preprocess(image_path, clinical_text)
        results["preprocessed"] = True

        # ── STAGE 2: PREDICTION ──────────────────────────────────────────────
        logger.info("\n=== STAGE 2: MULTIMODAL PREDICTION ===")
        pred_idx, confidence, pred_label = self._predict(image_tensor, input_ids, attention_mask)
        results["pred_class_idx"] = pred_idx
        results["pred_label"]     = pred_label
        results["confidence"]     = confidence
        logger.info(f"Predicted: {pred_label} ({confidence:.1%})")

        # ── STAGE 3: XAI ────────────────────────────────────────────────────
        logger.info("\n=== STAGE 3: EXPLAINABILITY ===")
        xai_results = self._run_xai(image_tensor, input_ids, attention_mask, clinical_text, pred_idx)
        results.update(xai_results)

        # ── STAGE 4: RAG RETRIEVAL ───────────────────────────────────────────
        logger.info("\n=== STAGE 4: RAG RETRIEVAL ===")
        if self.has_retriever:
            retrieved = self.retriever.retrieve(
                self.model,
                image_tensor,
                input_ids,
                attention_mask,
                top_k=self.config.FAISS_TOP_K,
                device=self.device,
            )[0]  # Single sample → first element
        else:
            retrieved = []
            logger.info("Retrieval skipped (no index).")
        results["retrieved_cases"] = retrieved

        # ── STAGE 5: MULTI-AGENT SYNTHESIS ───────────────────────────────────
        logger.info("\n=== STAGE 5: MULTI-AGENT SYNTHESIS ===")

        # Agent 1: Explanation
        top_shap_tokens = xai_results.get("top_shap_tokens", [])
        gradcam_desc    = xai_results.get("gradcam_description", "focal activation detected")
        dominant_mod    = xai_results.get("dominant_modality", "image")

        reasoning = self.exp_agent.generate_reasoning(
            diagnosis=pred_label,
            confidence=confidence,
            shap_tokens=top_shap_tokens,
            gradcam_region=gradcam_desc,
        )

        # Agent 2: Validation
        validation = self.val_agent.validate_prediction(
            predicted_label=pred_label,
            retrieved_cases=retrieved,
        )

        # Agent 3: Summary Report
        final_report = self.sum_agent.generate_report(
            patient_synopsis=clinical_text[:400],
            diagnosis=pred_label,
            confidence=confidence,
            clinical_reasoning=reasoning,
            validation_report=validation,
            image_modality=image_modality,
        )

        results["reasoning"]     = reasoning
        results["validation"]    = validation
        results["final_report"]  = final_report

        logger.info("\n=== PIPELINE COMPLETE ===")
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE STAGE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, image_path: str, text: str) -> Tuple:
        """Load image + tokenize text → tensors on device."""
        # Image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}. Using black image.")
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        # Text
        enc = self.tokenizer(
            text,
            max_length=self.config.MAX_TEXT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)       # (1, seq_len)
        attention_mask = enc["attention_mask"].to(self.device)  # (1, seq_len)

        return img_tensor, input_ids, attention_mask

    def _predict(self, img: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> Tuple:
        """Run forward pass and return (pred_idx, confidence, label_name)."""
        pred_indices, probs = self.model.predict(img, ids, mask)
        pred_idx   = pred_indices[0].item()
        confidence = probs[0, pred_idx].item()
        pred_label = self.config.CLASS_NAMES[pred_idx]
        return pred_idx, confidence, pred_label

    def _run_xai(
        self,
        img: torch.Tensor,
        ids: torch.Tensor,
        mask: torch.Tensor,
        text: str,
        pred_idx: int,
    ) -> Dict:
        """Run Grad-CAM, SHAP, and Integrated Gradients."""
        xai = {}

        # Grad-CAM
        try:
            cam = self.gradcam_engine.generate_cam(img, ids, mask, target_class=pred_idx)
            xai["gradcam_map"]  = cam[0]  # (224, 224)
            # Create human-readable description of hotspot location
            hot_y, hot_x = np.unravel_index(cam[0].argmax(), cam[0].shape)
            region = self._describe_gradcam_region(hot_x, hot_y)
            xai["gradcam_description"] = region
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
            xai["gradcam_map"] = np.zeros((224, 224))
            xai["gradcam_description"] = "Grad-CAM unavailable"

        # SHAP (text branch)
        try:
            shap_vals = self.shap_engine.explain([text], max_evals=50)
            top_tokens = self.shap_engine.get_top_tokens(shap_vals, 0, pred_idx, top_k=10)
            xai["shap_values"]    = shap_vals
            xai["top_shap_tokens"] = top_tokens
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            xai["top_shap_tokens"] = []

        # Integrated Gradients (modality contribution)
        try:
            ig_results = self.ig_engine.compute_attributions(
                img, ids, mask, target_class=pred_idx, n_steps=30
            )
            xai["dominant_modality"]      = ig_results["dominant_modality"][0]
            xai["image_contribution_pct"] = float(ig_results["image_contribution_pct"][0])
            xai["text_contribution_pct"]  = float(ig_results["text_contribution_pct"][0])
        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")
            xai["dominant_modality"] = "image"

        return xai

    @staticmethod
    def _describe_gradcam_region(x: int, y: int, img_size: int = 224) -> str:
        """Convert pixel coordinates to quadrant description."""
        h_label = "upper" if y < img_size // 2 else "lower"
        w_label = "left" if x < img_size // 2 else "right"
        return f"{h_label}-{w_label} quadrant of the scan"
