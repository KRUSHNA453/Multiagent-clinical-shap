"""
xai_engine.py
=============
Unified Explainability Engine for the Multimodal Clinical Predictor.

Implements ALL four XAI methods as required:
  A) Grad-CAM   — visual heatmap on DenseNet-121 image branch
  B) SHAP       — token-level importance for Bio_ClinicalBERT text branch
  C) LIME       — superpixel importance map (image, supplementary)
  D) Integrated Gradients (Captum) — cross-modal contribution analysis

Clinical Motivation:
  XAI is non-negotiable in clinical AI — "black box" predictions cannot
  be trusted by physicians. Each method answers a different clinical question:
  - Grad-CAM:  "Which region of the scan drove this prediction?"
  - SHAP:      "Which clinical words were the strongest diagnostic signals?"
  - LIME:      "Would the prediction change if I masked this image region?"
  - IG:        "Did vision or text dominate this prediction?"
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# A) GRAD-CAM ENGINE
# =============================================================================
class GradCAMEngine:
    """
    Gradient-weighted Class Activation Mapping for DenseNet-121.

    Uses pytorch-grad-cam library (Jacob Gildenblat et al.):
    https://github.com/jacobgil/pytorch-grad-cam

    Clinical interpretation:
        Red/warm regions = image areas that most strongly activated the
        predicted class. In radiology, this should highlight lesions,
        masses, or pathological findings.
    """

    def __init__(self, model, target_layer_name: str = "features.norm5"):
        """
        Args:
            model: MultimodalClinicalPredictor instance (on eval device).
            target_layer_name: Named module in DenseNet to hook.
                'features.norm5' = final BatchNorm before GAP — best for DenseNet.
        """
        self.model = model
        self.model.eval()

        # Retrieve the target layer for Grad-CAM hook
        self.target_layer = self._get_layer(target_layer_name)
        self._activations = None
        self._gradients   = None

        # Register forward/backward hooks to capture activations and gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
        logger.info(f"GradCAMEngine initialized on layer: {target_layer_name}")

    def _get_layer(self, name: str):
        """Navigate nested module by dot-separated name."""
        module = self.model.image_encoder
        for part in name.split("."):
            module = getattr(module, part)
        return module

    def _save_activation(self, module, input, output):
        """Forward hook: saves feature map activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: saves gradients of the loss w.r.t. activations."""
        self._gradients = grad_output[0].detach()

    def generate_cam(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a batch of images.

        Args:
            images:        (B, 3, 224, 224) input images
            input_ids:     (B, seq_len) text tokens
            attention_mask:(B, seq_len) attention mask
            target_class:  Class index to explain (None = predicted class)
        Returns:
            cams: (B, 224, 224) normalized heatmaps, values in [0, 1]
        """
        self.model.train(False)
        images = images.requires_grad_(True)

        # Forward pass
        logits = self.model(images, input_ids, attention_mask)

        # Determine which class to explain
        if target_class is None:
            target_class = logits.argmax(dim=-1)  # Use predicted class

        # Backward pass: compute gradients of the target class score
        self.model.zero_grad()
        # Select the target class score for each sample in the batch
        if isinstance(target_class, torch.Tensor):
            loss = logits[range(len(logits)), target_class].sum()
        else:
            loss = logits[:, target_class].sum()
        loss.backward()

        # Pool gradients over spatial dimensions (H, W) → importance weight per channel
        # This is the α_k^c term in the original Grad-CAM paper
        pooled_grads = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weight activations by gradient importance
        weighted_activations = (self._activations * pooled_grads)       # (B, C, H, W)

        # Sum across channels → raw Grad-CAM map
        cam = weighted_activations.sum(dim=1)                           # (B, H, W)

        # Apply ReLU: only keep positive activations (features that support the class)
        cam = F.relu(cam)

        # Upsample to input image size (224x224)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, 224, 224)

        # Normalize to [0, 1] per sample
        cam = cam.detach().cpu().numpy()
        for i in range(len(cam)):
            cam_min, cam_max = cam[i].min(), cam[i].max()
            if cam_max - cam_min > 1e-8:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)
            else:
                cam[i] = np.zeros_like(cam[i])

        return cam  # (B, 224, 224) float32

    def overlay_cam_on_image(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            image:    (224, 224, 3) uint8 numpy array [0, 255]
            cam:      (224, 224)  float32 [0, 1] Grad-CAM heatmap
            alpha:    Transparency blend factor (0 = image only, 1 = cam only)
            colormap: Matplotlib colormap name (default 'jet' = red=hot, blue=cold)
        Returns:
            overlay: (224, 224, 3) uint8 blended image
        """
        # Apply colormap to CAM
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(cam)[:, :, :3]           # RGBA → RGB, float [0,1]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Normalize original image to [0, 255] uint8 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Blend: overlay = alpha * heatmap + (1-alpha) * image
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        return overlay

    def save_gradcam_figure(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        overlay: np.ndarray,
        title: str,
        save_path: str,
    ):
        """Save a side-by-side figure: [Original | Grad-CAM | Overlay]."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original_image)
        axes[0].set_title("Original Scan")
        axes[0].axis("off")

        im = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.suptitle(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Grad-CAM figure saved: {save_path}")


# =============================================================================
# B) SHAP ENGINE (Text Branch)
# =============================================================================
class SHAPEngine:
    """
    SHAP explainability for Bio_ClinicalBERT text branch.

    Uses shap.Explainer with the HuggingFace text masking strategy.
    Computes token-level Shapley values showing which clinical words
    most influenced the predicted disease class.

    Clinical interpretation:
        Positive SHAP value = token PUSHED model toward predicted diagnosis.
        Negative SHAP value = token PUSHED model AWAY from predicted diagnosis.
        Key clinical terms (symptoms, findings) should have high |SHAP|.
    """

    def __init__(self, model, tokenizer, device: torch.device = None, max_length: int = 256):
        """
        Args:
            model:      MultimodalClinicalPredictor instance.
            tokenizer:  Bio_ClinicalBERT tokenizer.
            device:     Torch device.
            max_length: BERT token max length.
        """
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        try:
            import shap
        except ModuleNotFoundError:
            shap = None
            logger.warning("SHAP is not installed; SHAP explanations will be disabled.")

        self._shap_available = shap is not None
        if not self._shap_available:
            self.explainer = None
            return

        logger.info("Initializing SHAP Text Explainer with pipeline masker...")

        # Build a text-classification pipeline for SHAP to wrap
        # SHAP's Text Masker handles BERT tokenization internally (mask = [MASK] token)
        self.explainer = shap.Explainer(
            self._shap_predict_fn,
            masker=shap.maskers.Text(self.tokenizer),
            algorithm="auto",
            output_names=None,   # Will be set after first call
        )

    def _shap_predict_fn(self, texts: List[str]) -> np.ndarray:
        """
        SHAP-compatible prediction function for text-only branch.

        NOTE: For SHAP we use text-only predictions (image branch set to zeros).
        This isolates the TEXT branch contribution, which is what we want for
        clinical word importance analysis.
        """
        self.model.eval()
        batch_size = 8  # Process in mini-batches to avoid OOM
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            encoding = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Zero-image: pass black images so we isolate text contribution
            B = len(batch)
            dummy_images = torch.zeros(B, 3, 224, 224, device=self.device)

            with torch.no_grad():
                logits = self.model(dummy_images, input_ids, attention_mask)
                probs  = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def explain(self, texts: List[str], max_evals: int = 100) -> object:
        """
        Compute SHAP values for a list of clinical text samples.

        Args:
            texts:     List of clinical text strings to explain.
            max_evals: Max SHAP evaluations (higher = more accurate, slower).
        Returns:
            shap_values: SHAP Explanation object.
        """
        if not getattr(self, "_shap_available", False):
            return None
        logger.info(f"Computing SHAP values for {len(texts)} text samples...")
        shap_values = self.explainer(texts, max_evals=max_evals)
        return shap_values

    @staticmethod
    def _extract_scores(shap_values, sample_idx: int, class_idx: int) -> np.ndarray:
        """
        Robustly extract the per-token SHAP score vector for one sample + class.

        shap.maskers.Text stores .values in several formats depending on the
        SHAP version and number of samples:

          - Python list of 2-D arrays:   v[i] → (n_tokens_i, n_classes)
          - numpy object-array (ragged):  v[i] → (n_tokens_i, n_classes)
              dtype=object because each sample has a different token count.
              np.asarray(v, dtype=float) FAILS on this — never do it.
          - numpy float array 3-D:        (n_samples, n_tokens, n_classes)
          - numpy float array 2-D:        (n_tokens, n_classes)  [single sample]

        Strategy: ALWAYS index by sample_idx first to obtain one sample's
        array, THEN cast to float and pick the class column.
        This avoids the ragged-conversion ValueError entirely.

        Always returns a *1-D* float64 array of length n_tokens.
        """
        v = shap_values.values

        # ── Step 1: extract the single-sample slice ────────────────────────────
        # Works for: plain list, numpy object-array (ragged), numpy float 3-D.
        # For a genuine 2-D float array the whole thing IS one sample already.
        if isinstance(v, (list, np.ndarray)) and (
            isinstance(v, list) or v.dtype == object or v.ndim == 3
        ):
            sample_v = np.asarray(v[sample_idx], dtype=float)
        else:
            # 2-D float numpy array — sample dim already squeezed away
            sample_v = np.asarray(v, dtype=float)

        # ── Step 2: pick the class column if values are (n_tokens, n_classes) ──
        if sample_v.ndim == 2:
            n_classes = sample_v.shape[1]
            col = min(class_idx, n_classes - 1)   # guard against out-of-range
            return sample_v[:, col]

        # Already 1-D (single-class or already flattened)
        return sample_v.ravel()

    def get_top_tokens(
        self,
        shap_values,
        sample_idx: int,
        class_idx: int,
        top_k: int = 15,
    ) -> List[Dict]:
        """
        Extract top-K most important tokens for a specific prediction.

        Args:
            shap_values: SHAP Explanation object (handles 1-D / 2-D / 3-D .values).
            sample_idx:  Which sample to extract from.
            class_idx:   Which class's SHAP values to use.
            top_k:       Number of top tokens to return.
        Returns:
            List of {'token': str, 'shap_score': float} dicts, sorted by |score|.
        """
        tokens = shap_values.data[sample_idx]
        scores = self._extract_scores(shap_values, sample_idx, class_idx)

        token_scores = [
            {"token": t, "shap_score": float(s)}
            for t, s in zip(tokens, scores)
            if t.strip() and t not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        # Sort by absolute importance (most impactful first)
        token_scores.sort(key=lambda x: abs(x["shap_score"]), reverse=True)
        return token_scores[:top_k]

    def plot_waterfall(
        self,
        shap_values,
        sample_idx: int,
        class_idx: int,
        class_name: str,
        save_path: str,
    ):
        """Save SHAP waterfall plot for a single sample + class."""
        import shap
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            # Standard 3-D indexing
            sv_slice = shap_values[sample_idx, :, class_idx]
        except (IndexError, TypeError):
            # Fallback: let SHAP index by sample only
            sv_slice = shap_values[sample_idx]
        shap.plots.waterfall(sv_slice, show=False)
        plt.title(f"SHAP Waterfall — Class: {class_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"SHAP waterfall plot saved: {save_path}")


# =============================================================================
# C) LIME ENGINE (Image Branch)
# =============================================================================
class LIMEEngine:
    """
    LIME (Local Interpretable Model-Agnostic Explanations) for images.

    Uses superpixel segmentation (SLIC) to identify image regions that
    are most influential for the prediction. Complements Grad-CAM by
    using a model-agnostic perturbation approach.

    Clinical interpretation:
        Green superpixels = regions that SUPPORT the predicted diagnosis.
        Red superpixels   = regions that CONTRADICT the prediction.
    """

    def __init__(self, model, device: torch.device = None):
        """
        Args:
            model:  MultimodalClinicalPredictor instance.
            device: Torch device.
        """
        from lime import lime_image
        self.model  = model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.explainer = lime_image.LimeImageExplainer()
        logger.info("LIMEEngine initialized.")

    def explain_image(
        self,
        image: np.ndarray,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: int = 100,
        num_features: int = 10,
        target_class: Optional[int] = None,
    ):
        """
        Generate LIME explanation for a single image.

        Args:
            image:          (224, 224, 3) numpy uint8 image
            input_ids:      (1, seq_len) text token IDs
            attention_mask: (1, seq_len) attention mask
            num_samples:    Number of perturbed images to generate
            num_features:   Number of superpixels to highlight
            target_class:   Class to explain (None = auto from prediction)
        Returns:
            explanation: LIME ImageExplanation object
        """
        import torchvision.transforms as T

        # Normalize transform to match training
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        def batch_predict(images_np: np.ndarray) -> np.ndarray:
            """Wrapper for LIME: takes (N, H, W, C) uint8, returns (N, num_classes)."""
            self.model.eval()
            batch_probs = []
            bs = 4  # Small batch for LIME perturbations
            for i in range(0, len(images_np), bs):
                batch_imgs = images_np[i : i + bs]
                tensors = torch.stack([
                    normalize(img.astype(np.uint8)) for img in batch_imgs
                ]).to(self.device)

                # Replicate text for each image in batch
                B = len(batch_imgs)
                ids  = input_ids.expand(B, -1).to(self.device)
                mask = attention_mask.expand(B, -1).to(self.device)

                with torch.no_grad():
                    logits = self.model(tensors, ids, mask)
                    probs  = F.softmax(logits, dim=-1)
                batch_probs.append(probs.cpu().numpy())
            return np.vstack(batch_probs)

        # Determine target class
        if target_class is None:
            from torchvision import transforms
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            img_t = t(image.astype(np.uint8)).unsqueeze(0).to(self.device)
            ids   = input_ids.to(self.device)
            mask  = attention_mask.to(self.device)
            with torch.no_grad():
                logits = self.model(img_t, ids, mask)
                target_class = logits.argmax(dim=-1).item()

        explanation = self.explainer.explain_instance(
            image.astype(np.double),
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
        )
        return explanation, target_class

    def get_lime_overlay(self, explanation, target_class: int, num_features: int = 10) -> Tuple:
        """
        Get the superpixel mask and overlay image from LIME explanation.

        Returns:
            mask:    (H, W) boolean array (True = important superpixel)
            overlay: (H, W, 3) uint8 image with colored superpixels
        """
        from skimage.segmentation import mark_boundaries
        temp_img, mask = explanation.get_image_and_mask(
            label=target_class,
            positive_only=False,
            num_features=num_features,
            hide_rest=False,
        )
        overlay = mark_boundaries(
            temp_img.astype(np.uint8), mask, color=(1, 1, 0), mode="thick"
        )
        return mask, (overlay * 255).astype(np.uint8)


# =============================================================================
# D) INTEGRATED GRADIENTS ENGINE (Modality Contribution Analysis)
# =============================================================================
class IntegratedGradientsEngine:
    """
    Integrated Gradients using Captum for cross-modal contribution analysis.

    Answers: "What fraction of the prediction came from IMAGE vs TEXT?"
    This is the MULTIMODAL-SPECIFIC metric required in the specification.

    Method: Computes attributions on the FUSED embedding (concatenated
    image + text embeddings), then sums over image-dims vs text-dims
    to get per-modality contribution scores.
    """

    def __init__(self, model):
        """
        Args:
            model: MultimodalClinicalPredictor instance.
        """
        from captum.attr import IntegratedGradients
        self.model = model
        # Wrap the model's forward to accept a single fused tensor
        # We insert a fusion-only forward path via a helper wrapper
        self.ig = IntegratedGradients(self._fusion_forward)
        self.img_dim  = model.image_encoder.embedding_dim   # 1024
        self.text_dim = model.text_encoder.embedding_dim    # 768
        logger.info("IntegratedGradientsEngine initialized.")

    def _fusion_forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """
        Captum-compatible forward: takes fused (B, 1792) and returns logits.
        Splits the embedding back into image/text parts before passing to fusion head.
        """
        img_part  = fused_embedding[:, :self.img_dim]
        text_part = fused_embedding[:, self.img_dim:]
        return self.model.fusion_head(img_part, text_part)

    def compute_attributions(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> Dict:
        """
        Compute Integrated Gradients attributions for the fusion head.

        Args:
            images, input_ids, attention_mask: Model inputs.
            target_class:  Class to attribute (None = predicted class).
            n_steps:       IG approximation steps (higher = more accurate).
        Returns:
            Dict with:
                'attributions':          (B, 1792) IG attribution values
                'image_attribution':     (B,) sum of image-feature attributions
                'text_attribution':      (B,) sum of text-feature attributions
                'dominant_modality':     List['image'|'text'] per sample
                'image_contribution_pct': (B,) percentage image vs total
        """
        self.model.eval()

        # Get fused embedding (not through fusion head — raw concat)
        with torch.no_grad():
            img_emb  = self.model.image_encoder(images)
            text_emb = self.model.text_encoder(input_ids, attention_mask)
        fused = torch.cat([img_emb, text_emb], dim=-1).requires_grad_(True)

        # Baseline: zero embedding (no information)
        baseline = torch.zeros_like(fused)

        # Determine target class
        if target_class is None:
            with torch.no_grad():
                logits = self._fusion_forward(fused.detach())
                target_class = logits.argmax(dim=-1).tolist()
                if isinstance(target_class, int):
                    target_class = [target_class] * len(images)

        # Build Captum-compatible target tensor
        # Captum requires: int (same class for whole batch) OR LongTensor (B,) per-sample
        if isinstance(target_class, int):
            captum_target = target_class
        else:
            captum_target = torch.tensor(target_class, dtype=torch.long, device=fused.device)

        # Compute Integrated Gradients
        attributions, delta = self.ig.attribute(
            fused,
            baselines=baseline,
            target=captum_target,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Split attributions back into image and text parts
        attr_np  = attributions.detach().cpu().numpy()
        img_attr = np.abs(attr_np[:, :self.img_dim]).sum(axis=1)   # (B,)
        txt_attr = np.abs(attr_np[:, self.img_dim:]).sum(axis=1)   # (B,)
        total    = img_attr + txt_attr + 1e-8

        img_pct  = img_attr / total * 100
        dominant = ["image" if p >= 50 else "text" for p in img_pct]

        return {
            "attributions":           attr_np,
            "image_attribution":      img_attr,
            "text_attribution":       txt_attr,
            "dominant_modality":      dominant,
            "image_contribution_pct": img_pct,
            "text_contribution_pct":  (1 - img_attr / total) * 100,
        }


# =============================================================================
# XAI SUMMARY TABLE BUILDER
# =============================================================================
def build_xai_summary_table(
    sample_ids: List[str],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    top_shap_tokens: List[List[Dict]],
    gradcam_summaries: List[str],
    dominant_modalities: List[str],
) -> "pd.DataFrame":
    """
    Build the XAI Summary Table required in Notebook 04.

    Columns:
        Sample_ID | True_Label | Predicted_Label | Confidence |
        Top_SHAP_Tokens | GradCAM_Region | Dominant_Modality

    Args:
        sample_ids:          List of image IDs.
        true_labels:         Ground truth class names.
        pred_labels:         Predicted class names.
        confidences:         Softmax confidence scores [0,1].
        top_shap_tokens:     List of top-SHAP token dicts per sample.
        gradcam_summaries:   Human-readable CAM region descriptions.
        dominant_modalities: 'image' or 'text' per sample.
    Returns:
        pandas DataFrame XAI summary table.
    """
    import pandas as pd

    def format_tokens(token_list: List[Dict]) -> str:
        if not token_list:
            return "N/A"
        return ", ".join([
            f"{t['token']} ({t['shap_score']:+.3f})"
            for t in token_list[:5]
        ])

    rows = []
    for i, sid in enumerate(sample_ids):
        rows.append({
            "Sample_ID":        sid,
            "True_Label":       true_labels[i],
            "Predicted_Label":  pred_labels[i],
            "Confidence":       f"{confidences[i]:.2%}",
            "Top_SHAP_Tokens":  format_tokens(top_shap_tokens[i] if i < len(top_shap_tokens) else []),
            "GradCAM_Region":   gradcam_summaries[i] if i < len(gradcam_summaries) else "N/A",
            "Dominant_Modality": dominant_modalities[i] if i < len(dominant_modalities) else "N/A",
        })
    return pd.DataFrame(rows)
