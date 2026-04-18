"""
multimodal_predictor.py
========================
Dual-branch multimodal neural network for clinical image + text classification.

Architecture:
  IMAGE BRANCH  : Pretrained DenseNet-121 (torchvision) → 1024-dim embedding
  TEXT BRANCH   : Bio_ClinicalBERT [CLS] token          → 768-dim embedding
  FUSION HEAD   : Concat [1024+768=1792] → Linear(512) → ReLU → Dropout → Linear(num_classes)

Clinical Motivation:
  DenseNet-121 was chosen because it is the backbone of CheXNet (Rajpurkar et al. 2017),
  the landmark model for chest X-ray diagnosis, and has been validated broadly across
  CT/MRI modalities in transfer learning settings.
  Bio_ClinicalBERT was trained on MIMIC-III clinical notes, making it the gold standard
  text encoder for patient history and radiology caption understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """
    DenseNet-121 image encoder.

    Loads ImageNet-pretrained DenseNet-121, removes the final classifier layer,
    and returns a 1024-dimensional feature embedding for each image.
    The dense connectivity pattern of DenseNet makes it particularly effective
    for medical images where low-level texture features (lesion boundaries,
    calcifications) are clinically important.
    """

    def __init__(self, pretrained: bool = True, freeze_base: bool = False):
        """
        Args:
            pretrained:   Load ImageNet weights (True = transfer learning).
            freeze_base:  Freeze all DenseNet weights except final block (for fine-tuning).
        """
        super().__init__()
        # Load pretrained DenseNet-121
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove the original 1000-class classifier — we use the feature maps directly
        # DenseNet-121 feature dim = 1024 (after global average pooling)
        self.features = densenet.features
        self.embedding_dim = 1024

        if freeze_base:
            # Clinical transfer learning strategy: freeze early layers, fine-tune later blocks
            # This preserves low-level edge/texture features from ImageNet pre-training
            for name, param in self.features.named_parameters():
                if "denseblock4" not in name and "norm5" not in name:
                    param.requires_grad = False
            logger.info("ImageEncoder: Froze all DenseNet layers except denseblock4 and norm5.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, 224, 224)
        Returns:
            embedding: (B, 1024) global average pooled feature vector
        """
        # Extract dense feature maps
        features = self.features(x)              # (B, 1024, 7, 7) for 224x224 input
        # Apply ReLU (DenseNet convention before pooling)
        features = F.relu(features, inplace=False)  # inplace=False required for Grad-CAM backward hooks
        # Global average pooling: spatially aggregate the feature maps
        # Clinical rationale: GAP encourages the model to look at the WHOLE image
        # rather than one region, which improves localizability for Grad-CAM
        embedding = F.adaptive_avg_pool2d(features, (1, 1))  # (B, 1024, 1, 1)
        embedding = torch.flatten(embedding, 1)               # (B, 1024)
        return embedding


class TextEncoder(nn.Module):
    """
    Bio_ClinicalBERT text encoder.

    Encodes clinical text (radiology captions + patient history) using
    Bio_ClinicalBERT (Emily Alsentzer et al., 2019), which was fine-tuned on
    2M de-identified clinical notes from MIMIC-III.

    Returns the [CLS] token embedding (768-dim), which BERT is designed to use
    as a summary representation of the entire input sequence.
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        dropout_rate: float = 0.1,
        freeze_base: bool = False,
    ):
        """
        Args:
            model_name:   HuggingFace model ID.
            dropout_rate: Applied to [CLS] embedding before fusion.
            freeze_base:  Freeze all BERT layers (only train fusion head).
        """
        super().__init__()
        logger.info(f"Loading text encoder: {model_name}")
        config = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        # Load BERT body only (no classification head — we do fusion separately)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.embedding_dim = self.bert.config.hidden_size  # 768 for BERT-base

        self.dropout = nn.Dropout(dropout_rate)

        if freeze_base:
            # Freeze all but the last 2 transformer layers for parameter-efficient training
            # Clinical context: early BERT layers capture general syntax (useful),
            # later layers capture domain-specific semantics (need fine-tuning)
            modules_to_freeze = [self.bert.embeddings, *self.bert.encoder.layer[:-2]]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
            logger.info("TextEncoder: Froze all BERT layers except last 2.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len) tokenized text IDs
            attention_mask: (B, seq_len) 1 for real tokens, 0 for padding
        Returns:
            cls_embedding: (B, 768) — [CLS] summary representation
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Extract [CLS] token (index 0) — the sentence-level summary embedding
        # BERT is pre-trained with a [CLS] classification objective (NSP)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return self.dropout(cls_embedding)


class FusionHead(nn.Module):
    """
    MLP Fusion Head that combines image and text embeddings.

    Architecture:
        Concat[image_emb, text_emb] → Linear(1792→512) → LayerNorm → ReLU
        → Dropout → Linear(512→num_classes)

    Clinical Rationale:
        Two-layer MLP allows the model to learn non-linear combinations of
        visual and textual features. The first layer reduces dimensionality
        (feature compression), while the second layer performs classification.
        LayerNorm stabilizes training when combining embeddings of different scales.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        fused_dim = image_dim + text_dim  # 1024 + 768 = 1792

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),   # Dimensionality reduction
            nn.LayerNorm(hidden_dim),            # Normalize fused representation
            nn.ReLU(inplace=False),              # inplace=False required for Grad-CAM backward hooks
            nn.Dropout(dropout_rate),            # Regularization
            nn.Linear(hidden_dim, num_classes),  # Final classification logits
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_emb: (B, 1024) image features
            text_emb:  (B, 768)  text features
        Returns:
            logits: (B, num_classes) raw class scores (no softmax — use CrossEntropyLoss)
        """
        # Concatenate both modalities along the feature dimension
        fused = torch.cat([image_emb, text_emb], dim=-1)  # (B, 1792)
        return self.fusion(fused)


class MultimodalClinicalPredictor(nn.Module):
    """
    Full Multimodal Clinical Predictor for MedPix disease classification.

    Combines:
      - DenseNet-121 (ImageNet pretrained) for radiology image understanding
      - Bio_ClinicalBERT for clinical text understanding
      - 2-layer MLP fusion head for joint prediction

    Supports:
      - Forward pass for training (returns logits)
      - predict() for inference (returns class index + probabilities)
      - get_image_embedding() / get_text_embedding() for FAISS/XAI use
    """

    def __init__(
        self,
        num_classes: int,
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        freeze_image_base: bool = False,
        freeze_text_base: bool = False,
    ):
        """
        Args:
            num_classes:        Number of disease categories (10 for MedPix).
            text_model_name:    HuggingFace model ID for text encoder.
            hidden_dim:         Fusion MLP hidden layer size.
            dropout_rate:       Dropout probability.
            freeze_image_base:  Freeze DenseNet feature layers.
            freeze_text_base:   Freeze BERT encoder layers.
        """
        super().__init__()
        self.num_classes = num_classes

        # Initialize both encoders
        self.image_encoder = ImageEncoder(
            pretrained=True, freeze_base=freeze_image_base
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            dropout_rate=dropout_rate,
            freeze_base=freeze_text_base,
        )

        # Fusion head
        self.fusion_head = FusionHead(
            image_dim=self.image_encoder.embedding_dim,   # 1024
            text_dim=self.text_encoder.embedding_dim,     # 768
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

        # Track parameter count for logging
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"MultimodalClinicalPredictor initialized: "
            f"{total_params:,} total params | {trainable_params:,} trainable"
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass returning logits.

        Args:
            images:         (B, 3, 224, 224) normalized image tensors
            input_ids:      (B, seq_len) BERT token IDs
            attention_mask: (B, seq_len) attention mask
        Returns:
            logits: (B, num_classes)
        """
        image_emb = self.image_encoder(images)
        text_emb  = self.text_encoder(input_ids, attention_mask)
        logits    = self.fusion_head(image_emb, text_emb)
        return logits

    def get_image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Returns raw image embedding from DenseNet (for FAISS/XAI)."""
        with torch.no_grad():
            return self.image_encoder(images)

    def get_text_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns raw text [CLS] embedding from BERT (for FAISS/XAI)."""
        with torch.no_grad():
            return self.text_encoder(input_ids, attention_mask)

    def get_fused_embedding(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns concatenated [image_emb | text_emb] for FAISS indexing."""
        img_emb  = self.image_encoder(images)
        txt_emb  = self.text_encoder(input_ids, attention_mask)
        return torch.cat([img_emb, txt_emb], dim=-1)  # (B, 1792)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        Inference-mode prediction.

        Returns:
            pred_indices: (B,) predicted class indices
            probabilities: (B, num_classes) softmax probabilities
        """
        self.eval()
        logits = self.forward(images, input_ids, attention_mask)
        probs  = F.softmax(logits, dim=-1)
        preds  = torch.argmax(probs, dim=-1)
        return preds, probs


class MultimodalDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MedPix multimodal data.

    Each sample returns:
        image_tensor:   (3, 224, 224) preprocessed and augmented image
        input_ids:      (seq_len,) BERT token IDs
        attention_mask: (seq_len,) BERT attention mask
        label:          int class index (0-9)
        sample_id:      string image ID (for tracking)
    """

    def __init__(
        self,
        df,                          # pandas DataFrame with columns: image_path, text, label, image_id
        image_transform=None,        # torchvision transform pipeline
        tokenizer=None,              # HuggingFace tokenizer
        max_length: int = 256,
        text_col: str = "text",
        label_col: str = "label",
        image_col: str = "image_path",
        id_col: str = "image_id",
        is_inference: bool = False,  # Skip labels during inference
    ):
        from PIL import Image
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col
        self.image_col = image_col
        self.id_col = id_col
        self.is_inference = is_inference
        self._Image = Image

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Image Loading & Transform ─────────────────────────────────────
        img_path = str(row[self.image_col])
        try:
            image = self._Image.open(img_path).convert("RGB")  # Ensure 3-channel
        except Exception as e:
            # Graceful fallback: return black image if file is missing/corrupt
            logger.warning(f"Could not load image {img_path}: {e}. Using black image.")
            image = self._Image.new("RGB", (224, 224), (0, 0, 0))

        if self.image_transform:
            image_tensor = self.image_transform(image)
        else:
            import torchvision.transforms as T
            image_tensor = T.ToTensor()(image)

        # ── Text Tokenization ────────────────────────────────────────────
        text = str(row[self.text_col]) if row[self.text_col] else ""
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = encoding["input_ids"].squeeze(0)       # Remove batch dim
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            # Dummy tensors if no tokenizer provided
            input_ids      = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # ── Build Sample Dict ────────────────────────────────────────────
        sample = {
            "image":          image_tensor,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image_id":       str(row[self.id_col]),
        }

        if not self.is_inference and self.label_col in self.df.columns:
            sample["label"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return sample
