"""
faiss_retrieval.py
==================
FAISS-based retrieval module for the Multimodal Clinical Pipeline.

Builds a vector index over training set FUSED embeddings (image + text),
enabling finding of the K most similar historical cases for any new patient.

Clinical Motivation:
    RAG (Retrieval-Augmented Generation) is critical in clinical AI:
    "This prediction resembles 5 past patients who all had X diagnosis"
    provides physicians with empirical evidence beyond the model's score.

Pipeline:
    1. Encode all training samples → fused embeddings (1792-dim)
    2. L2-normalize → unit spheres (converts to cosine similarity space)
    3. Build FAISS IndexFlatIP (inner product = cosine similarity after normalization)
    4. At inference: encode new sample → search top-K nearest neighbors
"""

import os
import pickle
import logging
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MultimodalFAISSRetriever:
    """
    FAISS retrieval engine for multimodal (image + text) clinical embeddings.

    Index type: IndexFlatIP (exact inner product search on L2-normalized vectors)
    → Equivalent to cosine similarity, which is preferred for semantic embeddings.

    Usage flow:
        retriever = MultimodalFAISSRetriever()
        retriever.build_index(model, dataloader, metadata)
        retriever.save(index_path, meta_path)
        # Later:
        retriever.load(index_path, meta_path)
        results = retriever.retrieve(model, images, input_ids, attention_mask, top_k=5)
    """

    def __init__(self, embedding_dim: int = 1792):
        """
        Args:
            embedding_dim: Dimension of fused embeddings (image 1024 + text 768 = 1792).
        """
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []   # Stores {image_id, label, label_name, text_snippet}
        logger.info(f"MultimodalFAISSRetriever, embedding_dim={embedding_dim}")

    def build_index(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        metadata_list: List[Dict],
        device: torch.device = None,
        max_records: int = 500,
    ):
        """
        Encode all samples in dataloader and build FAISS index.

        Args:
            model:          MultimodalClinicalPredictor (loads both branches).
            dataloader:     DataLoader yielding {image, input_ids, attention_mask, label, image_id}.
            metadata_list:  Parallel list of metadata dicts for each sample.
            device:         Torch device.
            max_records:    Maximum number of records to index (for memory efficiency).
        """
        from tqdm import tqdm
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        all_embeddings = []
        self.metadata  = []
        count = 0

        logger.info(f"Building FAISS index (max {max_records} records)...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding for FAISS")):
                if count >= max_records:
                    break

                images         = batch["image"].to(device)
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Get fused embedding (image + text concatenated)
                img_emb  = model.image_encoder(images)
                txt_emb  = model.text_encoder(input_ids, attention_mask)
                fused_emb = torch.cat([img_emb, txt_emb], dim=-1)  # (B, 1792)

                # L2-normalize: convert to unit vectors for cosine similarity
                fused_norm = F.normalize(fused_emb, p=2, dim=-1)
                all_embeddings.append(fused_norm.cpu().numpy())

                # Store metadata for each sample in batch
                batch_size = images.shape[0]
                for i in range(batch_size):
                    if count + i < len(metadata_list):
                        self.metadata.append(metadata_list[count + i])
                    else:
                        self.metadata.append({
                            "image_id": batch["image_id"][i],
                            "label_name": "Unknown",
                        })
                count += batch_size

        # Stack all embeddings
        embeddings_matrix = np.vstack(all_embeddings).astype("float32")
        # Trim to max_records if needed
        embeddings_matrix = embeddings_matrix[:max_records]
        self.metadata     = self.metadata[:max_records]

        logger.info(f"Embedding matrix shape: {embeddings_matrix.shape}")

        # Build FAISS IndexFlatIP (exact, inner product)
        # Using inner product on L2-normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_matrix)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def save(self, index_path: str, meta_path: str):
        """
        Save FAISS index and metadata to disk.

        Args:
            index_path: Path to save .bin FAISS index.
            meta_path:  Path to save .pkl metadata list.
        """
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"FAISS index saved to {index_path}")
        logger.info(f"Metadata saved to {meta_path}")

    def load(self, index_path: str, meta_path: str):
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to .bin FAISS index file.
            meta_path:  Path to .pkl metadata file.
        Raises:
            FileNotFoundError: If either file does not exist.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS metadata not found: {meta_path}")

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, "
                    f"{len(self.metadata)} metadata records.")

    def retrieve(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 5,
        device: torch.device = None,
    ) -> List[List[Dict]]:
        """
        Retrieve top-K most similar historical cases for each query sample.

        Args:
            model:          MultimodalClinicalPredictor.
            images:         (B, 3, 224, 224) query images.
            input_ids:      (B, seq_len) text tokens.
            attention_mask: (B, seq_len) attention mask.
            top_k:          Number of neighbors to retrieve.
            device:         Torch device.
        Returns:
            List of K result dicts per query sample:
            [{"rank":1, "similarity":0.95, "image_id":"MPX...", "label_name":"Neoplasm", ...}]
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build_index() or load() first.")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        with torch.no_grad():
            img_emb  = model.image_encoder(images.to(device))
            txt_emb  = model.text_encoder(input_ids.to(device), attention_mask.to(device))
            fused    = torch.cat([img_emb, txt_emb], dim=-1)
            # L2-normalize query (same space as indexed vectors)
            query    = F.normalize(fused, p=2, dim=-1).cpu().numpy().astype("float32")

        # Search FAISS index
        similarities, indices = self.index.search(query, top_k)  # (B, K), (B, K)

        results = []
        for b in range(len(images)):
            case_results = []
            for rank, (sim, idx) in enumerate(zip(similarities[b], indices[b])):
                if idx == -1:  # FAISS returns -1 for missing neighbors
                    continue
                meta = self.metadata[idx].copy()
                meta["rank"]       = rank + 1
                meta["similarity"] = float(sim)
                case_results.append(meta)
            results.append(case_results)

        return results

    def retrieve_metrics(
        self,
        query_labels: List[int],
        retrieved_results: List[List[Dict]],
        k_values: List[int] = [1, 3, 5],
    ) -> Dict:
        """
        Compute retrieval quality metrics for evaluation.

        Metrics:
            MAP@K:  Mean Average Precision at K
            R@K:    Recall at K (fraction of queries with correct label in top-K)
            NDCG@K: Normalized Discounted Cumulative Gain

        Args:
            query_labels:     Ground-truth class index per query.
            retrieved_results: Per-query list of retrieved case dicts (with 'label' key).
            k_values:         List of K values to evaluate.
        Returns:
            Dict of {metric_name: value}.
        """
        metrics = {}

        for k in k_values:
            ap_scores = []
            recall_hits = []
            ndcg_scores = []

            for query_label, results in zip(query_labels, retrieved_results):
                # Top-K results
                top_k_results = results[:k]
                retrieved_labels = [r.get("label", -1) for r in top_k_results]

                # Average Precision @ K
                hits = 0
                precision_at_i = []
                for i, lbl in enumerate(retrieved_labels, 1):
                    if lbl == query_label:
                        hits += 1
                        precision_at_i.append(hits / i)
                ap = np.mean(precision_at_i) if precision_at_i else 0.0
                ap_scores.append(ap)

                # Recall @ K
                correct_in_k = any(lbl == query_label for lbl in retrieved_labels)
                recall_hits.append(float(correct_in_k))

                # NDCG @ K
                rel = [1 if lbl == query_label else 0 for lbl in retrieved_labels]
                dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
                idcg = sum(1 / np.log2(i + 2) for i in range(min(1, k)))  # ideal = 1 hit
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

            metrics[f"MAP@{k}"]  = float(np.mean(ap_scores))
            metrics[f"R@{k}"]    = float(np.mean(recall_hits))
            metrics[f"NDCG@{k}"] = float(np.mean(ndcg_scores))

        return metrics
