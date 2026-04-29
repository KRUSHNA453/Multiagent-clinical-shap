"""
agents/explanation_agent.py
============================
Agent 1 — Clinical Explanation Agent.

Takes SHAP token scores and Grad-CAM region descriptions and translates
the model's reasoning into clear clinical language that a physician can
understand and act upon.

LLM Backend: Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API.
"""

import os
import logging
from typing import List, Dict
from huggingface_hub import InferenceClient
import config

logger = logging.getLogger(__name__)

class ExplanationAgent:
    """
    Clinical AI Explainability Expert Agent.
    Refactored to use InferenceClient direct integration.
    """

    def __init__(self):
        logger.info(f"Initializing Explanation Agent with {config.HF_MODEL}")
        self.client = InferenceClient(
            model=config.HF_MODEL,
            token=config.HF_TOKEN
        )
        self.system_prompt = """You are a clinical AI explainability expert. Given SHAP token scores and Grad-CAM region descriptions, translate the model's reasoning into clear clinical language that a physician can understand. Be specific, concise, and use proper medical terminology."""

    def generate_reasoning(
        self,
        diagnosis: str,
        confidence: float,
        shap_tokens: List[str],
        gradcam_region: str = "lesion area",
    ) -> str:
        """
        Generate clinical reasoning explanation.
        """
        # Handle shap_tokens being either List[str] or List[Dict]
        shap_token_list = []
        for t in shap_tokens:
            if isinstance(t, dict):
                shap_token_list.append(t.get('token', ''))
            else:
                shap_token_list.append(str(t))

        prompt = f"""
        Patient case analysis:
        - Predicted Diagnosis: {diagnosis}
        - Model Confidence: {confidence:.2%}
        - Key Clinical Tokens (SHAP): {', '.join(shap_token_list)}
        - Image Hotspot Region (Grad-CAM): {gradcam_region}

        Explain in 3-4 sentences why the model predicted {diagnosis}, referencing the specific tokens and image region above. 
        """

        try:
            logger.info("Explanation Agent generating reasoning...")
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Explanation Agent failed: {e}")
            return f"[LLM Error: {e}]"
