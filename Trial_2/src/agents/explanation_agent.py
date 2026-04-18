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
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    Clinical AI Explainability Expert Agent.

    System Role:
        "You are a clinical AI explainability expert. Given SHAP token scores and
         Grad-CAM region descriptions, translate the model's reasoning into clear
         clinical language that a physician can understand."

    Input:  SHAP results + Grad-CAM summary for a patient case.
    Output: Paragraph explaining WHY the model made its prediction.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        hf_mode: str = "api",
        max_tokens: int = 600,
    ):
        """
        Args:
            model_name: HuggingFace model repository (Qwen recommended).
            hf_mode:    'api' = HF Inference API | 'local' = local pipeline.
            max_tokens: Max generation tokens per response.
        """
        logger.info(f"Initializing Explanation Agent: {model_name} (mode={hf_mode})")
        self.llm = self._load_llm(model_name, hf_mode, max_tokens)

        # System prompt — instructs LLM to act as clinical explainability expert
        self.prompt = PromptTemplate(
            input_variables=[
                "diagnosis", "confidence", "shap_tokens",
                "gradcam_region", "image_modality", "body_region",
            ],
            template="""You are a clinical AI explainability expert working with a radiologist.

The multimodal AI system has analyzed a {image_modality} scan of the {body_region} together with the patient's clinical notes, and has predicted the following diagnosis:

PREDICTED DIAGNOSIS: {diagnosis}
CONFIDENCE: {confidence}%

The model's explainability analysis identified:

KEY CLINICAL TERMS FROM NOTE (SHAP importance scores — positive = supports diagnosis):
{shap_tokens}

RADIOLOGICAL FINDING REGIONS (Grad-CAM heatmap highlighted):
{gradcam_region}

TASK: In 3-4 concise, professional sentences, explain WHY the combination of these radiological findings and clinical note features logically supports the diagnosis of {diagnosis}. 
Write for a physician audience. Use proper medical terminology. Be specific about the clinical reasoning chain.
Do NOT repeat the raw scores — synthesize them into clinical logic.

CLINICAL REASONING:""",
        )

    def _load_llm(self, model_name: str, hf_mode: str, max_tokens: int):
        """Load the LLM backend (HF API or local pipeline)."""
        try:
            if hf_mode == "api":
                from huggingface_hub import InferenceClient

                class HFAPIWrapper:
                    """Thin wrapper around HuggingFace InferenceClient for LangChain compatibility."""
                    def __init__(self, model: str, max_tok: int):
                        token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
                        self.client  = InferenceClient(api_key=token)
                        self.model   = model
                        self.max_tok = max_tok

                    def invoke(self, prompt: str) -> str:
                        """Send prompt to HF Inference API and return text response."""
                        try:
                            response = self.client.chat_completion(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": "You are a clinical AI medical expert."},
                                    {"role": "user",   "content": prompt},
                                ],
                                max_tokens=self.max_tok,
                                temperature=0.15,   # Low temperature = more deterministic, factual
                            )
                            return response.choices[0].message.content
                        except Exception as e:
                            logger.error(f"HF API call failed: {e}")
                            return f"[LLM Error — clinical explanation unavailable: {e}]"

                return HFAPIWrapper(model_name, max_tokens)

            elif hf_mode == "local":
                from langchain_huggingface import HuggingFacePipeline
                return HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": max_tokens},
                )
        except Exception as e:
            logger.warning(f"LLM backend failed to load: {e}. Agent will use fallback mode.")
            return None

    def generate_reasoning(
        self,
        diagnosis: str,
        confidence: float,
        shap_tokens: List[Dict],
        gradcam_region: str = "Diffuse activation across the lesion area",
        image_modality: str = "CT",
        body_region: str = "chest",
    ) -> str:
        """
        Generate clinical reasoning explanation.

        Args:
            diagnosis:      Predicted disease name (e.g., "Neoplasm, carcinoma").
            confidence:     Model confidence score [0.0, 1.0].
            shap_tokens:    List of {'token': str, 'shap_score': float} dicts.
            gradcam_region: Human-readable Grad-CAM region description.
            image_modality: CT or MR.
            body_region:    Anatomical body region.
        Returns:
            Clinical reasoning paragraph (string).
        """
        if self.llm is None:
            return self._fallback_reasoning(diagnosis, confidence, shap_tokens, gradcam_region)

        # Format SHAP tokens for the prompt
        shap_str = "\n".join([
            f"  • '{t['token']}' (score: {t['shap_score']:+.4f})"
            for t in sorted(shap_tokens, key=lambda x: abs(x["shap_score"]), reverse=True)[:10]
        ]) or "  No significant tokens identified."

        prompt_text = self.prompt.format(
            diagnosis=diagnosis,
            confidence=round(confidence * 100, 1),
            shap_tokens=shap_str,
            gradcam_region=gradcam_region,
            image_modality=image_modality,
            body_region=body_region,
        )

        logger.info("Explanation Agent generating clinical reasoning...")
        return self.llm.invoke(prompt_text)

    def _fallback_reasoning(
        self, diagnosis: str, confidence: float,
        shap_tokens: List[Dict], gradcam_region: str
    ) -> str:
        """Rule-based fallback when LLM is unavailable."""
        top_tokens = [t["token"] for t in shap_tokens[:5]]
        return (
            f"The model predicted '{diagnosis}' with {confidence:.1%} confidence. "
            f"Key clinical terms contributing to this prediction include: {', '.join(top_tokens)}. "
            f"The radiological analysis highlighted activation in the {gradcam_region}. "
            f"[Note: LLM reasoning unavailable — using keyword summary fallback]"
        )
