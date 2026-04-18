"""
agents/validation_agent.py
===========================
Agent 2 — Clinical Validation Agent.

Cross-checks the AI's predicted diagnosis against 5 retrieved similar
historical cases from the FAISS index. Identifies consistency and flags
any discrepancies that warrant physician attention.

LLM Backend: Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API.
"""

import os
import logging
from typing import List, Dict
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Clinical Validation Specialist Agent.

    System Role:
        "You are a clinical validation specialist. Given a predicted diagnosis
         and 5 retrieved similar historical cases with their outcomes, assess
         whether the prediction is consistent with historical evidence."

    Input:  Predicted diagnosis + RAG retrieved cases.
    Output: Validation report with confidence assessment and discrepancy flags.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        hf_mode: str = "api",
        max_tokens: int = 600,
    ):
        """
        Args:
            model_name: HuggingFace model repository.
            hf_mode:    'api' = HF Inference API | 'local' = local pipeline.
            max_tokens: Max generation tokens.
        """
        logger.info(f"Initializing Validation Agent: {model_name} (mode={hf_mode})")
        self.llm = self._load_llm(model_name, hf_mode, max_tokens)

        # Validation prompt — instructs LLM to act as clinical auditor
        self.prompt = PromptTemplate(
            input_variables=["patient_synopsis", "predicted_diagnosis", "confidence", "retrieved_cases"],
            template="""You are a clinical validation specialist and evidence-based medicine expert.

CURRENT PATIENT CASE:
{patient_synopsis}

AI PREDICTED DIAGNOSIS: {predicted_diagnosis} ({confidence}% confidence)

TOP-5 SIMILAR HISTORICAL CASES retrieved from clinical database (ordered by similarity):
{retrieved_cases}

TASK: Provide a structured validation assessment with:
1. CONSISTENCY CHECK: Is the prediction consistent with the historical cases? (Yes/Partially/No)
2. SUPPORTING EVIDENCE: Which historical cases align with the current prediction?
3. DISCREPANCIES: List any concerning mismatches (different diagnoses for similar presentations).
4. HISTORICAL CONFIDENCE: Based on the retrieved cases, how confident should the clinician be?
5. VERDICT: One overall sentence — is this prediction clinically reliable?

VALIDATION REPORT:""",
        )

    def _load_llm(self, model_name: str, hf_mode: str, max_tokens: int):
        """Load the LLM backend."""
        try:
            if hf_mode == "api":
                from huggingface_hub import InferenceClient

                class HFAPIWrapper:
                    def __init__(self, model: str, max_tok: int):
                        token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
                        self.client  = InferenceClient(api_key=token)
                        self.model   = model
                        self.max_tok = max_tok

                    def invoke(self, prompt: str) -> str:
                        try:
                            response = self.client.chat_completion(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": "You are a clinical validation specialist."},
                                    {"role": "user",   "content": prompt},
                                ],
                                max_tokens=self.max_tok,
                                temperature=0.1,  # Very low: validation needs consistency
                            )
                            return response.choices[0].message.content
                        except Exception as e:
                            logger.error(f"HF API call failed: {e}")
                            return f"[Validation Error: {e}]"

                return HFAPIWrapper(model_name, max_tokens)
            elif hf_mode == "local":
                from langchain_huggingface import HuggingFacePipeline
                return HuggingFacePipeline.from_model_id(
                    model_id=model_name,
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": max_tokens},
                )
        except Exception as e:
            logger.warning(f"Validation LLM backend failed: {e}. Using fallback.")
            return None

    def validate_prediction(
        self,
        patient_synopsis: str,
        predicted_diagnosis: str,
        confidence: float,
        retrieved_cases: List[Dict],
    ) -> str:
        """
        Validate the AI prediction against historical retrieved cases.

        Args:
            patient_synopsis:    Brief summary of current patient presentation.
            predicted_diagnosis: Model's predicted disease class.
            confidence:          Model confidence score [0.0, 1.0].
            retrieved_cases:     List of retrieved case dicts from FAISS:
                                 Each dict has: rank, similarity, image_id, label_name,
                                 text_snippet (truncated case text).
        Returns:
            Validation report string.
        """
        if not retrieved_cases:
            return "Validation skipped: No similar cases in retrieval index."

        if self.llm is None:
            return self._fallback_validation(predicted_diagnosis, retrieved_cases)

        # Format retrieved cases for the prompt
        cases_str = ""
        for case in retrieved_cases:
            rank       = case.get("rank", "?")
            sim        = case.get("similarity", 0.0)
            label      = case.get("label_name", "Unknown")
            img_id     = case.get("image_id", "?")
            snippet    = case.get("text", str(case.get("text_snippet", "No text available.")))[:200]
            cases_str += (
                f"\n  Case #{rank} (Similarity: {sim:.3f}) [{img_id}]\n"
                f"  Diagnosis: {label}\n"
                f"  Clinical Note: {snippet}...\n"
            )

        prompt_text = self.prompt.format(
            patient_synopsis=patient_synopsis[:400],  # Truncate for token budget
            predicted_diagnosis=predicted_diagnosis,
            confidence=round(confidence * 100, 1),
            retrieved_cases=cases_str,
        )

        logger.info("Validation Agent cross-checking prediction against historical cases...")
        return self.llm.invoke(prompt_text)

    def _fallback_validation(self, diagnosis: str, cases: List[Dict]) -> str:
        """Rule-based fallback when LLM is unavailable."""
        matching = [c for c in cases if c.get("label_name") == diagnosis]
        pct = len(matching) / max(len(cases), 1) * 100
        return (
            f"Historical consistency check: {len(matching)}/{len(cases)} retrieved cases "
            f"({pct:.0f}%) share the predicted diagnosis '{diagnosis}'. "
            f"Agreement rating: {'Strong' if pct >= 60 else 'Partial' if pct >= 30 else 'Weak'}. "
            f"[Note: LLM unavailable — using agreement-rate fallback]"
        )
