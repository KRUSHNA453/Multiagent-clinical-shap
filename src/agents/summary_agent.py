"""
agents/summary_agent.py
========================
Agent 3 — Clinical Summary Agent.

Synthesizes all pipeline outputs (prediction + XAI + validation) into
a professional, structured clinical report formatted for physician review.

LLM Backend: Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API.
"""

import os
import logging
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
import config

logger = logging.getLogger(__name__)

class SummaryAgent:
    """
    Expert Medical Report Writer Agent.
    Refactored to use InferenceClient direct integration.
    """

    def __init__(self):
        logger.info(f"Initializing Summary Agent with {config.HF_MODEL}")
        self.client = InferenceClient(
            model=config.HF_MODEL,
            token=config.HF_TOKEN
        )
        self.system_prompt = """You are an expert medical report writer. Synthesize the model prediction, XAI explanation, and validation findings into a professional, structured clinical report suitable for a physician. Use professional Markdown formatting."""

    def generate_report(
        self,
        patient_synopsis: str,
        diagnosis: str,
        confidence: float,
        clinical_reasoning: str,
        validation_report: str,
        image_modality: str = "Image",
    ) -> str:
        """
        Generate final clinical report.
        """
        prompt = f"""
        INPUT DATA:
        - Patient Synopsis: {patient_synopsis}
        - AI Predicted Diagnosis: {diagnosis}
        - Model Confidence: {confidence:.2%}
        - Explanation Agent Output: {clinical_reasoning}
        - Validation Agent Output: {validation_report}

        TASK: Generate a structured Markdown report with EXACTLY these sections:
        ## 🏥 Patient Summary
        ## 🔬 Predicted Diagnosis & Confidence
        ## 🧠 Clinical Reasoning (XAI)
        ## 📁 Historical Case Validation
        ## ⚠️ Caveats & Limitations
        ## 📋 Recommended Next Steps
        """

        try:
            logger.info("Summary Agent generating final report...")
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Summary Agent failed: {e}")
            return f"[Summary Error: {e}]"
