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
from huggingface_hub import InferenceClient
import config

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Clinical Validation Specialist Agent.
    Refactored to use InferenceClient direct integration.
    """

    def __init__(self):
        logger.info(f"Initializing Validation Agent with {config.HF_MODEL}")
        self.client = InferenceClient(
            model=config.HF_MODEL,
            token=config.HF_TOKEN
        )
        self.system_prompt = """You are a clinical validation specialist. Given a predicted diagnosis and 5 retrieved similar historical cases with their outcomes, assess whether the prediction is consistent with historical evidence. Identify specific patterns or discrepancies."""

    def validate_prediction(
        self,
        predicted_label: str,
        retrieved_cases: List[Dict],
    ) -> str:
        """
        Validate prediction against historical cases.
        """
        cases_str = "\n".join([
            f"- Case {c.get('image_id', c.get('id', '?'))} ({c.get('similarity', 0.0):.2%} sim): Diagnosed as '{c.get('label_name', c.get('label', 'Unknown'))}'"
            for c in retrieved_cases
        ])

        prompt = f"""
        Prediction to Validate: {predicted_label}

        Historical Similar Cases:
        {cases_str}

        Task: Compare the current prediction against these 5 historical cases. 
        1. Is the prediction consistent with the majority of high-similarity cases?
        2. Are there any 'label shift' cases (high similarity but different diagnosis)?
        3. Provide a final validation verdict: [Strongly Supported | Moderately Supported | Discrepancy Found].
        """

        try:
            logger.info("Validation Agent validating prediction...")
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
            logger.error(f"Validation Agent failed: {e}")
            return f"[Validation Error: {e}]"
