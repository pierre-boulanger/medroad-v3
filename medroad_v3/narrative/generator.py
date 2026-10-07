"""
MedROAD V3 — Claude API Narrative Generator
Generates structured three-sentence clinical narratives
conditioned on ensemble risk score + top-5 SHAP features.
"""
from __future__ import annotations

import logging

import anthropic

from medroad_v3 import config

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a clinical decision support system embedded in an ICU electronic health record.
Your role is to generate concise, accurate, and actionable clinical alerts.
Always write exactly three sentences:
  1. State the risk score and its clinical significance.
  2. Name the two most important physiological drivers from the SHAP analysis.
  3. Recommend the single most appropriate immediate clinical action.
Use plain clinical language. Do not speculate beyond the provided data.
Do not mention AI, machine learning, or algorithms in the output.
"""


def _format_feature(name: str, shap_val: float) -> str:
    """Convert internal feature name to clinical label."""
    label = (
        name.replace("vital_", "")
            .replace("lab_", "")
            .replace("meta_", "")
            .replace("miss_vital_", "missing ")
            .replace("miss_lab_", "missing ")
            .replace("delta_", "rising ")
            .replace("_", " ")
    )
    direction = "elevated" if shap_val > 0 else "reduced"
    return f"{label} ({direction}, SHAP={shap_val:+.3f})"


class NarrativeGenerator:
    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None
        if config.ANTHROPIC_API_KEY:
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            logger.warning("ANTHROPIC_API_KEY not set — narrative generation disabled")

    def generate(
        self,
        patient_id:   str,
        risk_score:   float,
        top_features: list[tuple[str, float]],
        feature_dict: dict[str, float],
    ) -> str:
        """
        Call Claude API and return a three-sentence clinical narrative.
        Falls back to a template if API is unavailable.
        """
        if self._client is None:
            return self._template_narrative(risk_score, top_features)

        feature_lines = "\n".join(
            f"  {i+1}. {_format_feature(name, phi)}"
            for i, (name, phi) in enumerate(top_features[:5])
        )

        user_prompt = (
            f"Patient ID: {patient_id}\n"
            f"Ensemble risk score: {risk_score:.3f} (threshold 0.72)\n"
            f"Top physiological drivers (SHAP analysis):\n{feature_lines}\n\n"
            "Write the three-sentence clinical alert."
        )

        try:
            response = self._client.messages.create(
                model       = config.CLAUDE_MODEL,
                max_tokens  = config.CLAUDE_MAX_TOKENS,
                temperature = config.CLAUDE_TEMPERATURE,
                system      = _SYSTEM_PROMPT,
                messages    = [{"role": "user", "content": user_prompt}],
            )
            narrative = response.content[0].text.strip()
            logger.debug("Narrative generated (%d chars)", len(narrative))
            return narrative
        except anthropic.APIError as exc:
            logger.error("Claude API error: %s — falling back to template", exc)
            return self._template_narrative(risk_score, top_features)

    @staticmethod
    def _template_narrative(
        risk_score: float,
        top_features: list[tuple[str, float]],
    ) -> str:
        severity = "high" if risk_score >= 0.85 else "elevated"
        driver1 = _format_feature(*top_features[0]) if top_features else "unknown"
        driver2 = _format_feature(*top_features[1]) if len(top_features) > 1 else "unknown"
        return (
            f"MedROAD V3 has assigned a {severity} deterioration risk score of "
            f"{risk_score:.3f}, exceeding the F1-optimised clinical threshold of 0.72. "
            f"The primary physiological drivers are {driver1} and {driver2}. "
            "Immediate bedside clinical assessment is recommended."
        )
