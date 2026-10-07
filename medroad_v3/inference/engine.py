"""
MedROAD V3 — Inference Engine
Orchestrates feature extraction → XGBoost + LSTM + Transformer →
ensemble meta-learner → SHAP → narrative → FHIR write-back.
End-to-end target: <675 ms without narrative, <2475 ms with Claude.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch

from medroad_v3 import config
from medroad_v3.features.engineering import FeatureEngineer
from medroad_v3.fhir.client import FHIRClient
from medroad_v3.fhir.writeback import FHIRWriteBack
from medroad_v3.models.deep_models import (
    LSTMModel,
    TemperatureScaledModel,
    TransformerModel,
    load_model,
    predict_sequence,
)
from medroad_v3.models.ensemble import EnsembleMeta, EnsembleResult
from medroad_v3.models.xgboost_model import XGBoostClassifier
from medroad_v3.narrative.generator import NarrativeGenerator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Stateless per-patient inference.  All models are loaded once at startup.
    """

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()
        self.xgb   = XGBoostClassifier()
        self.lstm:  torch.nn.Module | None = None
        self.tf:    torch.nn.Module | None = None
        self.meta   = EnsembleMeta()
        self.narrative_gen = NarrativeGenerator()
        self._loaded = False
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self, model_dir: str = config.MODEL_DIR) -> None:
        """Load all persisted models from disk."""
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                "Run training/train.py first."
            )
        self.xgb.load(os.path.join(model_dir, "xgb"))

        lstm_path = os.path.join(model_dir, "lstm.pt")
        self.lstm = load_model(
            lambda: TemperatureScaledModel(LSTMModel()),
            lstm_path, self.device
        )

        tf_path = os.path.join(model_dir, "transformer.pt")
        self.tf = load_model(
            lambda: TemperatureScaledModel(TransformerModel()),
            tf_path, self.device
        )

        self.meta.load(os.path.join(model_dir, "ensemble"))
        self._loaded = True
        logger.info("All models loaded from %s (device=%s)", model_dir, self.device)

    # ── Core inference ────────────────────────────────────────────────────────

    def infer(
        self,
        patient_id:      str,
        window_obs:      list[dict[str, Any]],   # vitals in last 5 min
        lab_obs_24h:     list[dict[str, Any]],   # labs in last 24 h
        prior_labs:      dict[str, float] | None,
        med_requests:    list[dict[str, Any]],
        encounter_start: datetime | None,
        sequence:        np.ndarray | None = None,  # (seq_len, 55) pre-built
        encounter_id:    str | None = None,
        generate_narrative: bool = True,
        fhir_client:     FHIRClient | None = None,
    ) -> EnsembleResult:
        """
        Full inference pipeline for one patient at one timepoint.
        Returns EnsembleResult; optionally writes back to FHIR.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded — call load_models() first")

        t0 = time.perf_counter()

        # 1. Feature engineering
        vector, feature_dict = self.feature_engineer.build_vector(
            window_obs, lab_obs_24h, prior_labs, med_requests, encounter_start
        )

        # 2. Build sequence for deep models
        if sequence is None:
            # Replicate static vector as a trivial sequence (real use: windowed)
            sequence = np.tile(vector, (config.SEQ_LEN, 1))  # (seq_len, 55)

        # 3. XGBoost prediction + SHAP
        p_xgb = self.xgb.predict_single(vector)
        shap_vals   = self.xgb.shap_values(vector)
        top_features = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        # 4. LSTM prediction
        p_lstm = predict_sequence(self.lstm, sequence, device=self.device)  # type: ignore

        # 5. Transformer prediction
        p_tf = predict_sequence(self.tf, sequence, device=self.device)  # type: ignore

        # 6. Ensemble meta-learner
        risk_score = self.meta.predict(p_xgb, p_lstm, p_tf)
        alert      = risk_score >= self.meta.threshold

        t_inference = time.perf_counter() - t0
        logger.info(
            "Patient %s | XGB=%.3f LSTM=%.3f TF=%.3f → R=%.3f (alert=%s) [%.0f ms]",
            patient_id, p_xgb, p_lstm, p_tf, risk_score, alert, t_inference * 1000
        )

        # 7. Narrative (only if alert and API key configured)
        narrative = ""
        if alert and generate_narrative and config.ANTHROPIC_API_KEY:
            narrative = self.narrative_gen.generate(
                patient_id   = patient_id,
                risk_score   = risk_score,
                top_features = top_features,
                feature_dict = feature_dict,
            )
            t_total = time.perf_counter() - t0
            logger.info("Narrative generated [%.0f ms total]", t_total * 1000)

        result = EnsembleResult(
            risk_score        = risk_score,
            xgb_score         = p_xgb,
            lstm_score        = p_lstm,
            transformer_score = p_tf,
            alert             = alert,
            shap_values       = shap_vals,
            top_features      = top_features,
        )

        # 8. FHIR write-back (if alert and client provided)
        if alert and fhir_client is not None:
            wb = FHIRWriteBack(fhir_client)
            wb.write_alert(
                patient_id    = patient_id,
                risk_score    = risk_score,
                narrative     = narrative or "Narrative generation disabled.",
                shap_values   = shap_vals,
                feature_vector = feature_dict,
                encounter_id  = encounter_id,
            )

        return result
