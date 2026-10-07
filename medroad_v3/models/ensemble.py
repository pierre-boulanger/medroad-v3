"""
MedROAD V3 — Ensemble Meta-Learner
Combines XGBoost + LSTM + Transformer outputs via logistic regression
meta-learner trained on out-of-fold predictions.
Calibrated with isotonic regression.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from medroad_v3 import config

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    risk_score:      float                   # calibrated ensemble probability
    xgb_score:       float
    lstm_score:      float
    transformer_score: float
    alert:           bool                    # risk_score >= tau
    shap_values:     dict[str, float]        # from XGBoost TreeExplainer
    top_features:    list[tuple[str, float]] # top-5 SHAP features


class EnsembleMeta:
    """
    Logistic regression meta-learner over (p_xgb, p_lstm, p_tf).
    Calibrated with isotonic regression on a held-out validation set.
    """

    def __init__(self, threshold: float = config.RISK_THRESHOLD) -> None:
        self.threshold = threshold
        self.meta: LogisticRegression | None       = None
        self.calibrator: IsotonicRegression | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        xgb_oof:  np.ndarray,   # (N,) out-of-fold XGBoost probabilities
        lstm_oof: np.ndarray,   # (N,) out-of-fold LSTM probabilities
        tf_oof:   np.ndarray,   # (N,) out-of-fold Transformer probabilities
        y:        np.ndarray,   # (N,) labels
        xgb_val:  np.ndarray,   # (M,) validation set for calibration
        lstm_val: np.ndarray,
        tf_val:   np.ndarray,
        y_val:    np.ndarray,
    ) -> None:
        # Fit meta-learner on OOF predictions
        X_oof = np.column_stack([xgb_oof, lstm_oof, tf_oof])
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta.fit(X_oof, y)
        logger.info("Meta-learner weights: XGB=%.3f LSTM=%.3f TF=%.3f bias=%.3f",
                    self.meta.coef_[0][0], self.meta.coef_[0][1],
                    self.meta.coef_[0][2], self.meta.intercept_[0])

        # Isotonic regression calibration on validation set
        X_val = np.column_stack([xgb_val, lstm_val, tf_val])
        raw_val = self.meta.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_val, y_val)

        # Evaluation
        cal_val = self.calibrator.predict(raw_val)
        auc = roc_auc_score(y_val, cal_val)
        f1  = f1_score(y_val, (cal_val >= self.threshold).astype(int))
        logger.info("Ensemble val AUROC=%.4f  F1@τ=%.2f=%.4f", auc, self.threshold, f1)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        p_xgb:  float,
        p_lstm: float,
        p_tf:   float,
    ) -> float:
        """Return calibrated ensemble risk score in [0, 1]."""
        if self.meta is None:
            raise RuntimeError("Ensemble not fitted — call fit() first")
        X = np.array([[p_xgb, p_lstm, p_tf]])
        raw  = float(self.meta.predict_proba(X)[0, 1])
        cal  = float(self.calibrator.predict([raw])[0])
        return cal

    def predict_many(
        self,
        p_xgb: np.ndarray,
        p_lstm: np.ndarray,
        p_tf: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised equivalent of predict() over whole arrays.

        Calling predict() per row costs one sklearn dispatch each, which at
        tens of thousands of held-out windows dominates evaluation time and
        churns memory badly enough to risk the process.
        """
        if self.meta is None:
            raise RuntimeError("Ensemble not fitted - call fit() first")
        X = np.column_stack([np.asarray(p_xgb), np.asarray(p_lstm),
                             np.asarray(p_tf)])
        raw = self.meta.predict_proba(X)[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "ensemble_meta.pkl"), "wb") as f:
            pickle.dump({"meta": self.meta, "calibrator": self.calibrator,
                         "threshold": self.threshold}, f)
        logger.info("Ensemble saved to %s", directory)

    def load(self, directory: str) -> None:
        with open(os.path.join(directory, "ensemble_meta.pkl"), "rb") as f:
            state = pickle.load(f)
        self.meta        = state["meta"]
        self.calibrator  = state["calibrator"]
        self.threshold   = state.get("threshold", config.RISK_THRESHOLD)
        logger.info("Ensemble loaded from %s (τ=%.2f)", directory, self.threshold)
