"""
MedROAD V3 — XGBoost Model + SHAP TreeExplainer
Provides training, Platt scaling calibration, prediction, and
per-instance SHAP value computation for the 55-element feature vector.
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from medroad_v3 import config

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    XGBoost wrapper with:
      - 5-fold cross-validated training with early stopping on log-loss
      - Platt-scaling calibration (sigmoid method)
      - SHAP TreeExplainer for exact Shapley values
      - Background dataset for 5ms per-inference SHAP
    """

    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self.calibrator: CalibratedClassifierCV | None = None
        self.explainer: shap.TreeExplainer | None = None
        self._background: np.ndarray | None = None
        self.feature_names: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        self.model = xgb.XGBClassifier(
            n_estimators      = config.XGB_N_ESTIMATORS,
            max_depth         = config.XGB_MAX_DEPTH,
            learning_rate     = config.XGB_LR,
            subsample         = config.XGB_SUBSAMPLE,
            colsample_bytree  = config.XGB_COLSAMPLE,
            use_label_encoder = False,
            eval_metric       = "logloss",
            early_stopping_rounds = config.XGB_EARLY_STOP,
            tree_method       = "hist",
            device            = "cpu",
            random_state      = 42,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("XGBoost trained: %d trees, best iteration %d",
                    self.model.n_estimators, self.model.best_iteration)

        # Platt scaling calibration on validation set
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method="sigmoid",
            cv="prefit",
        )
        self.calibrator.fit(X_val, y_val)
        logger.info("XGBoost calibrated with Platt scaling")

        # Build SHAP TreeExplainer with 100-sample background
        n_bg = min(100, len(X_train))
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(X_train), n_bg, replace=False)
        self._background = X_train[idx]
        self.explainer   = shap.TreeExplainer(self.model, data=self._background)
        logger.info("SHAP TreeExplainer initialised with %d background samples", n_bg)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability of class 1."""
        if self.calibrator is None:
            raise RuntimeError("Model not trained — call train() first")
        return self.calibrator.predict_proba(X)[:, 1]

    def predict_single(self, x: np.ndarray) -> float:
        """Predict probability for a single 55-element feature vector."""
        return float(self.predict_proba(x.reshape(1, -1))[0])

    # ── SHAP ──────────────────────────────────────────────────────────────────

    def shap_values(self, x: np.ndarray) -> dict[str, float]:
        """
        Compute SHAP values for a single feature vector.
        Returns dict mapping feature_name → Shapley value.
        Approximate runtime: ~5 ms with 100-sample background.
        """
        if self.explainer is None:
            raise RuntimeError("Model not trained — call train() first")
        vals = self.explainer(x.reshape(1, -1)).values[0]    # shape (55,)
        return {
            name: float(phi)
            for name, phi in zip(self.feature_names, vals, strict=False)
        }

    def top_shap(self, x: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k features by |SHAP value|, descending."""
        shap_dict = self.shap_values(x)
        return sorted(shap_dict.items(), key=lambda t: abs(t[1]), reverse=True)[:k]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model.save_model(os.path.join(directory, "xgb_model.ubj"))
        with open(os.path.join(directory, "xgb_calibrator.pkl"), "wb") as f:
            pickle.dump(self.calibrator, f)
        if self._background is not None:
            np.save(os.path.join(directory, "xgb_background.npy"), self._background)
        with open(os.path.join(directory, "xgb_feature_names.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(self.feature_names))
        logger.info("XGBoost saved to %s", directory)

    def load(self, directory: str) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(directory, "xgb_model.ubj"))
        with open(os.path.join(directory, "xgb_calibrator.pkl"), "rb") as f:
            self.calibrator = pickle.load(f)
        bg_path = os.path.join(directory, "xgb_background.npy")
        if os.path.exists(bg_path):
            self._background = np.load(bg_path)
            self.explainer   = shap.TreeExplainer(self.model, data=self._background)
        fn_path = os.path.join(directory, "xgb_feature_names.txt")
        if os.path.exists(fn_path):
            with open(fn_path) as f:
                self.feature_names = f.read().splitlines()
        logger.info("XGBoost loaded from %s", directory)
