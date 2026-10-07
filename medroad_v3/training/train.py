"""
MedROAD V3 — Training Pipeline
Trains XGBoost, LSTM, and Transformer on MIMIC-IV cardiac ICU data,
fits the ensemble meta-learner, and persists all models to MODEL_DIR.

Expected MIMIC-IV CSV layout (one row per 5-min window):
  patient_id, window_start, label, <55 feature columns in order>

Usage:
    python -m medroad_v3.training.train --data /path/to/mimic_windows.csv
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from medroad_v3 import config
from medroad_v3.models.deep_models import (
    LSTMModel,
    TemperatureScaledModel,
    TransformerModel,
    predict_batched,
    save_model,
)
from medroad_v3.models.ensemble import EnsembleMeta
from medroad_v3.models.xgboost_model import XGBoostClassifier
from medroad_v3.training.sequences import (
    build_patient_sequences,
    build_sequences_legacy,
    sequence_variation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_mimic_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load MIMIC-IV window CSV.
    Returns X (N, D), y (N,), feature_names list.
    """
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    meta_cols    = ["patient_id", "stay_id", "window_start", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    assert len(feature_cols) == config.N_FEATURES, \
        f"Expected {config.N_FEATURES} feature columns, got {len(feature_cols)}"
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    logger.info("Loaded %d windows  (pos=%d, neg=%d)", len(y), y.sum(), (1-y).sum())
    return X, y, feature_cols


def build_sequences(
    X: np.ndarray,
    seq_len: int = config.SEQ_LEN,
    patient_ids: np.ndarray | None = None,
    window_starts: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build (N, seq_len, D) sequences.

    When patient_ids and window_starts are supplied, genuine sequences are
    assembled from consecutive windows of the same patient. Without them the
    function falls back to tiling, which produces constant sequences and leaves
    the LSTM and Transformer with no temporal signal; a warning is emitted
    because that fallback should never be used for a reported result.
    """
    if patient_ids is not None and window_starts is not None:
        return build_patient_sequences(X, patient_ids, window_starts, seq_len)

    logger.warning(
        "build_sequences called without patient_ids/window_starts: emitting "
        "constant sequences. Pass both to train the sequence models properly."
    )
    return build_sequences_legacy(X, seq_len)


# ── Deep model training ───────────────────────────────────────────────────────

def train_deep_model(
    model:      nn.Module,
    X_seq_tr:   np.ndarray,
    y_tr:       np.ndarray,
    X_seq_val:  np.ndarray,
    y_val:      np.ndarray,
    lr:         float,
    epochs:     int,
    batch_size: int,
    device:     str = "cpu",
) -> tuple[nn.Module, float]:
    """Train LSTM or Transformer; return calibrated model and val AUROC."""
    model = model.to(device)
    # A fold containing no positives would otherwise give pos_weight = 1e6,
    # which overflows the loss. Cap at a ratio that is still aggressive but
    # finite.
    rate = float(y_tr.mean())
    pw = (1 - rate) / rate if rate > 0 else 1.0
    pos_weight = torch.tensor([min(pw, 100.0)], dtype=torch.float32).to(device)
    if rate == 0:
        logger.warning("training fold contains no positive windows")
    # BCEWithLogitsLoss fuses the sigmoid into the loss via the log-sum-exp
    # trick, which is numerically stable where sigmoid followed by BCELoss is
    # not: under a large positive weight the latter can drive activations into
    # saturation, produce NaN, and trip a device-side assertion that the input
    # lies in [0, 1]. pos_weight is also a first-class argument here, so the
    # class imbalance of Sec. 6.5 no longer needs manual per-sample weighting.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_seq_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr,     dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device != "cpu"))

    X_v = torch.tensor(X_seq_val, dtype=torch.float32).to(device)

    best_auc   = 0.0
    best_state = None
    patience   = 10
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                logger.error(
                    "non-finite loss at epoch %d; stopping this model. Reduce "
                    "the learning rate or check for degenerate feature columns.",
                    epoch)
                return model, best_auc
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = model(X_v).cpu().numpy()
        # A validation split containing a single class gives an undefined
        # AUROC. That is a property of the split, not a training failure, so
        # the epoch is skipped rather than allowed to poison early stopping.
        if len(np.unique(y_val)) < 2:
            if epoch == 1:
                logger.warning(
                    "validation split has only one class; early stopping is "
                    "disabled and the final epoch's weights will be kept")
            auc = float("nan")
        else:
            auc = roc_auc_score(y_val, val_probs)

        if np.isfinite(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (best AUROC=%.4f)", epoch, best_auc)
                break

        if epoch % 10 == 0:
            logger.info("Epoch %3d | loss=%.4f | val AUROC=%.4f",
                        epoch, epoch_loss / len(y_t), auc)

    if best_state is None:
        logger.warning(
            "no epoch improved on the initial score; keeping the final weights")
    else:
        model.load_state_dict(best_state)
    logger.info("Best val AUROC: %.4f", best_auc)

    # Temperature scaling calibration
    cal_model = TemperatureScaledModel(model)
    cal_model.calibrate(X_seq_val, y_val)
    return cal_model, best_auc


# ── Main training pipeline ────────────────────────────────────────────────────

def train(data_path: str, output_dir: str = config.MODEL_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    X, y, feature_names = load_mimic_data(data_path)

    # Patient identifiers and window times drive both the splits and the
    # sequence construction, so they are read alongside the features.
    meta_df = pd.read_csv(data_path, usecols=["patient_id", "window_start"])
    groups = meta_df["patient_id"].to_numpy()
    times = meta_df["window_start"].to_numpy()

    # Splits are grouped by patient, never by window. Consecutive five-minute
    # windows from one stay are near-identical, so a random window split puts
    # the same patient on both sides and the model memorises individuals
    # instead of learning deterioration. That inflates validation AUROC towards
    # 1.0 and the result is meaningless.
    def grouped_split(idx, test_frac, seed=42):
        n_splits = max(int(round(1 / test_frac)), 2)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
        tr, te = next(sgkf.split(X[idx], y[idx], groups=groups[idx]))
        return idx[tr], idx[te]

    all_idx = np.arange(len(y))
    tv_idx, test_idx = grouped_split(all_idx, 0.20)
    train_idx, val_idx = grouped_split(tv_idx, 0.25)

    for name, a, b in (("train/val", train_idx, val_idx),
                       ("train/test", train_idx, test_idx),
                       ("val/test", val_idx, test_idx)):
        overlap = set(groups[a]) & set(groups[b])
        if overlap:
            raise RuntimeError(
                f"{len(overlap)} patients appear on both sides of the {name} "
                f"split; results would be inflated by leakage")
    for nm, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        n_pos = int(y[idx].sum())
        if n_pos == 0:
            raise RuntimeError(
                f"the {nm} split contains no positive windows, so nothing can "
                f"be fitted or scored against it. With a small cohort the "
                f"grouped split can land every event on one side; use more "
                f"stays (drop --quick) or a different seed."
            )
        if n_pos < 10:
            logger.warning("%s split has only %d positive windows", nm, n_pos)

    # Persist the split. Any experiment that loads these trained models must
    # evaluate on these same held-out patients; deriving a fresh split gives a
    # "held-out" set largely composed of patients the models were fitted on,
    # and the resulting score is inflated without anything looking wrong.
    import json
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "train_patients": sorted(int(g) for g in set(groups[train_idx])),
            "val_patients":   sorted(int(g) for g in set(groups[val_idx])),
            "test_patients":  sorted(int(g) for g in set(groups[test_idx])),
            "seed": 42,
        }, fh)

    logger.info("patients  train=%d val=%d test=%d (disjoint)",
                len(set(groups[train_idx])), len(set(groups[val_idx])),
                len(set(groups[test_idx])))

    X_tv,    y_tv    = X[tv_idx],    y[tv_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # Sequences are assembled from each patient's own consecutive windows.
    X_seq_tv    = build_sequences(X_tv,    patient_ids=groups[tv_idx],
                                  window_starts=times[tv_idx])
    X_seq_train = build_sequences(X_train, patient_ids=groups[train_idx],
                                  window_starts=times[train_idx])
    X_seq_val   = build_sequences(X_val,   patient_ids=groups[val_idx],
                                  window_starts=times[val_idx])

    # ── 1. XGBoost ────────────────────────────────────────────────────────────
    logger.info("=== Training XGBoost ===")
    xgb_clf = XGBoostClassifier()
    xgb_clf.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    xgb_clf.save(os.path.join(output_dir, "xgb"))
    val_auc = roc_auc_score(y_val, xgb_clf.predict_proba(X_val))
    logger.info("XGBoost val AUROC: %.4f", val_auc)

    # Guard against constant sequences. If this fires, the sequence models are
    # being trained on tiled copies of a single window and cannot learn
    # anything temporal, which silently invalidates the ensemble rationale.
    var = sequence_variation(X_seq_train)
    logger.info("sequence temporal variation: %.5f", var)
    if var < 1e-8:
        logger.error(
            "sequences are constant; pass patient_ids and window_starts to "
            "build_sequences or the LSTM and Transformer are wasted capacity"
        )

    # ── 2. LSTM ───────────────────────────────────────────────────────────────
    logger.info("=== Training LSTM ===")
    lstm, lstm_auc = train_deep_model(
        LSTMModel(), X_seq_train, y_train, X_seq_val, y_val,
        lr=config.LSTM_LR, epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH, device=device,
    )
    save_model(lstm, os.path.join(output_dir, "lstm.pt"))
    logger.info("LSTM val AUROC: %.4f", lstm_auc)

    # ── 3. Transformer ────────────────────────────────────────────────────────
    logger.info("=== Training Transformer ===")
    tf_model, tf_auc = train_deep_model(
        TransformerModel(), X_seq_train, y_train, X_seq_val, y_val,
        lr=config.TF_LR, epochs=config.TF_EPOCHS,
        batch_size=config.TF_BATCH, device=device,
    )
    save_model(tf_model, os.path.join(output_dir, "transformer.pt"))
    logger.info("Transformer val AUROC: %.4f", tf_auc)

    # ── 4. Out-of-fold predictions for meta-learner ───────────────────────────
    logger.info("=== Generating out-of-fold predictions ===")
    skf   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tv_groups = groups[tv_idx]
    xgb_oof  = np.zeros(len(y_tv))
    lstm_oof  = np.zeros(len(y_tv))
    tf_oof    = np.zeros(len(y_tv))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv, groups=tv_groups)):
        logger.info("OOF fold %d/5", fold + 1)
        # XGBoost OOF
        xgb_fold = XGBoostClassifier()
        xgb_fold.train(X_tv[tr_idx], y_tv[tr_idx],
                       X_tv[va_idx], y_tv[va_idx], feature_names)
        xgb_oof[va_idx] = xgb_fold.predict_proba(X_tv[va_idx])

        # LSTM OOF (lightweight: 10 epochs for OOF only)
        lstm_fold, _ = train_deep_model(
            LSTMModel(), X_seq_tv[tr_idx], y_tv[tr_idx],
            X_seq_tv[va_idx], y_tv[va_idx],
            lr=config.LSTM_LR, epochs=10,
            batch_size=config.LSTM_BATCH, device=device,
        )
        lstm_fold.eval()
        with torch.no_grad():
            pred = predict_batched(lstm_fold, X_seq_tv[va_idx], device=device)
            if not np.isfinite(pred).all():
                logger.warning("LSTM fold %d produced non-finite output; "
                               "substituting the base rate", fold + 1)
                pred = np.full_like(pred, float(y_tv.mean()))
            lstm_oof[va_idx] = pred

        # TF OOF
        tf_fold, _ = train_deep_model(
            TransformerModel(), X_seq_tv[tr_idx], y_tv[tr_idx],
            X_seq_tv[va_idx], y_tv[va_idx],
            lr=config.TF_LR, epochs=10,
            batch_size=config.TF_BATCH, device=device,
        )
        tf_fold.eval()
        with torch.no_grad():
            pred = predict_batched(tf_fold, X_seq_tv[va_idx], device=device)
            if not np.isfinite(pred).all():
                logger.warning("Transformer fold %d produced non-finite output; "
                               "substituting the base rate", fold + 1)
                pred = np.full_like(pred, float(y_tv.mean()))
            tf_oof[va_idx] = pred

        # Fold models are not reused; releasing them keeps peak memory flat
        # across the five folds rather than letting it climb.
        del xgb_fold, lstm_fold, tf_fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 5. Meta-learner ───────────────────────────────────────────────────────
    logger.info("=== Training ensemble meta-learner ===")
    xgb_val_prob  = xgb_clf.predict_proba(X_val)
    lstm_val_prob = lstm_oof[:len(y_val)] if len(lstm_oof) >= len(y_val) else xgb_val_prob
    tf_val_prob   = tf_oof[:len(y_val)]   if len(tf_oof)   >= len(y_val) else xgb_val_prob

    lstm.eval()
    tf_model.eval()
    lstm_val_prob = predict_batched(lstm, X_seq_val, device=device)
    tf_val_prob   = predict_batched(tf_model, X_seq_val, device=device)

    for nm, arr in (("xgb", xgb_oof), ("lstm", lstm_oof), ("transformer", tf_oof)):
        bad = ~np.isfinite(arr)
        if bad.any():
            logger.warning("%s OOF has %d non-finite values; replacing with the "
                           "base rate so the meta-learner can still fit",
                           nm, int(bad.sum()))
            arr[bad] = float(y_tv.mean())

    meta = EnsembleMeta()
    meta.fit(
        xgb_oof, lstm_oof, tf_oof, y_tv,
        xgb_val_prob, lstm_val_prob, tf_val_prob, y_val,
    )
    meta.save(os.path.join(output_dir, "ensemble"))

    # ── 6. Held-out test evaluation ───────────────────────────────────────────
    # The test split has been untouched to this point: no model saw it during
    # training, no calibrator was fitted on it, and no threshold was chosen
    # against it. These are therefore the only numbers that should appear in a
    # paper. Validation figures are optimistic by construction.
    logger.info("=== Held-out test evaluation ===")

    # Release the training sequence tensors first. Together they run to a
    # couple of gigabytes, and holding them alongside the test tensor and a
    # CUDA context is enough to take the process down without a traceback.
    import gc
    del X_seq_tv, X_seq_train, X_seq_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_seq_test = build_sequences(X_test, patient_ids=groups[test_idx],
                                 window_starts=times[test_idx])
    xgb_test = xgb_clf.predict_proba(X_test)
    lstm.eval()
    tf_model.eval()
    lstm_test = predict_batched(lstm, X_seq_test, device=device)
    tf_test = predict_batched(tf_model, X_seq_test, device=device)

    ens_test = meta.predict_many(xgb_test, lstm_test, tf_test)

    from medroad_v3.models.calibration import calibration_report
    for name, p_ in (("XGBoost", xgb_test), ("LSTM", lstm_test),
                     ("Transformer", tf_test), ("Ensemble", ens_test)):
        try:
            logger.info("  %-12s %s", name, calibration_report(y_test, p_))
        except ValueError as exc:
            logger.warning("  %-12s not evaluable: %s", name, exc)

    logger.info("Training complete. Models saved to %s", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedROAD V3 models")
    parser.add_argument("--data",   required=True, help="Path to MIMIC-IV CSV")
    parser.add_argument("--output", default=config.MODEL_DIR, help="Model output directory")
    args = parser.parse_args()
    train(args.data, args.output)
