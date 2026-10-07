"""
Experiment 3 — Rule-based baselines on the same cohort.

The paper claims reduced alert fatigue, but supports it by comparing an
estimated 0.8 alerts per patient per shift against a literature figure of 2 to
5 hourly drawn from a different population. That is not a controlled
comparison and it is the weakest claim in the manuscript.

This experiment scores NEWS and MEWS on the same windows from the same feature
vector and compares alert burden at matched sensitivity. Matching on
sensitivity is what makes the comparison fair: any system can reduce alerts by
detecting less. The reportable quantity is how many alerts each system raises
while catching the same fraction of events.

Usage:
    python -m medroad_v3.experiments.baselines --data mimic_windows.csv
    python -m medroad_v3.experiments.baselines --synthetic
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from medroad_v3 import config
from medroad_v3.experiments.common import (
    SEED,
    alerts_per_patient_shift,
    auroc_ci,
    delong_test,
    latex_table,
    save_results,
    save_tex,
    threshold_at_sensitivity,
)

logger = logging.getLogger(__name__)

# Sensitivities at which alert burden is compared
SENS_GRID = (0.70, 0.80, 0.90)


# ══════════════════════════════════════════════════════════════════════════
# Rule-based scores
# ══════════════════════════════════════════════════════════════════════════

def _col(feature_names: list[str], *fragments: str) -> int | None:
    """First column whose name contains all fragments."""
    for i, n in enumerate(feature_names):
        low = n.lower()
        if all(f in low for f in fragments):
            return i
    return None


def _denorm(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Invert the min-max normalisation applied during feature engineering."""
    return v * (hi - lo) + lo


def news_score(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    National Early Warning Score, computed from the engineered vector.

    Columns are de-normalised back to clinical units using the reference
    ranges in features.engineering before the NEWS bands are applied, so the
    bands mean what they mean at the bedside. Oxygen supplementation and level
    of consciousness are approximated: the former is unavailable in the vector
    and scored zero, the latter is taken from the GCS channel.
    """
    from medroad_v3.features.engineering import _VITAL_RANGES

    n = len(X)
    score = np.zeros(n)

    def band(frag, ranges, edges, points):
        i = _col(feature_names, "vital_", frag)
        if i is None:
            return np.zeros(n)
        v = _denorm(X[:, i], *ranges)
        s = np.zeros(n)
        for (lo, hi), p in zip(edges, points, strict=True):
            s[(v >= lo) & (v < hi)] = p
        return s

    score += band("respiratory_rate", _VITAL_RANGES["respiratory_rate"],
                  [(-np.inf, 9), (9, 12), (12, 21), (21, 25), (25, np.inf)],
                  [3, 1, 0, 2, 3])
    score += band("spo2", _VITAL_RANGES["spo2"],
                  [(-np.inf, 92), (92, 94), (94, 96), (96, np.inf)],
                  [3, 2, 1, 0])
    score += band("temperature", _VITAL_RANGES["temperature"],
                  [(-np.inf, 35.1), (35.1, 36.1), (36.1, 38.1),
                   (38.1, 39.1), (39.1, np.inf)],
                  [3, 1, 0, 1, 2])
    score += band("systolic_bp", _VITAL_RANGES["systolic_bp"],
                  [(-np.inf, 91), (91, 101), (101, 111), (111, 220), (220, np.inf)],
                  [3, 2, 1, 0, 3])
    score += band("heart_rate", _VITAL_RANGES["heart_rate"],
                  [(-np.inf, 41), (41, 51), (51, 91), (91, 111),
                   (111, 131), (131, np.inf)],
                  [3, 1, 0, 1, 2, 3])
    score += band("gcs", _VITAL_RANGES["gcs"],
                  [(-np.inf, 15), (15, np.inf)], [3, 0])
    return score


def mews_score(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Modified Early Warning Score over the same channels."""
    from medroad_v3.features.engineering import _VITAL_RANGES

    n = len(X)
    score = np.zeros(n)

    def band(frag, ranges, edges, points):
        i = _col(feature_names, "vital_", frag)
        if i is None:
            return np.zeros(n)
        v = _denorm(X[:, i], *ranges)
        s = np.zeros(n)
        for (lo, hi), p in zip(edges, points, strict=True):
            s[(v >= lo) & (v < hi)] = p
        return s

    score += band("systolic_bp", _VITAL_RANGES["systolic_bp"],
                  [(-np.inf, 71), (71, 81), (81, 101), (101, 200), (200, np.inf)],
                  [3, 2, 1, 0, 2])
    score += band("heart_rate", _VITAL_RANGES["heart_rate"],
                  [(-np.inf, 41), (41, 51), (51, 101), (101, 111),
                   (111, 130), (130, np.inf)],
                  [2, 1, 0, 1, 2, 3])
    score += band("respiratory_rate", _VITAL_RANGES["respiratory_rate"],
                  [(-np.inf, 9), (9, 15), (15, 21), (21, 30), (30, np.inf)],
                  [2, 0, 1, 2, 3])
    score += band("temperature", _VITAL_RANGES["temperature"],
                  [(-np.inf, 35.1), (35.1, 38.5), (38.5, np.inf)], [2, 0, 2])
    score += band("gcs", _VITAL_RANGES["gcs"],
                  [(-np.inf, 15), (15, np.inf)], [3, 0])
    return score


# ══════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════

def run(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    medroad_prob: np.ndarray,
    n_boot: int = 2000,
) -> dict:
    systems = {
        "NEWS":        news_score(X, feature_names),
        "MEWS":        mews_score(X, feature_names),
        "MedROAD V3":  medroad_prob,
    }

    results: dict[str, dict] = {}
    for name, s in systems.items():
        entry = {"auroc": auroc_ci(y, s, n_boot=n_boot).to_dict(), "burden": {}}
        for sens in SENS_GRID:
            tau = threshold_at_sensitivity(y, s, sens)
            fired = s >= tau
            tp = float(((fired) & (y == 1)).sum())
            entry["burden"][f"sens_{int(sens*100)}"] = {
                "threshold": tau,
                "alerts_per_patient_shift": alerts_per_patient_shift(s, tau),
                "ppv": tp / max(fired.sum(), 1),
                "alert_fraction": float(fired.mean()),
            }
        results[name] = entry

    for name in ("NEWS", "MEWS"):
        _, _, p = delong_test(y, medroad_prob, systems[name])
        results[name]["delong_p_vs_medroad"] = p

    base = results["MedROAD V3"]["burden"]
    for name in ("NEWS", "MEWS"):
        for k, v in results[name]["burden"].items():
            v["alert_ratio_vs_medroad"] = (
                v["alerts_per_patient_shift"]
                / max(base[k]["alerts_per_patient_shift"], 1e-9)
            )
    return results


def to_latex(results: dict) -> str:
    rows = []
    for name, r in results.items():
        a = r["auroc"]
        cells = [name, f"{a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"]
        for sens in SENS_GRID:
            b = r["burden"][f"sens_{int(sens*100)}"]
            cells.append(f"{b['alerts_per_patient_shift']:.2f}")
        p = r.get("delong_p_vs_medroad")
        cells.append("---" if p is None else ("$<$0.001" if p < 1e-3 else f"{p:.3f}"))
        rows.append(cells)

    header = (["System", "AUROC [95\\% CI]"]
              + [f"Alerts @ {int(s*100)}\\% sens." for s in SENS_GRID]
              + ["$p$ vs V2"])
    return latex_table(
        rows, header,
        caption=(
            "Alert burden at matched sensitivity on the MIMIC-IV validation "
            "cohort. Alerts are per patient per 8-hour shift. Matching on "
            "sensitivity ensures each system is compared while detecting the "
            "same fraction of deterioration events."
        ),
        label="tab:baselines",
        col_spec="@{}ll" + "r" * len(SENS_GRID) + "r@{}",
        note=(
            "NEWS and MEWS are computed from the same engineered vector, "
            "de-normalised to clinical units. Oxygen supplementation is "
            "unavailable in the cohort and scored zero in NEWS."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MedROAD V3 rule-based baselines")
    ap.add_argument("--data", help="MIMIC-IV window CSV")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", default=config.MODEL_DIR,
                    help="directory holding the trained models")
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.synthetic:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        from medroad_v3.experiments.synth import synthetic_cohort
        X, y, feature_names = synthetic_cohort()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=SEED
        )
        lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        prob = lr.predict_proba(Xte)[:, 1]
        X, y = Xte, yte
    else:
        if not args.data:
            ap.error("--data is required unless --synthetic is given")

        # Checked before anything heavy is imported, so a missing split fails
        # in a second with a useful message rather than after a model load.
        md = Path(args.models)
        split_file = md / "split.json"
        if not split_file.exists():
            raise SystemExit(
                f"{split_file} not found. It records which patients the models "
                "were held out from, and without it this comparison would score "
                "MedROAD on its own training data. Retrain to generate it:\n"
                "    python -m medroad_v3.training.train --data <csv>"
            )

        import pandas as pd
        import torch

        from medroad_v3.models.deep_models import (
            LSTMModel,
            TemperatureScaledModel,
            TransformerModel,
            load_model,
            predict_batched,
        )
        from medroad_v3.models.ensemble import EnsembleMeta
        from medroad_v3.models.xgboost_model import XGBoostClassifier
        from medroad_v3.training.sequences import build_patient_sequences
        from medroad_v3.training.train import load_mimic_data

        X, y, feature_names = load_mimic_data(args.data)
        meta_df = pd.read_csv(args.data, usecols=["patient_id", "window_start"])
        groups = meta_df["patient_id"].to_numpy()
        times = meta_df["window_start"].to_numpy()

        # Evaluate on exactly the patients the models were held out from.
        import json
        test_patients = set(
            json.loads(split_file.read_text(encoding="utf-8"))["test_patients"])
        te = np.flatnonzero(np.isin(groups, list(test_patients)))
        logger.info("scoring the %d patients held out during training",
                    len(test_patients))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        xgb = XGBoostClassifier()
        xgb.load(str(md / "xgb"))
        lstm = load_model(lambda: TemperatureScaledModel(LSTMModel()),
                          str(md / "lstm.pt"), dev)
        tfm = load_model(lambda: TemperatureScaledModel(TransformerModel()),
                         str(md / "transformer.pt"), dev)
        ens = EnsembleMeta()
        ens.load(str(md / "ensemble"))

        S = build_patient_sequences(X[te], groups[te], times[te])
        prob = ens.predict_many(
            xgb.predict_proba(X[te]),
            predict_batched(lstm, S, device=dev),
            predict_batched(tfm, S, device=dev),
        )
        X, y = X[te], y[te]

    results = run(X, y.astype(float), feature_names, prob, n_boot=args.n_boot)

    print(f"\n{'system':>12}  {'AUROC':>22}   " +
          "  ".join(f"@{int(s*100)}%" for s in SENS_GRID))
    for name, r in results.items():
        a = r["auroc"]
        burdens = "  ".join(
            f"{r['burden'][f'sens_{int(s*100)}']['alerts_per_patient_shift']:5.2f}"
            for s in SENS_GRID
        )
        print(f"{name:>12}  {a['value']:.3f} [{a['lo']:.3f},{a['hi']:.3f}]  {burdens}")

    save_results(results, args.out, "baselines")
    save_tex(to_latex(results), args.out, "baselines_table")
    print(f"\nwrote {Path(args.out)}/baselines.json and baselines_table.tex")


if __name__ == "__main__":
    main()
