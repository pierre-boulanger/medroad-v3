"""
Experiment 1 — Ensemble ablation with significance testing.

The paper reports XGBoost 0.847, LSTM 0.831, Transformer 0.824 and the full
ensemble 0.871, but gives no intervals and no leave-one-out analysis. The
obvious reviewer question is whether the Transformer, the weakest single
model, earns its place. This experiment answers it.

For every variant it reports AUROC, ECE, MCE and Brier with stratified
bootstrap intervals, and tests each reduced ensemble against the full ensemble
using DeLong for AUROC and a paired bootstrap for ECE.

A null result is a publishable result here. If dropping the Transformer does
not significantly degrade the ensemble, say so and simplify the system.

Usage:
    python -m medroad_v3.experiments.ablation --data mimic_windows.csv
    python -m medroad_v3.experiments.ablation --synthetic      # smoke test
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from medroad_v3.experiments.common import (
    SEED,
    auroc_ci,
    brier_ci,
    delong_test,
    ece_ci,
    latex_table,
    mce_ci,
    paired_bootstrap_diff,
    save_results,
    save_tex,
)
from medroad_v3.models.calibration import (
    IsotonicCalibrator,
    expected_calibration_error,
    reliability_curve,
)

logger = logging.getLogger(__name__)

MODELS = ("xgboost", "lstm", "transformer")


# ══════════════════════════════════════════════════════════════════════════
# Base-model scores
# ══════════════════════════════════════════════════════════════════════════

def base_model_scores(
    X_tr, y_tr, X_te, seed: int = SEED,
    groups_tr=None, times_tr=None, groups_te=None, times_te=None,
) -> dict[str, np.ndarray]:
    """
    Fit the three base learners and return calibrated validation scores.

    Imports are local so that the ablation can be smoke-tested without torch
    or xgboost installed.
    """
    out: dict[str, np.ndarray] = {}

    from medroad_v3.models.xgboost_model import XGBoostClassifier
    xgb = XGBoostClassifier()
    n_val = max(int(0.2 * len(y_tr)), 50)
    xgb.train(X_tr[:-n_val], y_tr[:-n_val], X_tr[-n_val:], y_tr[-n_val:])
    out["xgboost"] = xgb.predict_proba(X_te)

    import torch

    from medroad_v3.models.deep_models import (
        LSTMModel,
        TransformerModel,
        predict_batched,
    )
    from medroad_v3.training.train import build_sequences, train_deep_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Sequences must come from each patient's own consecutive windows; tiling
    # gives the recurrent and attention models constant input and they score
    # exactly 0.5.
    S_tr = build_sequences(X_tr, patient_ids=groups_tr, window_starts=times_tr)
    S_te = build_sequences(X_te, patient_ids=groups_te, window_starts=times_te)
    for name, cls in (("lstm", LSTMModel), ("transformer", TransformerModel)):
        model, _ = train_deep_model(
            cls(), S_tr[:-n_val], y_tr[:-n_val], S_tr[-n_val:], y_tr[-n_val:],
            lr=1e-3, epochs=30, batch_size=256, device=dev,
        )
        out[name] = predict_batched(model, S_te, device=dev)
    return out


def synthetic_scores(n: int = 8000, seed: int = SEED):
    """
    Correlated synthetic scores reproducing the paper's reported ordering,
    used to verify the analysis pipeline end to end without MIMIC-IV.
    """
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.083, n).astype(float)
    shared = rng.normal(size=n)                      # common signal
    def mk(strength, noise):
        z = strength * (shared + 1.9 * y) + noise * rng.normal(size=n)
        return 1.0 / (1.0 + np.exp(-z))
    return y, {
        "xgboost":     mk(0.95, 0.75),
        "lstm":        mk(0.88, 0.85),
        "transformer": mk(0.84, 0.95),
    }


# ══════════════════════════════════════════════════════════════════════════
# Ensemble over an arbitrary subset
# ══════════════════════════════════════════════════════════════════════════

def ensemble_scores(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    subset: tuple[str, ...],
    seed: int = SEED,
) -> np.ndarray:
    """
    Out-of-fold meta-learner over the chosen subset, then isotonic calibration.

    The meta-learner is fitted out of fold so that the reported ensemble score
    is not optimistically biased by having seen its own inputs' labels.
    """
    X = np.column_stack([scores[m] for m in subset])
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]

    # Split the calibration half deterministically by stride rather than at
    # random. These rows are already the held-out patients, so no further
    # grouping is needed, and a stride keeps the halves reproducible.
    idx = np.arange(len(y))
    cal_idx, eval_idx = idx[0::2], idx[1::2]
    cal = IsotonicCalibrator().fit(oof[cal_idx], y[cal_idx])
    out = np.empty(len(y))
    out[eval_idx] = cal.transform(oof[eval_idx])
    out[cal_idx] = oof[cal_idx]

    # Raw out-of-fold scores are returned alongside the calibrated ones.
    # Isotonic calibration can collapse scores onto a handful of values, and
    # the resulting ties drive AUROC towards 0.5 even when the model ranks
    # well, so discrimination is measured before calibration.
    return oof, out


# ══════════════════════════════════════════════════════════════════════════
# Experiment
# ══════════════════════════════════════════════════════════════════════════

def run(scores: dict[str, np.ndarray], y: np.ndarray, n_boot: int = 2000) -> dict:
    variants: dict[str, np.ndarray] = {}

    for m in MODELS:
        variants[m] = scores[m]

    calibrated: dict[str, np.ndarray] = {m: scores[m] for m in MODELS}

    for pair in itertools.combinations(MODELS, 2):
        dropped = next(m for m in MODELS if m not in pair)
        raw, cal = ensemble_scores(scores, y, pair)
        variants[f"ensemble -{dropped}"] = raw
        calibrated[f"ensemble -{dropped}"] = cal

    full, full_cal = ensemble_scores(scores, y, MODELS)
    variants["ensemble (full)"] = full
    calibrated["ensemble (full)"] = full_cal

    results = {}
    for name, p in variants.items():
        row = {
            "auroc": auroc_ci(y, p, n_boot=n_boot).to_dict(),
            "ece":   ece_ci(y, calibrated[name], n_boot=n_boot).to_dict(),
            "mce":   mce_ci(y, calibrated[name], n_boot=n_boot).to_dict(),
            "brier": brier_ci(y, p, n_boot=n_boot).to_dict(),
        }
        if name != "ensemble (full)":
            _, _, pv = delong_test(y, full, p)
            row["delong_p_vs_full"] = pv
            row["ece_diff_vs_full"] = paired_bootstrap_diff(
                y, p, full, expected_calibration_error, n_boot=n_boot
            ).to_dict()
        results[name] = row
    return results, variants, calibrated


def to_latex(results: dict) -> str:
    rows = []
    for name, r in results.items():
        a, e = r["auroc"], r["ece"]
        auc = f"{a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"
        ece = f"{e['value']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"
        if "delong_p_vs_full" in r:
            pv = r["delong_p_vs_full"]
            ptxt = "$<$0.001" if pv < 1e-3 else f"{pv:.3f}"
        else:
            ptxt = "---"
        label = name.replace("_", r"\_").replace("-", "$-$")
        rows.append([label, auc, ece, f"{r['brier']['value']:.4f}", ptxt])

    return latex_table(
        rows,
        ["Variant", "AUROC [95\\% CI]", "ECE [95\\% CI]", "Brier", "$p$ vs full"],
        caption=(
            "Ensemble ablation. Intervals are stratified bootstrap percentile "
            "intervals over 2000 replicates. The $p$-value is DeLong's test for "
            "the AUROC difference against the full ensemble on the same windows."
        ),
        label="tab:ablation",
        col_spec="@{}lllrr@{}",
        note=(
            "A variant whose interval overlaps the full ensemble and whose "
            "$p$-value exceeds 0.05 is not distinguishable from it on this cohort."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MedROAD V3 ensemble ablation")
    ap.add_argument("--data", help="MIMIC-IV window CSV")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic scores to verify the pipeline")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.synthetic:
        y, scores = synthetic_scores()
        logger.info("synthetic cohort: n=%d, event rate=%.3f", len(y), y.mean())
    else:
        if not args.data:
            ap.error("--data is required unless --synthetic is given")
        import pandas as pd

        from medroad_v3.training.train import load_mimic_data
        X, y, _ = load_mimic_data(args.data)
        meta_df = pd.read_csv(args.data, usecols=["patient_id", "window_start"])
        groups = meta_df["patient_id"].to_numpy()
        times = meta_df["window_start"].to_numpy()

        # Grouped by patient. A random split over windows puts the same stay on
        # both sides and drives AUROC towards 1.0 regardless of real skill.
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        tr, te = next(sgkf.split(X, y, groups=groups))
        assert not (set(groups[tr]) & set(groups[te])), "patient leakage"
        logger.info("patients train=%d test=%d (disjoint)",
                    len(set(groups[tr])), len(set(groups[te])))

        scores = base_model_scores(
            X[tr], y[tr], X[te],
            groups_tr=groups[tr], times_tr=times[tr],
            groups_te=groups[te], times_te=times[te],
        )
        y = y[te].astype(float)

    results, variants_for_curve, calibrated = run(scores, y, n_boot=args.n_boot)

    print(f"\n{'variant':>20}  {'AUROC':>22}  {'ECE':>22}  {'p vs full':>10}")
    for name, r in results.items():
        a, e = r["auroc"], r["ece"]
        pv = r.get("delong_p_vs_full")
        ptxt = "—" if pv is None else (f"{pv:.2e}" if pv < 1e-3 else f"{pv:.3f}")
        print(f"{name:>20}  {a['value']:.3f} [{a['lo']:.3f},{a['hi']:.3f}]  "
              f"{e['value']:.3f} [{e['lo']:.3f},{e['hi']:.3f}]  {ptxt:>10}")

    save_results(results, args.out, "ablation")
    save_tex(to_latex(results), args.out, "ablation_table")

    # Reliability diagram coordinates, so the figure in the manuscript plots
    # measured points rather than illustrative ones.
    cal_full = calibrated["ensemble (full)"]
    raw_full = variants_for_curve["ensemble (full)"]
    series = {}
    for probs, label in ((raw_full, "uncalibrated"), (cal_full, "calibrated")):
        mp, of, cnt = reliability_curve(y, probs, n_bins=10)
        series[label] = " ".join(
            f"({m:.4f},{o:.4f})"
            for m, o, c in zip(mp, of, cnt, strict=True)
            if c > 0 and np.isfinite(m) and np.isfinite(o)
        )

    n_pos, n_tot = int(y.sum()), int(y.size)
    fig = rf"""% Reliability diagram - generated from measured predictions.
% Paste directly into the manuscript; requires pgfplots.
\begin{{figure}}[ht]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.9\linewidth, height=5.5cm,
  xlabel={{Mean predicted probability $\bar{{R}}_k$}},
  ylabel={{Observed event fraction $\bar{{Y}}_k$}},
  xmin=0, xmax=1, ymin=0, ymax=1,
  xtick={{0,0.2,0.4,0.6,0.8,1.0}}, ytick={{0,0.2,0.4,0.6,0.8,1.0}},
  grid=major, grid style={{dashed,gray!30}},
  legend pos=north west, legend style={{font=\footnotesize}}
]
\addplot[black, dashed, thin] coordinates {{(0,0)(1,1)}};
\addlegendentry{{Perfect calibration}}
\addplot[red!70, thick, mark=square*] coordinates {{
  {series['uncalibrated']}
}};
\addlegendentry{{Uncalibrated ensemble}}
\addplot[blue!80, thick, mark=*] coordinates {{
  {series['calibrated']}
}};
\addlegendentry{{Calibrated ensemble}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Reliability diagram for the ensemble on held-out patients
($n = {n_tot:,} windows, {n_pos:,} positive). Deciles containing no
observations are omitted. Points above the diagonal indicate underestimated
risk, below it overestimated.}}
\label{{fig:calibration_curve}}
\end{{figure}}
"""
    save_tex(fig, args.out, "reliability_figure")
    print(f"wrote {Path(args.out)}/reliability_figure.tex (paste into the paper)")
    print(f"\nwrote {Path(args.out)}/ablation.json and ablation_table.tex")


if __name__ == "__main__":
    main()
