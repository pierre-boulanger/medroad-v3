"""
Experiment 2 — Retrospective RPM degradation.

Section 3.6 of the paper claims the pipeline extends to wearable streams
without architectural change. That claim is currently supported by a mapping
table and nothing else, which is the weakest point in a submission to a
telemedicine special issue.

This experiment supplies retrospective evidence. MIMIC-IV windows are degraded
to what a given home monitoring kit could actually observe: channels with no
wearable analogue are removed, channels whose device reports less often than
the ICU charts them are coarsened to that cadence, and the remaining gaps are
imputed exactly as the deployed pipeline would impute them. The model is then
recalibrated on the degraded representation and the loss in AUROC and ECE is
reported per kit.

This is a simulation of sparsity, not a wearable validation study, and the
paper must say so. It establishes an upper bound on achievable RPM performance:
real wearables add measurement error on top of the sparsity modelled here.

Usage:
    python -m medroad_v3.experiments.rpm_degradation --data mimic_windows.csv
    python -m medroad_v3.experiments.rpm_degradation --synthetic
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from medroad_v3.experiments.common import (
    SEED,
    auroc_ci,
    delong_test,
    ece_ci,
    latex_table,
    save_results,
    save_tex,
)
from medroad_v3.models.calibration import IsotonicCalibrator
from medroad_v3.rpm.mapping import (
    BY_ICU_FEATURE,
    DEVICE_PROFILES,
    recommended_window_seconds,
)

logger = logging.getLogger(__name__)

# Kits in increasing order of instrumentation
KITS: dict[str, tuple[str, ...]] = {
    "ICU (reference)":    (),                  # no masking
    "HF kit (3 devices)": ("Pulse oximeter", "BP cuff", "Smart scale"),
    "Full RPM (5 devices)": tuple(p.name for p in DEVICE_PROFILES),
}


# ══════════════════════════════════════════════════════════════════════════
# Mapping a kit onto feature columns
# ══════════════════════════════════════════════════════════════════════════

def kit_loincs(kit: tuple[str, ...]) -> set[str]:
    return {lo for p in DEVICE_PROFILES if p.name in kit for lo in p.loincs}


def observable_icu_features(kit: tuple[str, ...]) -> set[str]:
    """ICU feature names a kit can observe, via the Table 3 mapping."""
    codes = kit_loincs(kit)
    return {
        f.icu_feature for f in BY_ICU_FEATURE.values()
        if f.usable and f.loinc in codes
    }


def unmapped_channels(feature_names: list[str]) -> dict[str, str]:
    """
    Device channels in Table 3 that have no corresponding column in the feature
    vector, and therefore contribute nothing to the model however good the
    sensor is.

    This is a substantive check, not a sanity check. A kit can only help to the
    extent the trained model has an input for what it measures: a mapping row
    for a signal the vector does not contain is aspirational rather than
    operative, and the paper should say which of its rows are which.
    """
    low = [n.lower() for n in feature_names]
    out: dict[str, str] = {}
    for p in DEVICE_PROFILES:
        for lo in p.loincs:
            feat = next(
                (f for f in BY_ICU_FEATURE.values() if f.loinc == lo), None
            )
            if feat is None or not feat.usable:
                continue
            if not any(feat.icu_feature in n for n in low):
                out[feat.icu_feature] = p.name
    return out


def column_mask(feature_names: list[str], kit: tuple[str, ...]) -> np.ndarray:
    """
    Boolean mask over feature columns: True where the kit can observe the
    underlying signal. Matching is by substring against the ICU feature name,
    which is how the engineered columns are named (``vital_spo2``,
    ``lab_troponin_i``, ``delta_bnp`` and so on).

    Metadata, missingness and temporal columns are always retained: a remote
    deployment still knows how complete its own data is, and that is precisely
    the signal the metadata features were introduced to carry.
    """
    if not kit:
        return np.ones(len(feature_names), dtype=bool)

    observable = observable_icu_features(kit)
    keep = np.zeros(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        low = name.lower()

        # Missingness indicators follow their channel. Retaining
        # miss_lab_troponin_i for a kit that never measures troponin keeps a
        # column that is constant by construction in deployment but varies in
        # the ICU training data, where it encodes which patients were worked up
        # and how. That is a documentation fingerprint, and because fingerprints
        # are patient-specific it transfers as noise, or worse as an inverted
        # signal, to held-out patients.
        if low.startswith("miss_"):
            keep[i] = any(f in low for f in observable)
            continue

        # Aggregate coverage and temporal context survive: a remote deployment
        # still knows how complete its own data is and what time it is.
        if low.startswith(("meta_", "hour_")) or low in ("sig", "bcr"):
            keep[i] = True
            continue

        keep[i] = any(f in low for f in observable)
    return keep


def coarsen(
    X: np.ndarray,
    feature_names: list[str],
    kit: tuple[str, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the kit to a feature matrix.

    Unobservable channels are zeroed, which is the deployed pipeline's
    behaviour for a channel with no wearable analogue. Observable channels
    whose device reports less often than once per window are thinned: a
    fraction of windows carry a stale carry-forward value rather than a fresh
    reading, with the staleness rate set by the device cadence against the
    recommended window width.
    """
    Xd = X.copy()
    keep = column_mask(feature_names, kit)
    Xd[:, ~keep] = 0.0
    if not kit:
        return Xd

    window = recommended_window_seconds(
        [p for p in DEVICE_PROFILES if p.name in kit]
    )
    for p in DEVICE_PROFILES:
        if p.name not in kit or p.cadence_s <= window:
            continue
        stale = 1.0 - window / p.cadence_s
        cols = [
            i for i, n in enumerate(feature_names)
            if keep[i] and any(
                f.icu_feature in n.lower()
                for f in BY_ICU_FEATURE.values() if f.loinc in p.loincs
            )
        ]
        for c in cols:
            idx = rng.random(len(Xd)) < stale
            Xd[idx, c] = np.roll(Xd[:, c], 1)[idx]   # carry forward
    return Xd


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════

def fit_and_score(X_tr, y_tr, X_te, seed: int = SEED
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a calibrated scorer on the degraded representation.

    Logistic regression is used deliberately rather than the full ensemble: the
    question is how much signal survives the degradation, and a linear scorer
    answers that without confounding it with capacity differences between
    architectures. Replace with the full ensemble for headline numbers.
    """
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    lr.fit(X_tr, y_tr)
    raw_tr = lr.predict_proba(X_tr)[:, 1]
    raw_te = lr.predict_proba(X_te)[:, 1]
    cal = IsotonicCalibrator().fit(raw_tr, y_tr)
    cal_te = cal.transform(raw_te)

    # Returned separately on purpose. Isotonic calibration fitted on training
    # scores can map an entire test set onto a single value once those scores
    # fall outside the fitted range, and a constant prediction has AUROC 0.5
    # however well the underlying model discriminates. Discrimination is
    # therefore measured on raw scores, calibration quality on calibrated ones.
    return raw_te, cal_te


def run(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_boot: int = 2000,
    seed: int = SEED,
    groups: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(seed)

    unmapped = unmapped_channels(feature_names)
    if unmapped:
        logger.warning(
            "%d Table 3 channels have no feature column and cannot contribute: %s",
            len(unmapped),
            ", ".join(f"{k} ({v})" for k, v in unmapped.items()),
        )

    if groups is None:
        groups = np.arange(len(y))        # degenerate: every row its own patient
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(X, y, groups=groups))
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    results: dict[str, dict] = {}
    reference: np.ndarray | None = None

    for label, kit in KITS.items():
        Xtr_d = coarsen(X_tr, feature_names, kit, rng)
        Xte_d = coarsen(X_te, feature_names, kit, rng)
        p_raw, p_cal = fit_and_score(Xtr_d, y_tr, Xte_d, seed)

        keep = column_mask(feature_names, kit)
        entry = {
            "devices": list(kit) or ["full ICU instrumentation"],
            "features_retained": int(keep.sum()),
            "features_total": len(feature_names),
            "window_seconds": recommended_window_seconds(
                [q for q in DEVICE_PROFILES if q.name in kit]
            ) if kit else 300,
            "auroc": auroc_ci(y_te, p_raw, n_boot=n_boot).to_dict(),
            "ece":   ece_ci(y_te, p_cal, n_boot=n_boot).to_dict(),
        }
        if reference is None:
            reference = p_raw
        else:
            _, _, pv = delong_test(y_te, reference, p_raw)
            entry["delong_p_vs_icu"] = pv
            entry["auroc_loss"] = (
                results["ICU (reference)"]["auroc"]["value"] - entry["auroc"]["value"]
            )
        results[label] = entry
    results["_unmapped_channels"] = unmapped
    return results


def to_latex(results: dict) -> str:
    rows = []
    for name, r in results.items():
        if name.startswith("_"):
            continue
        a = r["auroc"]
        loss = r.get("auroc_loss")
        rows.append([
            name,
            f"{r['features_retained']}/{r['features_total']}",
            f"{r['window_seconds'] // 60} min" if r["window_seconds"] < 86400 else "24 h",
            f"{a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]",
            f"{r['ece']['value']:.3f}",
            "---" if loss is None else f"$-${loss:.3f}",
        ])
    return latex_table(
        rows,
        ["Deployment", "Features", "Window", "AUROC [95\\% CI]", "ECE", "$\\Delta$AUROC"],
        caption=(
            "Retrospective degradation of MIMIC-IV windows to remote monitoring "
            "kits. Channels with no wearable analogue are removed and observable "
            "channels are coarsened to their device cadence; the scorer is "
            "recalibrated on each degraded representation."
        ),
        label="tab:rpm_degradation",
        col_spec="@{}llllrr@{}",
        note=(
            "This simulates sparsity only. Real wearable deployment adds "
            "measurement error, so these figures bound achievable RPM "
            "performance from above."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MedROAD V3 RPM degradation experiment")
    ap.add_argument("--data", help="MIMIC-IV window CSV")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results/rpm")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.synthetic:
        from medroad_v3.experiments.synth import synthetic_cohort
        X, y, feature_names = synthetic_cohort()
        logger.info("synthetic cohort: n=%d, d=%d, event rate=%.3f",
                    len(y), X.shape[1], y.mean())
    else:
        if not args.data:
            ap.error("--data is required unless --synthetic is given")
        from medroad_v3.training.train import load_mimic_data
        X, y, feature_names = load_mimic_data(args.data)

    grp = None
    if not args.synthetic:
        import pandas as pd
        grp = pd.read_csv(args.data, usecols=["patient_id"])["patient_id"].to_numpy()
    results = run(X, y.astype(float), feature_names, n_boot=args.n_boot, groups=grp)
    unmapped = results.get("_unmapped_channels", {})

    print(f"\n{'deployment':>22}  {'feat':>7}  {'AUROC':>22}  {'dAUROC':>8}")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        a = r["auroc"]
        loss = r.get("auroc_loss")
        ltxt = "—" if loss is None else f"-{loss:.3f}"
        print(f"{name:>22}  {r['features_retained']:>3}/{r['features_total']:<3}  "
              f"{a['value']:.3f} [{a['lo']:.3f},{a['hi']:.3f}]  {ltxt:>8}")

    if unmapped:
        print("\nChannels in Table 3 with no feature column (contribute nothing):")
        for k, v in unmapped.items():
            print(f"    {k:<16} published by {v}")

    save_results(results, args.out, "rpm_degradation")
    save_tex(to_latex(results), args.out, "rpm_degradation_table")
    print(f"\nwrote {Path(args.out)}/rpm_degradation.json and _table.tex")


if __name__ == "__main__":
    main()
