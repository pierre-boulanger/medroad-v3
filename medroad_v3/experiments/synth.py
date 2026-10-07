"""
Synthetic cohort generator.

Produces a feature matrix with the same column names and dimensionality as the
engineered MIMIC-IV vector, so every experiment in this package can be executed
end to end without credentialed data. Results from synthetic data verify that
the analysis pipeline runs and that its statistics behave; they say nothing
about clinical performance and must never appear in the paper.
"""
from __future__ import annotations

import numpy as np

from medroad_v3 import config
from medroad_v3.features.engineering import LAB_NAMES, VITAL_NAMES

SEED = 20260731


def feature_names() -> list[str]:
    """Column names matching medroad_v3.features.engineering.build_vector."""
    return (
        [f"vital_{n}_{st}" for n in VITAL_NAMES
         for st in ("mean", "std", "min", "rate")]
        + [f"lab_{n}" for n in LAB_NAMES]
        + ["meta_vital_cov", "meta_lab_cov", "meta_abg_present", "meta_doc_lag",
           "meta_vital_recency", "meta_med_burden", "meta_iv_drip", "meta_icu_los"]
        + [f"miss_vital_{n}" for n in VITAL_NAMES]
        + [f"miss_lab_{n}" for n in LAB_NAMES]
        + ["delta_troponin_i", "delta_bnp", "delta_lactate"]
        + ["delta_weight_24h", "delta_weight_72h"]
        + ["sig", "bcr"]
        + ["hour_sin", "hour_cos"]
    )


def synthetic_cohort(
    n: int = 12000,
    event_rate: float = 0.083,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate (X, y, feature_names) with a plausible signal structure.

    Signal is concentrated in the channels that genuinely drive cardiac
    deterioration, so that masking those channels in the RPM experiment
    produces a measurable loss rather than noise.
    """
    rng = np.random.default_rng(seed)
    names = feature_names()
    d = len(names)
    assert d == config.N_FEATURES, f"{d} != {config.N_FEATURES}"

    y = rng.binomial(1, event_rate, n).astype(float)
    X = rng.normal(0.5, 0.15, size=(n, d))

    # Channels carrying real signal, with effect sizes
    effects = {
        "lab_lactate": 0.30, "lab_troponin_i": 0.26, "lab_bnp": 0.22,
        "delta_lactate": 0.20, "delta_bnp": 0.16, "lab_creatinine": 0.14,
        "vital_spo2_mean": -0.24, "vital_heart_rate_mean": 0.20,
        "vital_systolic_bp_mean": -0.18, "vital_respiratory_rate_mean": 0.15,
        "delta_weight_72h": 0.28, "delta_weight_24h": 0.18,
        "vital_weight_mean": 0.08,
        "meta_doc_lag": 0.10, "meta_abg_present": 0.12,
    }
    idx = {nm: i for i, nm in enumerate(names)}
    for nm, beta in effects.items():
        if nm in idx:
            X[:, idx[nm]] += beta * y + rng.normal(0, 0.05, n)

    # Binary columns
    for i, nm in enumerate(names):
        if nm.startswith("miss_") or nm in ("meta_abg_present", "meta_iv_drip"):
            X[:, i] = (X[:, i] > 0.5).astype(float)

    return np.clip(X, 0.0, 1.0), y, names
