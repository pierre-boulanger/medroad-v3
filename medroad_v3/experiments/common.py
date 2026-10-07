"""
Shared utilities for the MedROAD V3 experiment suite.

Provides the statistics the paper currently lacks: bootstrap confidence
intervals on every reported metric, DeLong's test for comparing correlated
AUROCs, and LaTeX table emission so results go straight into the manuscript
without hand transcription.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from medroad_v3.models.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
)

logger = logging.getLogger(__name__)

N_BOOT_DEFAULT = 2000
SEED = 20260731


# ══════════════════════════════════════════════════════════════════════════
# Interval estimation
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Estimate:
    """A point estimate with a bootstrap percentile interval."""
    value: float
    lo: float
    hi: float
    n_boot: int = N_BOOT_DEFAULT

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def tex(self, decimals: int = 3) -> str:
        d = decimals
        return f"{self.value:.{d}f} [{self.lo:.{d}f}, {self.hi:.{d}f}]"

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOT_DEFAULT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Estimate:
    """
    Percentile bootstrap interval for any metric of the form f(y_true, y_prob).

    Resampling is stratified by outcome so that every replicate preserves the
    event rate. With an 8% event rate an unstratified bootstrap occasionally
    draws replicates containing no positives, for which AUROC is undefined.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("bootstrap requires both classes to be present")

    point = float(metric_fn(y_true, y_prob))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        ip = rng.choice(pos, len(pos), replace=True)
        ineg = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        reps[b] = metric_fn(y_true[idx], y_prob[idx])

    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), n_boot)


def auroc_ci(y_true, y_prob, **kw) -> Estimate:
    return bootstrap_metric(y_true, y_prob, roc_auc_score, **kw)


def ece_ci(y_true, y_prob, n_bins: int = 10, **kw) -> Estimate:
    return bootstrap_metric(
        y_true, y_prob, lambda a, b: expected_calibration_error(a, b, n_bins), **kw
    )


def mce_ci(y_true, y_prob, n_bins: int = 10, **kw) -> Estimate:
    return bootstrap_metric(
        y_true, y_prob, lambda a, b: maximum_calibration_error(a, b, n_bins), **kw
    )


def brier_ci(y_true, y_prob, **kw) -> Estimate:
    return bootstrap_metric(
        y_true, y_prob, lambda a, b: float(np.mean((b - a) ** 2)), **kw
    )


# ══════════════════════════════════════════════════════════════════════════
# DeLong's test for two correlated ROC curves
# ══════════════════════════════════════════════════════════════════════════

def _midrank(x: np.ndarray) -> np.ndarray:
    """
    Mid-ranks, ties averaged.

    Delegates to scipy's vectorised implementation. The hand-written loop this
    replaces was O(n) in Python and broke on large inputs, which only showed up
    once the cohort reached hundreds of thousands of windows.
    """
    return stats.rankdata(x, method="average")


def _structural_components(scores: np.ndarray, n_pos: int):
    """V10 / V01 components of the DeLong variance estimator."""
    m, n = n_pos, scores.shape[1] - n_pos
    pos = scores[:, :m]
    neg = scores[:, m:]
    k = scores.shape[0]

    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(scores[r]) for r in range(k)])

    auc = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    return auc, s01, s10, m, n


def delong_test(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
) -> tuple[float, float, float]:
    """
    DeLong's test for the difference between two correlated AUROCs computed
    on the same samples, which is the situation in an ablation where every
    variant is scored on the identical validation set.

    Returns (auc_a, auc_b, two-sided p-value).
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)          # positives first
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_sorted == 1))
    scores = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])

    auc, s01, s10, m, n = _structural_components(scores, n_pos)
    cov = s01 / m + s10 / n
    delta = np.array([1.0, -1.0])
    var = float(delta @ cov @ delta)
    if var <= 0:
        return float(auc[0]), float(auc[1]), 1.0
    z = (auc[0] - auc[1]) / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(auc[0]), float(auc[1]), float(p)


def paired_bootstrap_diff(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOT_DEFAULT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Estimate:
    """
    Paired bootstrap interval on metric(a) - metric(b). Used for ECE and Brier,
    where no analytic test like DeLong's exists. An interval excluding zero
    indicates a significant difference at the stated level.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(prob_a)
    b = np.asarray(prob_b)
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    point = float(metric_fn(y_true, a) - metric_fn(y_true, b))

    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        reps[i] = metric_fn(y_true[idx], a[idx]) - metric_fn(y_true[idx], b[idx])

    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), n_boot)


# ══════════════════════════════════════════════════════════════════════════
# Operating-point helpers
# ══════════════════════════════════════════════════════════════════════════

def threshold_at_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sens: float
) -> float:
    """
    Lowest threshold achieving at least the target sensitivity. Matching on
    sensitivity is what makes an alert-rate comparison between two scoring
    systems fair: a system that alerts more will always catch more.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = y_prob[y_true == 1]
    if len(pos) == 0:
        raise ValueError("no positive cases")
    return float(np.quantile(pos, 1.0 - target_sens))


def alerts_per_patient_shift(
    y_prob: np.ndarray,
    tau: float,
    windows_per_patient_shift: float = 96.0,
) -> float:
    """
    Expected alerts per patient per 8-hour shift. With 5-minute windows an
    8-hour shift contains 96 windows.
    """
    return float((np.asarray(y_prob) >= tau).mean() * windows_per_patient_shift)


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════

def latex_table(
    rows: list[list[str]],
    header: list[str],
    caption: str,
    label: str,
    col_spec: str | None = None,
    note: str | None = None,
) -> str:
    """Emit a booktabs table ready to paste into the manuscript."""
    spec = col_spec or ("@{}l" + "r" * (len(header) - 1) + "@{}")
    out = [
        r"\begin{table}[ht]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\",
        r"\midrule",
    ]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(rf"\\[2pt] \footnotesize {note}")
    out.append(r"\end{table}")
    return "\n".join(out)


def save_results(obj: dict, out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", p)
    return p


def save_tex(tex: str, out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.tex"
    p.write_text(tex, encoding="utf-8")
    logger.info("wrote %s", p)
    return p
