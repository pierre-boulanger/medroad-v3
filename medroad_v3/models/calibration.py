"""
MedROAD V3 — Probability Calibration
Implements Section 6 of the paper ("Model Calibration").

Provides:
  * Calibration metrics: ECE, MCE, Brier score, Hosmer-Lemeshow
  * Platt scaling          (used for XGBoost)
  * Temperature scaling    (used for LSTM / Transformer)
  * Isotonic regression    (used for the ensemble meta-learner)
  * Distributional-shift detection and site-specific recalibration
  * Threshold selection over the sensitivity-specificity trade-off
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Calibration metrics  (paper Section 6.1)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationReport:
    ece: float           # Expected Calibration Error
    mce: float           # Maximum Calibration Error
    brier: float         # Brier score
    auroc: float
    hl_stat: float       # Hosmer-Lemeshow chi-square statistic
    hl_pvalue: float
    n_bins: int

    def __str__(self) -> str:
        return (f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  "
                f"Brier={self.brier:.4f}  AUROC={self.auroc:.4f}  "
                f"HL chi2={self.hl_stat:.2f} (p={self.hl_pvalue:.3f})")

    def to_dict(self) -> dict:
        return asdict(self)


def _bin_indices(probs: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Equal-width binning of [0, 1] as used for ECE/MCE in the paper."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        out.append(np.where(mask)[0])
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    n = len(y_true)
    ece = 0.0
    for idx in _bin_indices(y_prob, n_bins):
        if len(idx) == 0:
            continue
        acc  = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece += (len(idx) / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """MCE = max_b |acc(B_b) - conf(B_b)| over non-empty bins."""
    gaps = []
    for idx in _bin_indices(y_prob, n_bins):
        if len(idx) == 0:
            continue
        gaps.append(abs(y_true[idx].mean() - y_prob[idx].mean()))
    return float(max(gaps)) if gaps else 0.0


def hosmer_lemeshow(
    y_true: np.ndarray, y_prob: np.ndarray, n_groups: int = 10
) -> tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test using deciles of risk.
    Returns (chi-square statistic, p-value). Large p => well calibrated.
    """
    order = np.argsort(y_prob)
    groups = np.array_split(order, n_groups)
    stat = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        obs  = y_true[g].sum()
        exp  = y_prob[g].sum()
        n_g  = len(g)
        denom = exp * (1.0 - exp / n_g)
        if denom > 1e-9:
            stat += (obs - exp) ** 2 / denom
    dof = max(n_groups - 2, 1)
    p = 1.0 - stats.chi2.cdf(stat, dof)
    return float(stat), float(p)


def calibration_report(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """Compute the full metric set reported in Table 'calibration' of the paper."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-7, 1 - 1e-7)
    hl_stat, hl_p = hosmer_lemeshow(y_true, y_prob, n_bins)
    return CalibrationReport(
        ece       = expected_calibration_error(y_true, y_prob, n_bins),
        mce       = maximum_calibration_error(y_true, y_prob, n_bins),
        brier     = float(brier_score_loss(y_true, y_prob)),
        auroc     = float(roc_auc_score(y_true, y_prob)),
        hl_stat   = hl_stat,
        hl_pvalue = hl_p,
        n_bins    = n_bins,
    )


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean_predicted, observed_frequency, bin_counts) for plotting the
    reliability diagram (paper Fig. 'calibration_curve').
    """
    mp, of, cnt = [], [], []
    for idx in _bin_indices(np.asarray(y_prob), n_bins):
        if len(idx) == 0:
            mp.append(np.nan)
            of.append(np.nan)
            cnt.append(0)
            continue
        mp.append(float(np.asarray(y_prob)[idx].mean()))
        of.append(float(np.asarray(y_true)[idx].mean()))
        cnt.append(int(len(idx)))
    return np.array(mp), np.array(of), np.array(cnt)


# ══════════════════════════════════════════════════════════════════════════
# Calibrators  (paper Section 6.2)
# ══════════════════════════════════════════════════════════════════════════

class PlattScaling:
    """
    Sigmoid (Platt) calibration — applied to XGBoost in the paper.
    Fits p_cal = sigmoid(a * logit(p) + b) on a held-out validation set.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e10, solver="lbfgs")
        self.fitted = False

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> PlattScaling:
        X = self._logit(np.asarray(y_prob)).reshape(-1, 1)
        self._lr.fit(X, np.asarray(y_true))
        self.fitted = True
        logger.info("Platt scaling fitted: a=%.4f b=%.4f",
                    self._lr.coef_[0][0], self._lr.intercept_[0])
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PlattScaling not fitted")
        X = self._logit(np.asarray(y_prob)).reshape(-1, 1)
        return self._lr.predict_proba(X)[:, 1]


class TemperatureScaling:
    """
    Single-parameter temperature scaling — applied to LSTM and Transformer.
    Minimises NLL over T with p_cal = sigmoid(logit(p) / T).

    Implemented with a scalar golden-section search so the module has no
    hard dependency on PyTorch (training/train.py uses the torch variant).
    """

    def __init__(self, t_min: float = 0.05, t_max: float = 10.0) -> None:
        self.temperature = 1.0
        self._t_min, self._t_max = t_min, t_max
        self.fitted = False

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def _nll(self, T: float, logits: np.ndarray, y: np.ndarray) -> float:
        p = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-6)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray,
            tol: float = 1e-4) -> TemperatureScaling:
        logits = self._logit(np.asarray(y_prob))
        y = np.asarray(y_true).astype(float)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = self._t_min, self._t_max
        c, d = b - gr * (b - a), a + gr * (b - a)
        while abs(b - a) > tol:
            if self._nll(c, logits, y) < self._nll(d, logits, y):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
        self.temperature = (a + b) / 2.0
        self.fitted = True
        logger.info("Temperature scaling fitted: T=%.4f", self.temperature)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScaling not fitted")
        logits = self._logit(np.asarray(y_prob))
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


class IsotonicCalibrator:
    """
    Isotonic regression — applied to the ensemble meta-learner output.
    Non-parametric and monotone; the strongest of the three but the most
    data-hungry, which is why the paper applies it only at the ensemble stage.
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(np.asarray(y_prob), np.asarray(y_true))
        self.fitted = True
        logger.info("Isotonic calibration fitted on %d samples", len(y_prob))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator not fitted")
        return self._iso.predict(np.asarray(y_prob))


# ══════════════════════════════════════════════════════════════════════════
# Distributional shift  (paper Section 6.3)
# ══════════════════════════════════════════════════════════════════════════

def detect_covariate_shift(
    x_ref: np.ndarray, x_new: np.ndarray, alpha: float = 0.01
) -> dict[int, float]:
    """
    Per-feature two-sample Kolmogorov-Smirnov test.
    Returns {feature_index: p_value} for features whose distribution has
    shifted significantly (p < alpha), signalling that recalibration is due.
    """
    shifted: dict[int, float] = {}
    for j in range(x_ref.shape[1]):
        _, p = stats.ks_2samp(x_ref[:, j], x_new[:, j])
        if p < alpha:
            shifted[j] = float(p)
    if shifted:
        logger.warning("Covariate shift detected in %d/%d features",
                       len(shifted), x_ref.shape[1])
    return shifted


def detect_label_shift(
    y_ref: np.ndarray, y_new: np.ndarray, alpha: float = 0.01
) -> tuple[bool, float]:
    """
    Two-proportion z-test on the event rate P(Y=1).
    Returns (shift_detected, p_value).
    """
    n1, n2 = len(y_ref), len(y_new)
    p1, p2 = y_ref.mean(), y_new.mean()
    p_pool = (y_ref.sum() + y_new.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return False, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return bool(p < alpha), float(p)


# ══════════════════════════════════════════════════════════════════════════
# Threshold selection  (paper Section 6.4)
# ══════════════════════════════════════════════════════════════════════════

def select_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    F1-maximising threshold. This is the procedure that yields tau = 0.72
    on the MIMIC-IV validation set in the paper.
    Returns (tau, best_f1).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    # precision_recall_curve returns one more element than thresholds
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return (float(thr[best]) if len(thr) else 0.5, float(f1[best]))


def select_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> tuple[float, float]:
    """
    Cost-sensitive threshold. A missed deterioration (FN) is far costlier
    than a false alert (FP); the default 10:1 ratio reflects the clinical
    asymmetry discussed in the paper.
    Returns (tau, minimum expected cost).
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    n_pos = float(np.sum(y_true))
    n_neg = float(len(y_true) - n_pos)
    cost = cost_fn * (1 - tpr) * n_pos + cost_fp * fpr * n_neg
    best = int(np.argmin(cost))
    return float(thr[best]), float(cost[best])


def alert_rate(y_prob: np.ndarray, tau: float) -> float:
    """Fraction of inference windows that would raise an alert at threshold tau."""
    return float((np.asarray(y_prob) >= tau).mean())
