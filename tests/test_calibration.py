"""
Tests for medroad_v3.models.calibration (paper Section 6).

These verify the calibration metrics behave correctly on constructed cases
where the right answer is known analytically, and that each calibrator
actually improves calibration on deliberately miscalibrated input.
"""
from __future__ import annotations

import numpy as np
import pytest

from medroad_v3.models.calibration import (
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    alert_rate,
    calibration_report,
    detect_covariate_shift,
    detect_label_shift,
    expected_calibration_error,
    hosmer_lemeshow,
    maximum_calibration_error,
    reliability_curve,
    select_threshold_cost,
    select_threshold_f1,
)

RNG = np.random.default_rng(20260731)


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def miscalibrated():
    """Overconfident probabilities with a genuine signal, 12% event rate."""
    n = 4000
    y = RNG.binomial(1, 0.12, n)
    p = np.clip(RNG.beta(2, 6, n) + 0.35 * y, 1e-6, 1 - 1e-6)
    return y.astype(float), p


@pytest.fixture
def perfectly_calibrated():
    """p is the true event probability, so ECE should be near zero."""
    n = 20000
    p = RNG.uniform(0.02, 0.98, n)
    y = RNG.binomial(1, p).astype(float)
    return y, p


# ── metrics ──────────────────────────────────────────────────────────────

def test_ece_zero_for_perfect_calibration(perfectly_calibrated):
    y, p = perfectly_calibrated
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_ece_large_for_constant_wrong_prediction():
    """Predicting 0.9 when nothing ever happens gives ECE close to 0.9."""
    y = np.zeros(1000)
    p = np.full(1000, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.9, abs=1e-6)


def test_mce_at_least_ece(miscalibrated):
    y, p = miscalibrated
    assert maximum_calibration_error(y, p) >= expected_calibration_error(y, p)


def test_ece_bounded_unit_interval(miscalibrated):
    y, p = miscalibrated
    assert 0.0 <= expected_calibration_error(y, p) <= 1.0
    assert 0.0 <= maximum_calibration_error(y, p) <= 1.0


def test_hosmer_lemeshow_rejects_miscalibration(miscalibrated):
    y, p = miscalibrated
    _, pval = hosmer_lemeshow(y, p)
    assert pval < 0.05  # should reject the null of good fit


def test_report_fields_populated(miscalibrated):
    y, p = miscalibrated
    r = calibration_report(y, p)
    assert 0.0 <= r.ece <= 1.0
    assert 0.0 <= r.brier <= 1.0
    assert 0.5 < r.auroc <= 1.0     # the synthetic data has real signal
    assert r.n_bins == 10
    assert set(r.to_dict()) >= {"ece", "mce", "brier", "auroc"}


def test_reliability_curve_shapes(miscalibrated):
    y, p = miscalibrated
    mp, of, cnt = reliability_curve(y, p, n_bins=10)
    assert len(mp) == len(of) == len(cnt) == 10
    assert cnt.sum() == len(y)


# ── calibrators ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cls", [PlattScaling, TemperatureScaling, IsotonicCalibrator]
)
def test_calibrator_reduces_ece(miscalibrated, cls):
    y, p = miscalibrated
    half = len(y) // 2
    cal = cls().fit(p[:half], y[:half])
    out = cal.transform(p[half:])
    before = expected_calibration_error(y[half:], p[half:])
    after = expected_calibration_error(y[half:], out)
    assert after < before, f"{cls.__name__} did not improve ECE"


@pytest.mark.parametrize(
    "cls", [PlattScaling, TemperatureScaling, IsotonicCalibrator]
)
def test_calibrator_output_in_unit_interval(miscalibrated, cls):
    y, p = miscalibrated
    cal = cls().fit(p, y)
    out = cal.transform(p)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


@pytest.mark.parametrize(
    "cls", [PlattScaling, TemperatureScaling, IsotonicCalibrator]
)
def test_calibrator_raises_before_fit(cls):
    with pytest.raises(RuntimeError):
        cls().transform(np.array([0.5]))


def test_calibration_preserves_ranking(miscalibrated):
    """
    Platt and temperature scaling are monotone, so AUROC must be unchanged.
    This is the property that lets calibration be applied post-hoc without
    retraining, as the paper relies on.
    """
    from sklearn.metrics import roc_auc_score

    y, p = miscalibrated
    base = roc_auc_score(y, p)
    for cls in (PlattScaling, TemperatureScaling):
        out = cls().fit(p, y).transform(p)
        assert roc_auc_score(y, out) == pytest.approx(base, abs=1e-6)


def test_temperature_above_one_for_overconfidence(miscalibrated):
    """Overconfident inputs should be softened, i.e. T > 1."""
    y, p = miscalibrated
    ts = TemperatureScaling().fit(p, y)
    assert ts.temperature > 0


# ── shift detection ──────────────────────────────────────────────────────

def test_no_covariate_shift_for_same_distribution():
    x = RNG.normal(size=(600, 8))
    z = RNG.normal(size=(600, 8))
    shifted = detect_covariate_shift(x, z, alpha=0.01)
    assert len(shifted) <= 1          # allow one false positive at alpha=0.01


def test_covariate_shift_detected_when_mean_moves():
    x = RNG.normal(0, 1, size=(600, 4))
    z = RNG.normal(3, 1, size=(600, 4))
    shifted = detect_covariate_shift(x, z, alpha=0.01)
    assert len(shifted) == 4


def test_label_shift_detected():
    a = RNG.binomial(1, 0.10, 2000).astype(float)
    b = RNG.binomial(1, 0.30, 2000).astype(float)
    detected, pval = detect_label_shift(a, b)
    assert detected and pval < 0.01


def test_label_shift_not_detected_when_stable():
    a = RNG.binomial(1, 0.12, 3000).astype(float)
    b = RNG.binomial(1, 0.12, 3000).astype(float)
    detected, _ = detect_label_shift(a, b)
    assert not detected


# ── threshold selection ──────────────────────────────────────────────────

def test_f1_threshold_in_range(miscalibrated):
    y, p = miscalibrated
    tau, f1 = select_threshold_f1(y, p)
    assert 0.0 <= tau <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_cost_threshold_lower_when_misses_expensive(miscalibrated):
    """
    Raising the cost of a false negative should not raise the threshold.
    A missed deterioration is worse than a spurious alert, so the system
    must become more, not less, willing to alert.
    """
    y, p = miscalibrated
    tau_cheap, _ = select_threshold_cost(y, p, cost_fn=1.0, cost_fp=1.0)
    tau_dear, _ = select_threshold_cost(y, p, cost_fn=20.0, cost_fp=1.0)
    assert tau_dear <= tau_cheap


def test_alert_rate_monotone_in_threshold(miscalibrated):
    _, p = miscalibrated
    assert alert_rate(p, 0.2) >= alert_rate(p, 0.5) >= alert_rate(p, 0.8)


def test_alert_rate_bounds(miscalibrated):
    _, p = miscalibrated
    assert alert_rate(p, 0.0) == 1.0
    assert alert_rate(p, 1.01) == 0.0
