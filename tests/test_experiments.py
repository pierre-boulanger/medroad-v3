"""
Tests for the experiment suite.

These check the statistical machinery against cases with known answers, since
an ablation is only worth running if its significance tests are trustworthy.
"""
from __future__ import annotations

from datetime import UTC

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from medroad_v3.experiments.ablation import run as run_ablation
from medroad_v3.experiments.ablation import synthetic_scores
from medroad_v3.experiments.baselines import mews_score, news_score
from medroad_v3.experiments.common import (
    alerts_per_patient_shift,
    auroc_ci,
    delong_test,
    latex_table,
    paired_bootstrap_diff,
    threshold_at_sensitivity,
)
from medroad_v3.experiments.rpm_degradation import (
    column_mask,
    observable_icu_features,
    unmapped_channels,
)
from medroad_v3.experiments.synth import feature_names, synthetic_cohort

RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def cohort():
    n = 3000
    y = RNG.binomial(1, 0.09, n).astype(float)
    strong = np.clip(RNG.beta(2, 6, n) + 0.33 * y, 1e-6, 1 - 1e-6)
    weak = np.clip(RNG.beta(2, 6, n) + 0.12 * y, 1e-6, 1 - 1e-6)
    return y, strong, weak


# ── DeLong ───────────────────────────────────────────────────────────────

def test_delong_identical_scores_gives_p_one(cohort):
    y, s, _ = cohort
    a, b, p = delong_test(y, s, s)
    assert a == pytest.approx(b)
    assert p == pytest.approx(1.0, abs=1e-6)

def test_delong_detects_real_difference(cohort):
    y, strong, weak = cohort
    a, b, p = delong_test(y, strong, weak)
    assert a > b and p < 0.01

def test_delong_auc_matches_sklearn(cohort):
    y, s, _ = cohort
    a, _, _ = delong_test(y, s, s)
    assert a == pytest.approx(roc_auc_score(y, s), abs=1e-6)

def test_delong_symmetric_in_pvalue(cohort):
    y, a, b = cohort
    assert delong_test(y, a, b)[2] == pytest.approx(delong_test(y, b, a)[2], abs=1e-9)


# ── Bootstrap ────────────────────────────────────────────────────────────

def test_bootstrap_interval_brackets_point_estimate(cohort):
    y, s, _ = cohort
    e = auroc_ci(y, s, n_boot=200)
    assert e.lo <= e.value <= e.hi

def test_bootstrap_requires_both_classes():
    with pytest.raises(ValueError):
        auroc_ci(np.zeros(50), RNG.random(50), n_boot=10)

def test_paired_diff_excludes_zero_for_real_gap(cohort):
    y, strong, weak = cohort
    d = paired_bootstrap_diff(y, strong, weak, roc_auc_score, n_boot=300)
    assert d.lo > 0

def test_paired_diff_includes_zero_for_identical(cohort):
    y, s, _ = cohort
    d = paired_bootstrap_diff(y, s, s, roc_auc_score, n_boot=100)
    assert d.lo <= 0 <= d.hi


# ── Operating points ─────────────────────────────────────────────────────

def test_threshold_achieves_requested_sensitivity(cohort):
    y, s, _ = cohort
    for target in (0.7, 0.8, 0.9):
        tau = threshold_at_sensitivity(y, s, target)
        assert (s[y == 1] >= tau).mean() >= target - 0.02

def test_higher_sensitivity_lowers_threshold(cohort):
    y, s, _ = cohort
    assert (threshold_at_sensitivity(y, s, 0.9)
            <= threshold_at_sensitivity(y, s, 0.7))

def test_alert_rate_scales_with_window_count(cohort):
    _, s, _ = cohort
    assert alerts_per_patient_shift(s, 0.5, 96) == pytest.approx(
        2 * alerts_per_patient_shift(s, 0.5, 48))


# ── Synthetic cohort ─────────────────────────────────────────────────────

def test_synthetic_cohort_matches_feature_dimension():
    X, y, names = synthetic_cohort(n=500)
    assert X.shape[1] == len(names) == 86
    assert len(y) == 500

def test_synthetic_features_in_unit_interval():
    X, _, _ = synthetic_cohort(n=300)
    assert X.min() >= 0.0 and X.max() <= 1.0

def test_synthetic_cohort_carries_signal():
    X, y, names = synthetic_cohort(n=4000)
    i = names.index("lab_lactate")
    assert X[y == 1, i].mean() > X[y == 0, i].mean()


# ── RPM masking ──────────────────────────────────────────────────────────

def test_empty_kit_retains_all_columns():
    names = feature_names()
    assert column_mask(names, ()).all()

def test_kit_mask_drops_some_columns():
    names = feature_names()
    m = column_mask(names, ("Pulse oximeter",))
    assert 0 < m.sum() < len(names)

def test_aggregate_metadata_always_retained():
    """Coverage and temporal context survive any kit: a remote deployment
    still knows how complete its own data is and what time it is."""
    names = feature_names()
    m = column_mask(names, ("Pulse oximeter",))
    for i, n in enumerate(names):
        if n.startswith(("meta_", "hour_")):
            assert m[i], n


def test_missingness_follows_its_channel():
    """
    A missingness indicator for a channel the kit cannot observe must be
    dropped. Kept, it is constant in deployment but varies in the ICU training
    data, where it encodes which patients were worked up and how, which is a
    documentation fingerprint rather than physiology.
    """
    names = feature_names()
    m = column_mask(names, ("Pulse oximeter",))
    idx = {n: i for i, n in enumerate(names)}
    assert m[idx["miss_vital_spo2"]]            # pulse oximeter measures this
    assert not m[idx["miss_lab_troponin_i"]]    # it does not measure this

def test_larger_kit_observes_at_least_as_much():
    small = observable_icu_features(("Pulse oximeter",))
    large = observable_icu_features(("Pulse oximeter", "BP cuff"))
    assert small <= large

def test_weight_is_mapped_to_a_feature_column():
    """
    Weight was added to the vector precisely because the Discussion calls daily
    weight the primary heart-failure decompensation signal. A smart scale must
    therefore be able to contribute.
    """
    assert "weight" not in unmapped_channels(feature_names())


def test_weight_trend_features_present():
    names = feature_names()
    assert "delta_weight_24h" in names
    assert "delta_weight_72h" in names


def test_unmapped_channels_still_reports_remaining_gaps():
    """
    Glucose and cardiac rhythm remain unmapped, so a CGM and an ambulatory ECG
    still cannot contribute. The experiment must keep surfacing this.
    """
    missing = unmapped_channels(feature_names())
    assert "glucose" in missing
    assert "cardiac_rhythm" in missing


# ── Rule-based baselines ─────────────────────────────────────────────────

def test_news_and_mews_are_non_negative():
    X, _, names = synthetic_cohort(n=400)
    assert news_score(X, names).min() >= 0
    assert mews_score(X, names).min() >= 0

def test_news_penalises_hypoxia():
    X, _, names = synthetic_cohort(n=200)
    i = names.index("vital_spo2_mean")
    healthy = X.copy()
    healthy[:, i] = 1.0      # top of SpO2 range
    hypoxic = X.copy()
    hypoxic[:, i] = 0.0      # bottom of range
    assert news_score(hypoxic, names).mean() > news_score(healthy, names).mean()


# ── End-to-end ───────────────────────────────────────────────────────────

def test_ablation_runs_and_covers_all_variants():
    y, scores = synthetic_scores(n=1500)
    res, _, _ = run_ablation(scores, y, n_boot=60)
    assert "ensemble (full)" in res
    assert sum(k.startswith("ensemble -") for k in res) == 3
    for name, r in res.items():
        assert 0.0 <= r["auroc"]["value"] <= 1.0
        if name != "ensemble (full)":
            assert 0.0 <= r["delong_p_vs_full"] <= 1.0


def test_latex_table_is_wellformed():
    t = latex_table([["a", "1"]], ["X", "Y"], "cap", "tab:t")
    assert t.count(r"\begin{tabular}") == t.count(r"\end{tabular}") == 1
    assert r"\label{tab:t}" in t


# ── Vital statistics ─────────────────────────────────────────────────────

def test_four_statistics_per_vital():
    names = feature_names()
    for st in ("mean", "std", "min", "rate"):
        assert f"vital_spo2_{st}" in names


def test_min_and_rate_distinguish_a_desaturation():
    """
    A falling SpO2 series and a flat one with the same mean must not produce
    the same features. This is the case the paper cites to justify retaining
    the minimum rather than the maximum.
    """
    from datetime import datetime, timedelta

    from medroad_v3.features.engineering import FeatureEngineer

    now = datetime.now(UTC)

    def obs(val, offs):
        return {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "59408-5"}]},
            "subject": {"reference": "Patient/T"},
            "effectiveDateTime": (now - timedelta(seconds=offs)).isoformat(),
            "valueQuantity": {"value": val},
        }

    fe = FeatureEngineer()
    falling = [obs(v, s) for v, s in [(98, 240), (96, 180), (93, 120), (90, 60), (88, 0)]]
    flat = [obs(93, s) for s in (240, 180, 120, 60, 0)]

    _, df = fe.build_vector(falling, [], None, [], None)
    _, dl = fe.build_vector(flat, [], None, [], None)

    assert df["vital_spo2_mean"] == pytest.approx(dl["vital_spo2_mean"], abs=1e-6)
    assert df["vital_spo2_min"] < dl["vital_spo2_min"]
    assert df["vital_spo2_rate"] < 0 and dl["vital_spo2_rate"] == pytest.approx(0.0)


# ── Feature bounds ───────────────────────────────────────────────────────

def test_all_engineered_features_bounded():
    """
    Every engineered feature must lie in [-1, 1]. Two defects escaped through
    this gap: documentation lag computed against wall-clock time instead of the
    window end, and a coverage divisor left at 8 after a ninth vital was added.
    Both left tree models untouched while wrecking the neural ones.
    """
    from datetime import datetime, timedelta

    from medroad_v3.features.engineering import FeatureEngineer

    now = datetime.now(UTC)

    def obs(code, val, offs=0):
        return {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
            "effectiveDateTime": (now - timedelta(seconds=offs)).isoformat(),
            "valueQuantity": {"value": val},
        }

    fe = FeatureEngineer()
    window = [obs(c, v, o) for c, v, o in [
        ("8867-4", 88, 240), ("8867-4", 94, 60),
        ("59408-5", 95, 180), ("55284-4", 120, 120),
        ("29463-7", 81, 60), ("8310-5", 37.1, 30),
    ]]
    labs = [obs("2532-0", 2.1, 3600), obs("10839-9", 0.4, 7200)]

    vec, named = fe.build_vector(
        window, labs, None, [], now - timedelta(days=3),
        now=now, prior_weights=(80.0, 78.5),
    )
    assert len(vec) == 86
    for name, value in named.items():
        assert -1.0001 <= value <= 1.0001, f"{name} = {value}"


def test_coverage_divisor_tracks_channel_count():
    """Coverage must be a fraction, never above 1, however many channels exist."""
    from datetime import datetime

    from medroad_v3.features.engineering import LAB_NAMES, VITAL_NAMES, FeatureEngineer

    now = datetime.now(UTC)
    fe = FeatureEngineer()
    allv = [{
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": c}]},
        "effectiveDateTime": now.isoformat(),
        "valueQuantity": {"value": 1.0},
    } for c in __import__("medroad_v3.config", fromlist=["x"]).LOINC_VITALS.values()]

    _, named = fe.build_vector(allv, [], None, [], None, now=now)
    assert named["meta_vital_cov"] == pytest.approx(1.0), len(VITAL_NAMES)
    assert named["meta_lab_cov"] == pytest.approx(0.0), len(LAB_NAMES)
