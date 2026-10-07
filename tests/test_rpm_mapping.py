"""
Tests for medroad_v3.rpm.mapping (paper Section 3.6, Table 3).

These check that the encoded mapping is internally consistent and that the
window-sizing and coverage logic behaves as the paper describes.
"""
from __future__ import annotations

from medroad_v3 import config
from medroad_v3.rpm.mapping import (
    BY_ICU_FEATURE,
    BY_LOINC,
    DEVICE_PROFILES,
    RPM_MAPPING,
    DeviceProfile,
    Fidelity,
    describe_deployment,
    expected_coverage,
    recommended_window_seconds,
    rpm_loinc_allowlist,
    rpm_subscription_criteria,
    unavailable_features,
)

HF_KIT = [p for p in DEVICE_PROFILES
          if p.name in ("Pulse oximeter", "BP cuff", "Smart scale")]


# ── table integrity ──────────────────────────────────────────────────────

def test_mapping_is_not_empty():
    assert len(RPM_MAPPING) >= 10


def test_loinc_codes_unique():
    codes = [f.loinc for f in RPM_MAPPING]
    assert len(codes) == len(set(codes))


def test_icu_feature_names_unique():
    names = [f.icu_feature for f in RPM_MAPPING]
    assert len(names) == len(set(names))


def test_indexes_agree_with_table():
    assert len(BY_ICU_FEATURE) == len(RPM_MAPPING)
    assert len(BY_LOINC) == len(RPM_MAPPING)
    for f in RPM_MAPPING:
        assert BY_ICU_FEATURE[f.icu_feature] is f
        assert BY_LOINC[f.loinc] is f


def test_loinc_format_plausible():
    """LOINC codes are digits with a single check digit after a hyphen."""
    for f in RPM_MAPPING:
        body, _, check = f.loinc.partition("-")
        assert body.isdigit(), f.loinc
        assert check.isdigit() and len(check) == 1, f.loinc


def test_all_fidelities_are_valid_enum():
    for f in RPM_MAPPING:
        assert isinstance(f.fidelity, Fidelity)


def test_usable_matches_fidelity():
    for f in RPM_MAPPING:
        assert f.usable == (f.fidelity is not Fidelity.UNAVAILABLE)


def test_unavailable_features_have_no_cadence():
    """A feature with no wearable analogue cannot have a sampling cadence."""
    for f in RPM_MAPPING:
        if not f.usable:
            assert f.cadence_s is None


def test_usable_features_have_cadence():
    for f in RPM_MAPPING:
        if f.usable:
            assert f.cadence_s is not None and f.cadence_s > 0


def test_known_unavailable_features():
    """Troponin and ventilator pressure have no wearable equivalent."""
    missing = set(unavailable_features())
    assert "troponin_i" in missing
    assert "ventilator_pressure" in missing


def test_core_heart_failure_signals_are_direct():
    """Weight, SpO2 and systolic BP drive HF decompensation detection."""
    for name in ("weight", "spo2", "systolic_bp"):
        assert BY_ICU_FEATURE[name].fidelity is Fidelity.DIRECT


# ── device profiles ──────────────────────────────────────────────────────

def test_device_profiles_reference_known_loincs():
    known = {f.loinc for f in RPM_MAPPING}
    for prof in DEVICE_PROFILES:
        for code in prof.loincs:
            assert code in known, f"{prof.name} publishes unknown LOINC {code}"


def test_device_profiles_have_positive_cadence():
    for prof in DEVICE_PROFILES:
        assert prof.cadence_s > 0


# ── window sizing ────────────────────────────────────────────────────────

def test_window_defaults_to_icu_when_no_devices():
    assert recommended_window_seconds([]) == config.WINDOW_SECONDS


def test_window_never_below_icu_default():
    assert recommended_window_seconds(list(DEVICE_PROFILES)) >= config.WINDOW_SECONDS


def test_daily_device_forces_daily_window():
    """
    A smart scale reports once per day, so any kit containing it needs a
    24-hour window or nearly every window will be empty.
    """
    scale = [p for p in DEVICE_PROFILES if p.name == "Smart scale"]
    assert recommended_window_seconds(scale) == 86400


def test_fast_only_kit_keeps_short_window():
    fast = [p for p in DEVICE_PROFILES if p.name in ("Pulse oximeter", "CGM")]
    assert recommended_window_seconds(fast) == 300


# ── coverage ─────────────────────────────────────────────────────────────

def test_coverage_zero_without_devices():
    assert expected_coverage([], 3600) == 0.0


def test_coverage_in_unit_interval():
    for kit in ([], HF_KIT, list(DEVICE_PROFILES)):
        cov = expected_coverage(kit, 86400)
        assert 0.0 <= cov <= 1.0


def test_more_devices_never_reduce_coverage():
    w = 86400
    assert expected_coverage(list(DEVICE_PROFILES), w) >= expected_coverage(HF_KIT, w)


def test_short_window_excludes_slow_devices():
    """A five-minute window cannot capture a once-daily weight reading."""
    assert expected_coverage(list(DEVICE_PROFILES), 300) < \
           expected_coverage(list(DEVICE_PROFILES), 86400)


# ── subscription criteria ────────────────────────────────────────────────

def test_allowlist_excludes_unavailable():
    allow = rpm_loinc_allowlist()
    for f in RPM_MAPPING:
        assert (f.loinc in allow) == f.usable


def test_subscription_criteria_is_valid_fhir_search():
    crit = rpm_subscription_criteria()
    assert crit.startswith("Observation?code=http://loinc.org|")
    codes = crit.split("|", 1)[1].split(",")
    assert set(codes) == rpm_loinc_allowlist()


# ── deployment summary ───────────────────────────────────────────────────

def test_describe_deployment_reports_expected_keys():
    r = describe_deployment(HF_KIT)
    assert set(r) >= {
        "devices", "window_seconds", "expected_coverage",
        "unavailable_features", "n_direct", "n_approximate",
    }


def test_describe_deployment_counts_match_table():
    r = describe_deployment(list(DEVICE_PROFILES))
    assert r["n_direct"] == sum(1 for f in RPM_MAPPING
                                if f.fidelity is Fidelity.DIRECT)
    assert r["n_approximate"] == sum(1 for f in RPM_MAPPING
                                     if f.fidelity is Fidelity.APPROXIMATE)


def test_describe_deployment_warns_on_low_coverage(caplog):
    """A single slow device should trigger the recalibration warning."""
    thin = [DeviceProfile("Scale only", ("29463-7",), 86400)]
    with caplog.at_level("WARNING"):
        describe_deployment(thin)
    assert any("coverage" in rec.message.lower() for rec in caplog.records)
