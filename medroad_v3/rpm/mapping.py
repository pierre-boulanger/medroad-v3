"""
MedROAD V3 — Remote Patient Monitoring Extension
Implements Section 3.6 of the paper ("Extension to Remote Patient Monitoring
Sensor Streams") and the ICU-to-wearable feature mapping of Table 3.

The central claim the paper makes is that the FHIR R4 Observation resource is
device-agnostic, so the ingestion, brokering, and inference layers require no
modification to accept wearable streams. What *does* change is:

  1. which LOINC codes are expected to be present,
  2. the observation cadence (and therefore the window width), and
  3. the imputation policy for ICU-only features that have no wearable analogue.

This module encodes exactly those three things and nothing else, so that the
in-hospital pipeline in medroad_v3.features.engineering stays untouched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from medroad_v3 import config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Mapping fidelity  (Table 3: bold = direct, italic = approximate)
# ══════════════════════════════════════════════════════════════════════════

class Fidelity(StrEnum):
    DIRECT      = "direct"       # same physiological quantity, same units
    APPROXIMATE = "approximate"  # functional equivalent, different method
    UNAVAILABLE = "unavailable"  # no wearable analogue; impute or drop


@dataclass(frozen=True)
class RPMFeature:
    """One row of Table 3."""
    icu_feature:   str
    loinc:         str
    wearable:      str
    fidelity:      Fidelity
    cadence_s:     int | None = None   # typical seconds between observations
    note:          str = ""

    @property
    def usable(self) -> bool:
        return self.fidelity is not Fidelity.UNAVAILABLE


# ══════════════════════════════════════════════════════════════════════════
# Table 3 — ICU feature to wearable analogue
# ══════════════════════════════════════════════════════════════════════════

RPM_MAPPING: tuple[RPMFeature, ...] = (
    RPMFeature("spo2",             "59408-5", "Pulse oximeter (SpO2)",
               Fidelity.DIRECT,      cadence_s=300),
    RPMFeature("heart_rate",       "8867-4",  "Wrist PPG heart rate",
               Fidelity.DIRECT,      cadence_s=60),
    RPMFeature("systolic_bp",      "8480-6",  "Home oscillometric cuff",
               Fidelity.DIRECT,      cadence_s=43200,
               note="Typically twice daily"),
    RPMFeature("diastolic_bp",     "8462-4",  "Home oscillometric cuff",
               Fidelity.DIRECT,      cadence_s=43200,
               note="Reported alongside systolic by the same device"),
    RPMFeature("weight",           "29463-7", "Smart scale (daily weight)",
               Fidelity.DIRECT,      cadence_s=86400,
               note="Primary heart-failure decompensation signal"),
    RPMFeature("glucose",          "2345-7",  "CGM interstitial glucose",
               Fidelity.DIRECT,      cadence_s=300),
    RPMFeature("temperature",      "8310-5",  "Wearable skin temperature",
               Fidelity.APPROXIMATE, cadence_s=300,
               note="Skin temp is offset from core temp"),
    RPMFeature("respiratory_rate", "9279-1",  "Accelerometer breathing rate",
               Fidelity.APPROXIMATE, cadence_s=300),
    RPMFeature("map",              "8478-0",  "MAP estimate from cuff",
               Fidelity.APPROXIMATE, cadence_s=43200,
               note="Derived, not arterial-line measured"),
    RPMFeature("cardiac_rhythm",   "8884-9",  "Ambulatory ECG patch",
               Fidelity.APPROXIMATE, cadence_s=86400,
               note="Rhythm classification, not continuous waveform"),
    RPMFeature("troponin_i",       "42757-5", "None",
               Fidelity.UNAVAILABLE,
               note="Serial troponin has no wearable analogue; zero-impute"),
    RPMFeature("ventilator_pressure", "76248-8", "None (replace with AHI)",
               Fidelity.UNAVAILABLE,
               note="Substitute apnoea-hypopnoea index where available"),
)

BY_ICU_FEATURE: dict[str, RPMFeature] = {f.icu_feature: f for f in RPM_MAPPING}
BY_LOINC:       dict[str, RPMFeature] = {f.loinc: f for f in RPM_MAPPING}


# ══════════════════════════════════════════════════════════════════════════
# Device profiles
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeviceProfile:
    """A wearable device class and the LOINC codes it publishes."""
    name:      str
    loincs:    tuple[str, ...]
    cadence_s: int

DEVICE_PROFILES: tuple[DeviceProfile, ...] = (
    DeviceProfile("Pulse oximeter",  ("59408-5", "8867-4"),          300),
    DeviceProfile("CGM",             ("2345-7",),                    300),
    DeviceProfile("Ambulatory ECG",  ("8884-9", "8867-4"),         86400),
    DeviceProfile("BP cuff",         ("8480-6", "8462-4", "8478-0"), 43200),
    DeviceProfile("Smart scale",     ("29463-7",),                  86400),
)


# ══════════════════════════════════════════════════════════════════════════
# Window sizing  (Section 3.6: sparse streams need wider windows)
# ══════════════════════════════════════════════════════════════════════════

def recommended_window_seconds(profiles: list[DeviceProfile]) -> int:
    """
    The five-minute ICU window assumes dense vital-sign coverage. For an RPM
    cohort the window must be at least as wide as the slowest device cadence,
    or most windows will be empty. Returns the recommended window width.
    """
    if not profiles:
        return config.WINDOW_SECONDS
    slowest = max(p.cadence_s for p in profiles)
    # One full cycle of the slowest device, floored at the ICU default.
    return max(slowest, config.WINDOW_SECONDS)


def expected_coverage(profiles: list[DeviceProfile], window_s: int) -> float:
    """
    Fraction of the 55-element vector expected to be populated by a given
    device set over a given window. Feeds the meta_vital_cov / meta_lab_cov
    features and gives an a-priori estimate of imputation load.
    """
    available = {lo for p in profiles if p.cadence_s <= window_s for lo in p.loincs}
    mappable  = {f.loinc for f in RPM_MAPPING if f.usable}
    if not mappable:
        return 0.0
    return len(available & mappable) / len(mappable)


# ══════════════════════════════════════════════════════════════════════════
# Feature policy for RPM deployment
# ══════════════════════════════════════════════════════════════════════════

def unavailable_features() -> list[str]:
    """ICU features with no wearable analogue — these must be imputed."""
    return [f.icu_feature for f in RPM_MAPPING if not f.usable]


def rpm_loinc_allowlist() -> set[str]:
    """LOINC codes an RPM deployment should subscribe to."""
    return {f.loinc for f in RPM_MAPPING if f.usable}


def rpm_subscription_criteria() -> str:
    """
    FHIR Subscription criteria string restricted to RPM-relevant LOINC codes.
    Used in place of the broad category=vital-signs criteria when the
    deployment is remote-only.
    """
    codes = ",".join(sorted(rpm_loinc_allowlist()))
    return f"Observation?code=http://loinc.org|{codes}"


def describe_deployment(profiles: list[DeviceProfile]) -> dict:
    """
    Summarise what an RPM deployment with the given devices can and cannot do.
    Intended for logging at startup so the operator sees the imputation burden
    before any inference runs.
    """
    window = recommended_window_seconds(profiles)
    cov    = expected_coverage(profiles, window)
    report = {
        "devices":              [p.name for p in profiles],
        "window_seconds":       window,
        "expected_coverage":    round(cov, 3),
        "unavailable_features": unavailable_features(),
        "n_direct":             sum(1 for f in RPM_MAPPING
                                    if f.fidelity is Fidelity.DIRECT),
        "n_approximate":        sum(1 for f in RPM_MAPPING
                                    if f.fidelity is Fidelity.APPROXIMATE),
    }
    logger.info(
        "RPM deployment: %d devices, window=%ds, expected coverage=%.1f%%",
        len(profiles), window, cov * 100,
    )
    if cov < 0.5:
        logger.warning(
            "Expected feature coverage below 50%% — model was trained on dense "
            "ICU data and will rely heavily on imputation. Recalibration on an "
            "RPM cohort is strongly recommended (see paper Section 6.3)."
        )
    return report


__all__ = [
    "Fidelity", "RPMFeature", "DeviceProfile",
    "RPM_MAPPING", "BY_ICU_FEATURE", "BY_LOINC", "DEVICE_PROFILES",
    "recommended_window_seconds", "expected_coverage",
    "unavailable_features", "rpm_loinc_allowlist",
    "rpm_subscription_criteria", "describe_deployment",
]
