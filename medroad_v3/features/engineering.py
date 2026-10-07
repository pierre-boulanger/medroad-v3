"""
MedROAD V3 — Feature Engineering
Builds the 55-element feature vector from FHIR Observation resources.

Vector layout (all 55 elements):
  [0:8]   — 8 vital-sign values (mean over 5-min window)
  [8:20]  — 12 laboratory values (most recent within 24h)
  [20:28] — 8 metadata / coverage features
  [28:36] — 8 vital-sign missingness indicators (binary)
  [36:48] — 12 lab missingness indicators (binary)
  [48:51] — 3 temporal delta features (ΔTropI, ΔBNP, ΔLactate)
  [51:53] — 2 physiological interaction scores (SIG, BCR)
  [53:55] — 2 temporal context features (hour-of-day sin/cos)
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

from medroad_v3 import config

logger = logging.getLogger(__name__)

# Reference ranges for normalisation (min, max)
_VITAL_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate":       (20.0, 250.0),
    "systolic_bp":      (50.0, 250.0),
    "diastolic_bp":     (20.0, 150.0),
    "spo2":             (60.0,  100.0),
    "respiratory_rate": (4.0,   60.0),
    "temperature":      (32.0,  42.0),
    "weight":           (30.0,  250.0),
    "gcs":              (3.0,   15.0),
    "pain_score":       (0.0,   10.0),
}

_LAB_RANGES: dict[str, tuple[float, float]] = {
    "troponin_i":  (0.0,     5.0),
    "bnp":         (0.0,  2000.0),
    "creatinine":  (0.0,    20.0),
    "potassium":   (1.0,    10.0),
    "sodium":      (100.0, 180.0),
    "bicarbonate": (5.0,    45.0),
    "pco2":        (10.0,  120.0),
    "ph":          (6.8,    7.8),
    "lactate":     (0.0,    20.0),
    "hematocrit":  (10.0,   70.0),
    "wbc":         (0.0,    50.0),
    "magnesium":   (0.0,     5.0),
}

VITAL_NAMES = list(config.LOINC_VITALS.keys())   # ordered
LAB_NAMES   = list(config.LOINC_LABS.keys())      # ordered


def _extract_value(obs: dict[str, Any]) -> float | None:
    """Extract numeric value from a FHIR Observation."""
    if "valueQuantity" in obs:
        return obs["valueQuantity"].get("value")
    if "valueInteger" in obs:
        return float(obs["valueInteger"])
    if "valueString" in obs:
        try:
            return float(obs["valueString"])
        except (ValueError, TypeError):
            return None
    # Component observations (e.g. BP panel)
    components = obs.get("component", [])
    if components:
        vals = []
        for c in components:
            v = c.get("valueQuantity", {}).get("value")
            if v is not None:
                vals.append(float(v))
        return vals[0] if vals else None
    return None


def _obs_time(obs: dict[str, Any]) -> datetime:
    ts = obs.get("effectiveDateTime") or obs.get("issued", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        dt = datetime.now(UTC)
    return dt


def _normalise(value: float, lo: float, hi: float) -> float:
    """Clip + min-max normalise to [0, 1]."""
    clamped = max(lo, min(hi, value))
    rng = hi - lo
    return (clamped - lo) / rng if rng > 0 else 0.0


def _get_loinc_code(obs: dict[str, Any]) -> str | None:
    """Extract primary LOINC code from an Observation."""
    for coding in obs.get("code", {}).get("coding", []):
        if coding.get("system", "").startswith("http://loinc.org"):
            return coding.get("code")
    return None


class FeatureEngineer:
    """
    Converts a dict of FHIR Observations for one patient into a
    normalised 55-element numpy vector ready for model inference.
    """

    def __init__(self) -> None:
        self._loinc_to_vital = {v: k for k, v in config.LOINC_VITALS.items()}
        self._loinc_to_lab   = {v: k for k, v in config.LOINC_LABS.items()}

    def build_vector(
        self,
        window_obs: list[dict[str, Any]],       # vitals in last 5 min
        lab_obs_24h: list[dict[str, Any]],       # labs in last 24 h
        prior_labs: dict[str, float] | None,     # labs from previous 24-h period (for deltas)
        med_requests: list[dict[str, Any]],      # active MedicationRequests
        encounter_start: datetime | None,        # for ICU LOS
        now: datetime | None = None,
        prior_weights: tuple[float | None, float | None] = (None, None),
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Returns:
            vector  — shape (55,) float32
            names   — dict mapping position-index str to feature name
        """
        if now is None:
            now = datetime.now(UTC)

        # ── Group window observations by LOINC ───────────────────────────────
        vital_window: dict[str, list[tuple[float, datetime]]] = {
            n: [] for n in VITAL_NAMES
        }
        for obs in window_obs:
            code = _get_loinc_code(obs)
            if code and code in self._loinc_to_vital:
                v = _extract_value(obs)
                if v is not None:
                    vital_window[self._loinc_to_vital[code]].append(
                        (float(v), _obs_time(obs))
                    )
        for samples in vital_window.values():
            samples.sort(key=lambda t: t[1])

        # ── Group 24-h lab observations ───────────────────────────────────────
        lab_latest: dict[str, tuple[float, datetime]] = {}
        for obs in lab_obs_24h:
            code = _get_loinc_code(obs)
            if code and code in self._loinc_to_lab:
                name = self._loinc_to_lab[code]
                v = _extract_value(obs)
                if v is not None:
                    t = _obs_time(obs)
                    if name not in lab_latest or t > lab_latest[name][1]:
                        lab_latest[name] = (float(v), t)

        # ── Vital statistics: four per parameter ─────────────────────────────
        # Mean, standard deviation, minimum and rate of change are computed
        # over the window. The minimum is retained rather than the maximum
        # because deterioration in cardiac patients typically presents as a
        # transient trough, such as an SpO2 desaturation episode, which a mean
        # over the same window would mask.
        vital_vals_raw: dict[str, float] = {}
        vital_missing: dict[str, bool] = {}
        vital_norm: list[float] = []

        for name in VITAL_NAMES:
            lo, hi = _VITAL_RANGES[name]
            samples = vital_window.get(name, [])
            values = [v for v, _ in samples]

            if values:
                mean_v = float(np.mean(values))
                std_v = float(np.std(values))
                min_v = float(np.min(values))
                if len(samples) >= 2:
                    dt = (samples[-1][1] - samples[0][1]).total_seconds()
                    rate = (values[-1] - values[0]) / dt if dt > 0 else 0.0
                else:
                    rate = 0.0
                vital_missing[name] = False
            else:
                mean_v = (lo + hi) / 2.0      # median imputation
                std_v = 0.0
                min_v = mean_v
                rate = 0.0
                vital_missing[name] = True

            vital_vals_raw[name] = mean_v
            span = max(hi - lo, 1e-6)
            vital_norm += [
                _normalise(mean_v, lo, hi),
                float(np.clip(std_v / (span / 4.0), 0.0, 1.0)),
                _normalise(min_v, lo, hi),
                # Rate is normalised against a full-range excursion across the
                # window and signed, so a fall and a rise of equal magnitude are
                # distinguishable rather than collapsed.
                float(np.clip(
                    rate * config.WINDOW_SECONDS / span, -1.0, 1.0)),
            ]

        # ── [8:20] Lab values (normalised, most-recent in 24h) ───────────────
        lab_vals_raw: dict[str, float] = {}
        lab_missing:  dict[str, bool]  = {}
        for name in LAB_NAMES:
            if name in lab_latest:
                lab_vals_raw[name] = lab_latest[name][0]
                lab_missing[name]  = False
            else:
                lo, hi = _LAB_RANGES[name]
                lab_vals_raw[name] = (lo + hi) / 2.0
                lab_missing[name]  = True

        lab_norm = [
            _normalise(lab_vals_raw[n], *_LAB_RANGES[n]) for n in LAB_NAMES
        ]

        # ── [20:28] Metadata features ─────────────────────────────────────────
        # Divisors derive from the name lists rather than being written in.
        # Hardcoding 8 here is what made vital coverage report 1.125 once weight
        # was added as a ninth vital: the count moved and the denominator did not.
        vital_cov = sum(1 for m in vital_missing.values() if not m) / len(VITAL_NAMES)
        lab_cov   = sum(1 for m in lab_missing.values()  if not m) / len(LAB_NAMES)
        abg_present    = float(not (
            lab_missing.get("ph", True) or
            lab_missing.get("pco2", True) or
            lab_missing.get("lactate", True)
        ))
        # doc_lag: normalised age of most recent vital (0=just now, 1=5 min ago)
        recent_times = [_obs_time(o) for o in window_obs]
        if recent_times:
            most_recent = max(recent_times)
            lag_s = (now - most_recent).total_seconds()
            # Clamped to [0, 1]. A negative lag means an observation timestamped
            # after the window end, which indicates a misaligned clock rather
            # than a real measurement, and must not propagate as a feature.
            doc_lag = float(np.clip(lag_s / config.WINDOW_SECONDS, 0.0, 1.0))
        else:
            doc_lag = 1.0

        vital_recency = doc_lag  # same concept for vital-sign recency

        # med_burden: count of active meds, capped at 20, normalised
        med_burden = min(len(med_requests), 20) / 20.0

        # iv_drip: any continuous infusion active (route = IV infusion)
        iv_drip = float(any(
            "infus" in str(mr.get("dosageInstruction", [{}])[0]
                           .get("route", {})
                           .get("text", "")).lower()
            for mr in med_requests if mr.get("dosageInstruction")
        ))

        # icu_los: days since encounter start, log-normalised (cap 30 days)
        if encounter_start:
            los_days = max(0.0, (now - encounter_start).total_seconds() / 86400.0)
            icu_los = float(np.clip(
                math.log1p(min(los_days, 30.0)) / math.log1p(30.0), 0.0, 1.0))
        else:
            icu_los = 0.0

        metadata = [vital_cov, lab_cov, abg_present, doc_lag,
                    vital_recency, med_burden, iv_drip, icu_los]

        # ── [28:36] Vital missingness indicators ─────────────────────────────
        vital_miss_vec = [float(vital_missing[n]) for n in VITAL_NAMES]

        # ── [36:48] Lab missingness indicators ───────────────────────────────
        lab_miss_vec = [float(lab_missing[n]) for n in LAB_NAMES]

        # ── [48:51] Temporal delta features ──────────────────────────────────
        # ΔTropI, ΔBNP, ΔLactate (current - prior, normalised by range)
        delta_names = ["troponin_i", "bnp", "lactate"]
        deltas: list[float] = []
        for name in delta_names:
            current = lab_vals_raw.get(name, 0.0)
            prior   = (prior_labs or {}).get(name, current)
            lo, hi  = _LAB_RANGES[name]
            delta_norm = (current - prior) / max(hi - lo, 1e-6)
            deltas.append(float(np.clip(delta_norm, -1.0, 1.0)))

        # ── [51:53] Physiological interaction scores ──────────────────────────
        # SIG = Na + K - Cl - HCO3  (approximated; Cl not in panel → use Na+K-HCO3-12)
        na  = lab_vals_raw.get("sodium",      140.0)
        k   = lab_vals_raw.get("potassium",     4.0)
        hco3= lab_vals_raw.get("bicarbonate",  24.0)
        sig = (na + k - hco3 - 12.0) / 30.0  # normalised, typical range [-0.5, 2]
        sig_norm = float(np.clip(sig, -1.0, 2.0) / 2.0)

        # BCR = BNP / max(Creatinine, 0.1)  — cardiorenal stress index
        bnp_val  = lab_vals_raw.get("bnp",        100.0)
        creat    = lab_vals_raw.get("creatinine",    1.0)
        bcr      = bnp_val / max(creat, 0.1)
        bcr_norm = float(np.clip(math.log1p(bcr) / math.log1p(20000.0), 0.0, 1.0))

        # ── Weight-change features ───────────────────────────────────────────
        # Weight is the one channel whose signal lives on a daily timescale:
        # a gain above ~2 kg in 72 h is the classic marker of heart-failure
        # decompensation, and the absolute weight carries far less information
        # than its trend. Both deltas are therefore computed explicitly rather
        # than left to the window statistics, which cannot see across days.
        w_now = vital_vals_raw.get("weight")
        w_24, w_72 = prior_weights
        scale = config.WEIGHT_DELTA_SCALE

        def _wdelta(prior: float | None) -> float:
            if w_now is None or prior is None or vital_missing.get("weight", True):
                return 0.0
            return float(np.clip((w_now - prior) / max(scale, 1e-6), -1.0, 1.0))

        delta_w24 = _wdelta(w_24)
        delta_w72 = _wdelta(w_72)

        # ── [53:55] Temporal context (hour of day as sin/cos) ─────────────────
        hour      = now.hour + now.minute / 60.0
        hour_sin  = math.sin(2 * math.pi * hour / 24.0)
        hour_cos  = math.cos(2 * math.pi * hour / 24.0)

        # ── Assemble vector ───────────────────────────────────────────────────
        vector = (
            vital_norm       +   # [0:8]
            lab_norm         +   # [8:20]
            metadata         +   # [20:28]
            vital_miss_vec   +   # [28:36]
            lab_miss_vec     +   # [36:48]
            deltas           +   # lab deltas (3)
            [delta_w24, delta_w72] +  # weight deltas (2)
            [sig_norm, bcr_norm] +    # interaction scores (2)
            [hour_sin, hour_cos]      # temporal context (2)
        )

        assert len(vector) == config.N_FEATURES, \
            f"Feature vector length {len(vector)} ≠ {config.N_FEATURES}"

        arr = np.array(vector, dtype=np.float32)

        # Build feature-name dict for SHAP labelling
        feature_dict: dict[str, float] = {}
        all_names = (
            [f"vital_{n}_{st}" for n in VITAL_NAMES
             for st in ("mean", "std", "min", "rate")] +
            [f"lab_{n}"   for n in LAB_NAMES]   +
            ["meta_vital_cov", "meta_lab_cov", "meta_abg_present",
             "meta_doc_lag", "meta_vital_recency", "meta_med_burden",
             "meta_iv_drip", "meta_icu_los"] +
            [f"miss_vital_{n}" for n in VITAL_NAMES] +
            [f"miss_lab_{n}"   for n in LAB_NAMES]   +
            ["delta_troponin_i", "delta_bnp", "delta_lactate"] +
            ["delta_weight_24h", "delta_weight_72h"] +
            ["sig", "bcr"] +
            ["hour_sin", "hour_cos"]
        )
        for name, val in zip(all_names, arr.tolist(), strict=True):
            feature_dict[name] = float(val)

        return arr, feature_dict

    @property
    def feature_names(self) -> list[str]:
        dummy, names = self.build_vector([], [], None, [], None)
        return list(names.keys())
