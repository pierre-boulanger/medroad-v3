"""
MIMIC-IV to window CSV.

Retraining after a feature-vector change cannot reuse the previous training
CSV: the four per-vital statistics and the weight deltas are computed from
timestamped observations, and an aggregated CSV has already thrown those away.
This module rebuilds the training matrix from MIMIC-IV event tables through the
same FeatureEngineer the live pipeline uses, so training and inference cannot
drift apart.

Expected inputs are filtered extracts rather than the full MIMIC-IV tables:

    chartevents.csv   stay_id, charttime, itemid, valuenum
    labevents.csv     hadm_id, charttime, itemid, valuenum
    stays.csv         stay_id, hadm_id, subject_id, intime, outtime
    outcomes.csv      stay_id, event_time        (deterioration events)

Produce them with a SQL filter on the itemids in ITEMID_TO_LOINC, which keeps
the extract to a few GB rather than the full several hundred.

Usage:
    python -m medroad_v3.training.etl --mimic-dir ./mimic_extract --out windows.csv
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from medroad_v3 import config
from medroad_v3.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# MIMIC-IV itemid to LOINC
# ══════════════════════════════════════════════════════════════════════════
# MIMIC-IV identifies measurements by itemid, not LOINC, so the mapping has to
# be explicit. Verify these against d_items and d_labitems for your MIMIC-IV
# release before trusting a training run: itemids are not stable across
# versions, and a silently wrong mapping produces a model that trains cleanly
# and means nothing.

ITEMID_TO_LOINC: dict[int, str] = {
    # ── chartevents: vital signs ──────────────────────────────────────────
    220045: "8867-4",    # Heart Rate
    220179: "55284-4",   # Non-invasive BP systolic
    220050: "55284-4",   # Arterial BP systolic
    220180: "8462-4",    # Non-invasive BP diastolic
    220051: "8462-4",    # Arterial BP diastolic
    220277: "59408-5",   # SpO2
    220210: "9279-1",    # Respiratory rate
    224690: "9279-1",    # Respiratory rate (total)
    223761: "8310-5",    # Temperature Fahrenheit
    223762: "8310-5",    # Temperature Celsius
    220739: "59560-5",   # GCS eye opening
    223900: "59560-5",   # GCS verbal
    223901: "59560-5",   # GCS motor
    223791: "59574-4",   # Pain level
    226512: "29463-7",   # Admission weight (kg)
    224639: "29463-7",   # Daily weight

    # ── labevents ─────────────────────────────────────────────────────────
    51003:  "10839-9",   # Troponin T
    50963:  "30522-7",   # NTproBNP
    50912:  "2160-0",    # Creatinine
    50971:  "2823-3",    # Potassium
    50983:  "2090-9",    # Sodium
    50882:  "14682-9",   # Bicarbonate
    50818:  "2019-8",    # pCO2
    50820:  "59578-5",   # pH
    50813:  "2532-0",    # Lactate
    51221:  "4544-3",    # Hematocrit
    51301:  "6690-2",    # WBC
    50960:  "2601-3",    # Magnesium
}

VITAL_LOINCS = set(config.LOINC_VITALS.values())
LAB_LOINCS = set(config.LOINC_LABS.values())

# Temperature in Fahrenheit needs converting; weight in lb likewise.
FAHRENHEIT_ITEMIDS = {223761}


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════

def _read(path: Path, **kw) -> pd.DataFrame:
    logger.info("reading %s", path.name)
    df = pd.read_csv(path, **kw)
    logger.info("  %d rows", len(df))
    return df


def load_extracts(mimic_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(mimic_dir)
    out = {
        "chart":    _read(d / "chartevents.csv", parse_dates=["charttime"]),
        "labs":     _read(d / "labevents.csv", parse_dates=["charttime"]),
        "stays":    _read(d / "stays.csv", parse_dates=["intime", "outtime"]),
        "outcomes": _read(d / "outcomes.csv", parse_dates=["event_time"]),
    }
    for key in ("chart", "labs"):
        df = out[key]
        df["loinc"] = df["itemid"].map(ITEMID_TO_LOINC)
        unmapped = df["loinc"].isna().sum()
        if unmapped:
            logger.warning("%s: dropping %d rows with unmapped itemid", key, unmapped)
        df.dropna(subset=["loinc", "valuenum"], inplace=True)
        if key == "chart":
            f = df["itemid"].isin(FAHRENHEIT_ITEMIDS)
            df.loc[f, "valuenum"] = (df.loc[f, "valuenum"] - 32.0) * 5.0 / 9.0
        out[key] = df
    return out


def _as_observation(loinc: str, value: float, when) -> dict:
    """Wrap a row as the FHIR Observation shape FeatureEngineer expects."""
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc}]},
        "effectiveDateTime": pd.Timestamp(when).tz_localize("UTC").isoformat()
        if pd.Timestamp(when).tzinfo is None
        else pd.Timestamp(when).isoformat(),
        "valueQuantity": {"value": float(value)},
    }


# ══════════════════════════════════════════════════════════════════════════
# Window construction
# ══════════════════════════════════════════════════════════════════════════

def build_windows(
    data: dict[str, pd.DataFrame],
    horizon_hours: float = 12.0,
    window_seconds: int = config.WINDOW_SECONDS,
    max_stays: int | None = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """
    Produce one row per inference window.

    A window is labelled positive when a deterioration event occurs within
    ``horizon_hours`` after it. Windows after the first event in a stay are
    dropped, so the model is never asked to predict an event that has already
    happened, which would leak the outcome through post-event treatment.
    """
    fe = FeatureEngineer()
    chart, labs = data["chart"], data["labs"]
    stays, outcomes = data["stays"], data["outcomes"]

    first_event = outcomes.groupby("stay_id")["event_time"].min()
    chart_by_stay = dict(tuple(chart.groupby("stay_id")))
    labs_by_hadm = dict(tuple(labs.groupby("hadm_id")))

    # Sampled at random, not taken in stay_id order. MIMIC-IV stay_ids are
    # not arbitrary: taking the first N draws a block of admissions that are
    # correlated in time and site, so a capped run would be unrepresentative
    # in exactly the way a pilot is meant to rule out.
    if max_stays and max_stays < len(stays):
        stays = stays.sample(n=max_stays, random_state=sample_seed)
        logger.info("sampled %d of %d stays (seed %d)",
                    max_stays, len(stays), sample_seed)
    stay_rows = stays.itertuples()

    rows: list[dict] = []
    stay_list = list(stay_rows)
    total = len(stay_list)
    t_start = time.time()

    for n_done, stay in enumerate(stay_list, 1):
        if n_done % 500 == 0 or n_done == total:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1e-6)
            eta = (total - n_done) / max(rate, 1e-6)
            logger.info(
                "  %d/%d stays (%.0f%%)  %d windows  %.0f stays/s  ETA %.0f min",
                n_done, total, 100 * n_done / total, len(rows), rate, eta / 60,
            )
        ce = chart_by_stay.get(stay.stay_id)
        if ce is None or ce.empty:
            continue
        le = labs_by_hadm.get(stay.hadm_id, pd.DataFrame(
            columns=["charttime", "loinc", "valuenum"]))

        event_t = first_event.get(stay.stay_id, pd.NaT)
        stop = min(stay.outtime, event_t) if pd.notna(event_t) else stay.outtime

        # Columns are pulled out as numpy arrays once per stay. The previous
        # implementation called DataFrame.itertuples() on a fresh slice for
        # every five-minute window, which dominated runtime and, on one stay in
        # this cohort, failed inside pandas' own indexing machinery.
        ce_time = ce["charttime"].to_numpy()
        ce_loinc = ce["loinc"].to_numpy()
        ce_val = ce["valuenum"].to_numpy(dtype=float)

        le_time = le["charttime"].to_numpy() if len(le) else np.empty(0, dtype="datetime64[ns]")
        le_loinc = le["loinc"].to_numpy() if len(le) else np.empty(0, dtype=object)
        le_val = (le["valuenum"].to_numpy(dtype=float) if len(le)
                  else np.empty(0, dtype=float))

        t = stay.intime + timedelta(seconds=window_seconds)
        step = timedelta(seconds=window_seconds)
        weight_hist: list[tuple[pd.Timestamp, float]] = []

        while t <= stop:
            w_lo = np.datetime64(pd.Timestamp(t - step))
            w_hi = np.datetime64(pd.Timestamp(t))
            m = (ce_time > w_lo) & (ce_time <= w_hi)
            if not m.any():
                t += step
                continue

            w_loinc, w_val, w_time = ce_loinc[m], ce_val[m], ce_time[m]
            window_obs = [
                _as_observation(lo, v, ts)
                for lo, v, ts in zip(w_loinc, w_val, w_time, strict=True)
                if lo in VITAL_LOINCS
            ]

            lab_lo = np.datetime64(pd.Timestamp(t - timedelta(hours=24)))
            lm = (le_time > lab_lo) & (le_time <= w_hi)
            lab_obs = [
                _as_observation(lo, v, ts)
                for lo, v, ts in zip(le_loinc[lm], le_val[lm], le_time[lm],
                                     strict=True)
                if lo in LAB_LOINCS
            ]

            # weight history for the 24 h and 72 h deltas
            for lo, v, ts in zip(w_loinc, w_val, w_time, strict=True):
                if lo == "29463-7":
                    weight_hist.append((pd.Timestamp(ts), float(v)))

            def _weight_at(delta_h: float, _t=t, _hist=weight_hist) -> float | None:
                target = _t - timedelta(hours=delta_h)
                past = [v for ts_, v in _hist if ts_ <= target]
                return past[-1] if past else None

            intime_utc = pd.Timestamp(stay.intime)
            if intime_utc.tzinfo is None:
                intime_utc = intime_utc.tz_localize("UTC")

            # "now" must be the window end. Left to its default it becomes
            # wall-clock time, and because MIMIC-IV timestamps are shifted
            # decades into the future the documentation-lag features come out
            # around -13 million instead of within [0, 1]. Tree models are
            # scale-invariant and never notice; neural nets are destroyed by it.
            t_utc = pd.Timestamp(t)
            if t_utc.tzinfo is None:
                t_utc = t_utc.tz_localize("UTC")

            vec, _ = fe.build_vector(
                window_obs, lab_obs, None, [], intime_utc.to_pydatetime(),
                now=t_utc.to_pydatetime(),
                prior_weights=(_weight_at(24), _weight_at(72)),
            )

            label = int(
                pd.notna(event_t) and event_t <= t + timedelta(hours=horizon_hours)
            )
            rows.append({
                "patient_id": stay.subject_id,
                "stay_id": stay.stay_id,
                "window_start": w_lo,
                "label": label,
                **{f"f{i}": v for i, v in enumerate(vec)},
            })
            t += step

    df = pd.DataFrame(rows)
    logger.info("built %d windows, event rate %.4f", len(df), df["label"].mean())
    return df


def rename_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positional f0..fN with the engineered feature names."""
    fe = FeatureEngineer()
    names = fe.feature_names
    mapping = {f"f{i}": n for i, n in enumerate(names)}
    return df.rename(columns=mapping)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the MedROAD V3 training CSV")
    ap.add_argument("--mimic-dir", required=True, help="directory of filtered extracts")
    ap.add_argument("--out", default="mimic_windows.csv")
    ap.add_argument("--horizon-hours", type=float, default=12.0)
    ap.add_argument("--max-stays", type=int, default=None,
                    help="randomly sample this many stays, for a scaled pilot")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="seed for the stay sample, so a pilot is reproducible")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data = load_extracts(args.mimic_dir)
    df = build_windows(data, horizon_hours=args.horizon_hours,
                       max_stays=args.max_stays,
                       sample_seed=args.sample_seed)
    df = rename_feature_columns(df)

    n_feat = len([c for c in df.columns
                  if c not in ("patient_id", "stay_id", "window_start", "label")])
    assert n_feat == config.N_FEATURES, f"{n_feat} feature columns != {config.N_FEATURES}"

    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows, %d features)", args.out, len(df), n_feat)
    print(f"\nwindows: {len(df)}   event rate: {df['label'].mean():.4f}   "
          f"features: {n_feat}")
    print(f"patients: {df['patient_id'].nunique()}   stays: {df['stay_id'].nunique()}")


if __name__ == "__main__":
    main()
