#!/usr/bin/env python3
"""
End-to-end retraining for MedROAD V3.

Runs extract validation, ETL, training and the experiment suite as one
sequence, with gates between stages. The gates exist because the failure mode
that matters here is silent: a wrong itemid mapping, a mislabelled cohort or a
constant sequence tensor all train cleanly and produce plausible metrics that
mean nothing. Each stage refuses to hand off to the next unless its output
looks sane.

Usage:
    python scripts/retrain.py --mimic-dir ./mimic_extract
    python scripts/retrain.py --mimic-dir ./mimic_extract --quick    # 50 stays
    python scripts/retrain.py --mimic-dir ./mimic_extract --skip-etl # reuse CSV
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medroad_v3 import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrain")

REQUIRED_EXTRACTS = ("chartevents.csv", "labevents.csv", "stays.csv", "outcomes.csv")

# Plausibility bounds. Outside these, stop rather than train.
MIN_EVENT_RATE = 0.01
MAX_EVENT_RATE = 0.60
MIN_VITAL_COVERAGE = 0.50     # fraction of stays with a given vital present
MIN_WINDOWS = 1000


class GateFailure(RuntimeError):
    """A validation gate rejected its input."""


def banner(stage: str) -> None:
    log.info("")
    log.info("=" * 66)
    log.info("  %s", stage)
    log.info("=" * 66)


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — validate the extracts
# ══════════════════════════════════════════════════════════════════════════

def validate_extracts(mimic_dir: Path, strict: bool = True) -> dict:
    banner("STAGE 1  validate MIMIC-IV extracts")
    missing = [f for f in REQUIRED_EXTRACTS if not (mimic_dir / f).exists()]
    if missing:
        raise GateFailure(
            f"missing extracts: {', '.join(missing)}. "
            "Run sql/extract_mimic.sql first."
        )

    from medroad_v3.training.etl import ITEMID_TO_LOINC

    stays = pd.read_csv(mimic_dir / "stays.csv")
    outcomes = pd.read_csv(mimic_dir / "outcomes.csv")
    chart = pd.read_csv(mimic_dir / "chartevents.csv", usecols=["stay_id", "itemid"])

    n_stays = stays.stay_id.nunique()
    n_event = outcomes.stay_id.nunique()
    stay_event_rate = n_event / max(n_stays, 1)

    log.info("stays            : %d", n_stays)
    log.info("stays with event : %d  (%.1f%%)", n_event, 100 * stay_event_rate)

    # itemid coverage: a mapping wrong for this MIMIC release shows up here
    present = set(chart.itemid.unique())
    mapped = set(ITEMID_TO_LOINC)
    overlap = present & mapped
    log.info("mapped itemids present in chartevents: %d of %d",
             len(overlap), len(mapped))
    if not overlap:
        raise GateFailure(
            "no mapped itemid appears in chartevents. ITEMID_TO_LOINC is wrong "
            "for this MIMIC-IV release; check against mimiciv_icu.d_items."
        )

    coverage: dict[int, float] = {}
    for iid in sorted(overlap):
        c = chart.loc[chart.itemid == iid, "stay_id"].nunique() / max(n_stays, 1)
        coverage[int(iid)] = round(c, 3)

    hr = coverage.get(220045, 0.0)
    log.info("heart-rate coverage: %.1f%% of stays", 100 * hr)
    if hr < MIN_VITAL_COVERAGE:
        msg = (f"heart rate present in only {100*hr:.1f}% of stays; expected "
               f">{100*MIN_VITAL_COVERAGE:.0f}%. Likely a wrong itemid or a "
               "truncated extract.")
        if strict:
            raise GateFailure(msg)
        log.warning(msg)

    thin = {k: v for k, v in coverage.items() if v < 0.2}
    if thin:
        log.warning("itemids present in under 20%% of stays: %s", thin)

    log.info("extracts look usable")
    return {"n_stays": n_stays, "stay_event_rate": stay_event_rate,
            "itemid_coverage": coverage}


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — ETL
# ══════════════════════════════════════════════════════════════════════════

def run_etl(mimic_dir: Path, out_csv: Path, horizon: float,
            max_stays: int | None) -> None:
    banner("STAGE 2  build the window matrix")
    cmd = [sys.executable, "-m", "medroad_v3.training.etl",
           "--mimic-dir", str(mimic_dir), "--out", str(out_csv),
           "--horizon-hours", str(horizon)]
    if max_stays:
        cmd += ["--max-stays", str(max_stays)]
    log.info("$ %s", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    if r.returncode != 0:
        raise GateFailure("ETL failed")
    log.info("ETL completed in %.1f min", (time.time() - t0) / 60)


def validate_windows(csv: Path, strict: bool = True) -> dict:
    banner("STAGE 3  validate the window matrix")
    df = pd.read_csv(csv, nrows=200_000)
    meta = {"patient_id", "stay_id", "window_start", "label"}
    feats = [c for c in df.columns if c not in meta]

    log.info("windows (sampled): %d", len(df))
    log.info("feature columns  : %d", len(feats))

    if len(feats) != config.N_FEATURES:
        raise GateFailure(
            f"{len(feats)} feature columns but config.N_FEATURES is "
            f"{config.N_FEATURES}. The ETL and the model disagree."
        )

    rate = float(df.label.mean())
    log.info("window event rate: %.4f", rate)
    if not (MIN_EVENT_RATE <= rate <= MAX_EVENT_RATE):
        msg = (f"window event rate {rate:.4f} outside [{MIN_EVENT_RATE}, "
               f"{MAX_EVENT_RATE}]. Check the outcome query and the horizon.")
        if strict:
            raise GateFailure(msg)
        log.warning(msg)

    if len(df) < MIN_WINDOWS and strict:
        raise GateFailure(f"only {len(df)} windows; too few to train on")

    X = df[feats].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise GateFailure("non-finite values in the feature matrix")

    # Every engineered feature is bounded to [-1, 1] by construction. Anything
    # outside it is a defect, not an extreme patient, and it will wreck any
    # model that is not scale-invariant while leaving tree models untouched.
    lo, hi = X.min(axis=0), X.max(axis=0)
    out_of_range = [
        (f, float(a), float(b))
        for f, a, b in zip(feats, lo, hi, strict=True)
        if a < -1.001 or b > 1.001
    ]
    if out_of_range:
        detail = ", ".join(f"{f} [{a:.3g}, {b:.3g}]" for f, a, b in out_of_range[:5])
        raise GateFailure(
            f"{len(out_of_range)} feature column(s) fall outside [-1, 1]: {detail}"
        )

    constant = [f for f, s in zip(feats, X.std(axis=0), strict=True) if s == 0.0]
    if constant:
        log.warning("%d constant feature columns: %s%s", len(constant),
                    constant[:8], " ..." if len(constant) > 8 else "")

    # sequence check: the failure that used to be invisible
    from medroad_v3.training.sequences import (
        build_patient_sequences,
        sequence_variation,
    )
    sub = df.head(20_000)
    S = build_patient_sequences(
        sub[feats].to_numpy(dtype=np.float32),
        sub.patient_id.to_numpy(),
        sub.window_start.to_numpy(),
    )
    var = sequence_variation(S)
    log.info("sequence temporal variation: %.6f", var)
    if var < 1e-8:
        raise GateFailure(
            "sequences are constant. The LSTM and Transformer would learn "
            "nothing temporal; do not train in this state."
        )

    log.info("window matrix looks usable")
    return {"n_windows": int(len(df)), "event_rate": rate,
            "n_features": len(feats), "sequence_variation": var,
            "constant_columns": constant}


# ══════════════════════════════════════════════════════════════════════════
# Stage 4 — train
# ══════════════════════════════════════════════════════════════════════════

def run_training(csv: Path, model_dir: Path) -> None:
    banner("STAGE 4  train")
    cmd = [sys.executable, "-m", "medroad_v3.training.train",
           "--data", str(csv), "--output", str(model_dir)]
    log.info("$ %s", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    if r.returncode != 0:
        raise GateFailure("training failed")
    log.info("training completed in %.1f min", (time.time() - t0) / 60)

    expected = ["xgb", "lstm.pt", "transformer.pt", "ensemble"]
    missing = [e for e in expected if not (model_dir / e).exists()]
    if missing:
        raise GateFailure(f"training produced no {missing}")
    log.info("all model artefacts present")


# ══════════════════════════════════════════════════════════════════════════
# Stage 5 — experiments
# ══════════════════════════════════════════════════════════════════════════

def run_experiments(csv: Path, out_dir: Path) -> list[str]:
    banner("STAGE 5  experiments")
    done = []
    for mod, name in (
        ("medroad_v3.experiments.ablation", "ablation"),
        ("medroad_v3.experiments.rpm_degradation", "rpm"),
    ):
        cmd = [sys.executable, "-m", mod, "--data", str(csv),
               "--out", str(out_dir / name)]
        log.info("$ %s", " ".join(cmd))
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
        if r.returncode == 0:
            done.append(name)
        else:
            log.error("%s failed; continuing", name)
    return done


# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="MedROAD V3 end-to-end retraining")
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--windows", default="mimic_windows.csv")
    ap.add_argument("--models", default="models_saved")
    ap.add_argument("--results", default="results")
    ap.add_argument("--horizon-hours", type=float, default=12.0)
    ap.add_argument("--quick", action="store_true",
                    help="50 stays only, to verify the pipeline")
    ap.add_argument("--max-stays", type=int, default=None,
                    help="randomly sample this many stays. Use for a scaled "
                         "pilot, e.g. 1000, between the 50-stay smoke test and "
                         "the full cohort.")
    ap.add_argument("--skip-etl", action="store_true",
                    help="reuse an existing window CSV")
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--lenient", action="store_true",
                    help="warn instead of stopping on plausibility gates")
    args = ap.parse_args()

    mimic_dir = Path(args.mimic_dir)
    csv = Path(args.windows)
    models = Path(args.models)
    results = Path(args.results)
    strict = not args.lenient
    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        report["extracts"] = validate_extracts(mimic_dir, strict)

        if args.skip_etl:
            if not csv.exists():
                raise GateFailure(f"--skip-etl given but {csv} does not exist")
            log.info("reusing %s", csv)
        else:
            n_stays = args.max_stays or (50 if args.quick else None)
            if n_stays:
                log.info("sampling %d stays for this run", n_stays)
            run_etl(mimic_dir, csv, args.horizon_hours, n_stays)

        report["windows"] = validate_windows(csv, strict)
        run_training(csv, models)
        if not args.skip_experiments:
            report["experiments"] = run_experiments(csv, results)

    except GateFailure as e:
        banner("STOPPED")
        log.error("%s", e)
        report["failed"] = str(e)
        results.mkdir(parents=True, exist_ok=True)
        (results / "retrain_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        return 1

    banner("DONE")
    w = report["windows"]
    log.info("windows %d   event rate %.4f   features %d",
             w["n_windows"], w["event_rate"], w["n_features"])
    log.info("models  -> %s", models)
    log.info("results -> %s", results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "retrain_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    log.info("")
    log.info("Next: update the manuscript's results table from "
             "%s/ablation/ablation_table.tex", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
