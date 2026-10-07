"""
Experiment 4 — Latency on the real cohort.

The manuscript's latency table reports approximate per-stage figures from a
synthetic 20-patient workload, as means with no tail. For a real-time claim the
tail is what determines whether a deadline is met: a 60 ms mean with a 400 ms
p99 is a different system from one with a 70 ms p99, and only the second can be
promised to a clinician.

This measures the compute path on windows drawn from the actual cohort, each
stage timed separately, reported as median, p95 and p99 over many repetitions.

What is measured here: feature-vector assembly, each base model, the
meta-learner, SHAP attribution, and the narrative call when an API key is
present. What is not: FHIR webhook delivery, Kafka produce and consume, and
FHIR write-back, all of which need the live stack and are measured by
``--live`` against a running deployment.

Usage:
    python -m medroad_v3.experiments.latency --data mimic_windows.csv
    python -m medroad_v3.experiments.latency --data mimic_windows.csv --n 2000
    python -m medroad_v3.experiments.latency --data mimic_windows.csv --live
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC
from pathlib import Path

import numpy as np

from medroad_v3 import config
from medroad_v3.experiments.common import latex_table, save_results, save_tex

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════

def _summarise(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def time_stage(fn, n_rep: int, warmup: int = 20) -> dict:
    """Time a callable, discarding warmup iterations."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return _summarise(out)


# ══════════════════════════════════════════════════════════════════════════

def run(data: str, n_rep: int, model_dir: str, device: str | None = None) -> dict:
    import pandas as pd
    import torch

    from medroad_v3.models.deep_models import (
        LSTMModel,
        TemperatureScaledModel,
        TransformerModel,
        load_model,
    )
    from medroad_v3.models.ensemble import EnsembleMeta
    from medroad_v3.models.xgboost_model import XGBoostClassifier
    from medroad_v3.training.sequences import build_patient_sequences

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", dev)

    df = pd.read_csv(data, nrows=50_000, parse_dates=["window_start"])
    meta_cols = {"patient_id", "stay_id", "window_start", "label"}
    feats = [c for c in df.columns if c not in meta_cols]
    X = df[feats].to_numpy(dtype=np.float32)
    S = build_patient_sequences(
        X, df.patient_id.to_numpy(), df.window_start.to_numpy()
    )
    logger.info("loaded %d windows, %d features", len(X), len(feats))

    md = Path(model_dir)
    xgb = XGBoostClassifier()
    xgb.load(str(md / "xgb"))
    lstm = load_model(lambda: TemperatureScaledModel(LSTMModel()),
                      str(md / "lstm.pt"), dev)
    tfm = load_model(lambda: TemperatureScaledModel(TransformerModel()),
                     str(md / "transformer.pt"), dev)
    meta = EnsembleMeta()
    meta.load(str(md / "ensemble"))

    rng = np.random.default_rng(0)
    results: dict[str, dict] = {}

    # Single-window latency: the quantity that matters for a per-patient alert.
    def one_xgb():
        i = rng.integers(len(X))
        xgb.predict_single(X[i])

    def one_shap():
        i = rng.integers(len(X))
        xgb.shap_values(X[i])

    def _torch_one(model):
        def f():
            i = rng.integers(len(S))
            t = torch.tensor(S[i:i + 1], dtype=torch.float32).to(dev)
            with torch.no_grad():
                model(t)
            if dev == "cuda":
                torch.cuda.synchronize()
        return f

    def one_meta():
        meta.predict(float(rng.random()), float(rng.random()), float(rng.random()))

    results["XGBoost inference"] = time_stage(one_xgb, n_rep)
    results["LSTM inference"] = time_stage(_torch_one(lstm), n_rep)
    results["Transformer inference"] = time_stage(_torch_one(tfm), n_rep)
    results["Meta-learner"] = time_stage(one_meta, n_rep)
    results["SHAP attribution"] = time_stage(one_shap, max(n_rep // 5, 50))

    # Feature assembly, timed through the real FeatureEngineer rather than the
    # pre-built matrix, since that is what runs per window in deployment.
    from datetime import datetime, timedelta

    from medroad_v3.features.engineering import FeatureEngineer
    fe = FeatureEngineer()
    now = datetime.now(UTC)
    codes = list(config.LOINC_VITALS.values())
    window_obs = [{
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": c}]},
        "effectiveDateTime": (now - timedelta(seconds=30 * k)).isoformat(),
        "valueQuantity": {"value": 80.0 + k},
    } for k, c in enumerate(codes * 4)]
    lab_obs = [{
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org", "code": c}]},
        "effectiveDateTime": (now - timedelta(hours=2)).isoformat(),
        "valueQuantity": {"value": 1.5},
    } for c in config.LOINC_LABS.values()]

    results["Feature assembly"] = time_stage(
        lambda: fe.build_vector(window_obs, lab_obs, None, [], None, now=now),
        n_rep,
    )

    # End-to-end compute path, excluding transport.
    def full():
        i = int(rng.integers(len(X)))
        p1 = xgb.predict_single(X[i])
        t = torch.tensor(S[i:i + 1], dtype=torch.float32).to(dev)
        with torch.no_grad():
            p2 = float(lstm(t).item())
            p3 = float(tfm(t).item())
        if dev == "cuda":
            torch.cuda.synchronize()
        r = meta.predict(p1, p2, p3)
        if r >= meta.threshold:
            xgb.shap_values(X[i])

    results["Compute path (total)"] = time_stage(full, n_rep)
    return results


# ══════════════════════════════════════════════════════════════════════════

ORDER = [
    "Feature assembly",
    "XGBoost inference",
    "LSTM inference",
    "Transformer inference",
    "Meta-learner",
    "SHAP attribution",
    "Compute path (total)",
]


def to_latex(results: dict, device: str) -> str:
    rows = []
    for k in ORDER:
        if k not in results:
            continue
        r = results[k]
        bold = k.endswith("(total)")
        name = rf"\textbf{{{k}}}" if bold else k
        fmt = (lambda v: rf"\textbf{{{v:.2f}}}") if bold else (lambda v: f"{v:.2f}")
        rows.append([name, fmt(r["p50"]), fmt(r["p95"]), fmt(r["p99"]), fmt(r["max"])])

    return latex_table(
        rows,
        ["Stage", "p50 (ms)", "p95 (ms)", "p99 (ms)", "max (ms)"],
        caption=(
            "Per-stage inference latency measured on windows drawn from the "
            f"evaluation cohort ({device}). Each stage is timed over repeated "
            "single-window invocations after warm-up, which is the unit of work "
            "a per-patient alert represents. Transport stages (FHIR delivery, "
            "Kafka, write-back) are excluded; they are reported separately "
            "against the live deployment."
        ),
        label="tab:latency_measured",
        col_spec="@{}lrrrr@{}",
        note=(
            "Percentiles rather than means: for a real-time guarantee the tail "
            "determines whether the deadline is met."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MedROAD V3 latency benchmark")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default=config.MODEL_DIR)
    ap.add_argument("--n", type=int, default=1000, help="repetitions per stage")
    ap.add_argument("--device", default=None, help="cuda or cpu")
    ap.add_argument("--out", default="results/latency")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = run(args.data, args.n, args.models, dev)

    print(f"\n{'stage':>24}  {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   (ms)")
    for k in ORDER:
        if k not in results:
            continue
        r = results[k]
        print(f"{k:>24}  {r['p50']:8.2f} {r['p95']:8.2f} {r['p99']:8.2f} "
              f"{r['max']:8.2f}")

    results["_device"] = dev
    save_results(results, args.out, "latency")
    save_tex(to_latex(results, dev), args.out, "latency_table")
    print(f"\nwrote {Path(args.out)}/latency.json and latency_table.tex")


if __name__ == "__main__":
    main()
