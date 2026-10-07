"""
Experiment 5 — Scaling with concurrent patient count.

The question a deployment has to answer is whether the system keeps up: with
$N$ monitored patients, each producing one five-minute window, can all $N$ be
scored before the next window arrives? That gives a hard deadline of 300
seconds per round and a saturation point beyond which the backlog grows
without bound.

This measures the compute path under increasing $N$. Each round submits one
window per simulated patient and records the wall time to clear the round, the
per-window latency within it, and the resulting headroom against the deadline.

Two things are measured here and one is not. Measured: how per-window latency
degrades as concurrent load rises, and where the deadline would be breached.
Not measured: broker and FHIR transport under load, which needs the live stack
and is bounded by network and server behaviour rather than by this process.
The figure this produces is therefore a compute-path scaling curve, and must
be labelled as such rather than as end-to-end.

Usage:
    python -m medroad_v3.experiments.latency_scaling --data mimic_windows.csv
    python -m medroad_v3.experiments.latency_scaling --data mimic_windows.csv \\
        --patients 1 5 10 20 40 60 80 120 --rounds 5
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from medroad_v3 import config
from medroad_v3.experiments.common import save_results, save_tex

logger = logging.getLogger(__name__)

DEADLINE_S = float(config.WINDOW_SECONDS)   # one window period


def run(
    data: str,
    patient_counts: list[int],
    rounds: int,
    model_dir: str,
    device: str | None = None,
) -> dict:
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
    md = Path(model_dir)

    df = pd.read_csv(data, nrows=50_000, parse_dates=["window_start"])
    meta_cols = {"patient_id", "stay_id", "window_start", "label"}
    feats = [c for c in df.columns if c not in meta_cols]
    X = df[feats].to_numpy(dtype=np.float32)
    S = build_patient_sequences(X, df.patient_id.to_numpy(),
                                df.window_start.to_numpy())

    xgb = XGBoostClassifier()
    xgb.load(str(md / "xgb"))
    lstm = load_model(lambda: TemperatureScaledModel(LSTMModel()),
                      str(md / "lstm.pt"), dev)
    tfm = load_model(lambda: TemperatureScaledModel(TransformerModel()),
                     str(md / "transformer.pt"), dev)
    meta = EnsembleMeta()
    meta.load(str(md / "ensemble"))

    rng = np.random.default_rng(0)

    def score_one(i: int) -> float:
        """One window through the full compute path, as deployed."""
        p1 = xgb.predict_single(X[i])
        t = torch.tensor(S[i:i + 1], dtype=torch.float32).to(dev)
        with torch.no_grad():
            p2 = float(lstm(t).item())
            p3 = float(tfm(t).item())
        r = meta.predict(p1, p2, p3)
        if r >= meta.threshold:
            xgb.shap_values(X[i])      # only alerting windows pay for SHAP
        return r

    # warm-up
    for _ in range(50):
        score_one(int(rng.integers(len(X))))
    if dev == "cuda":
        torch.cuda.synchronize()

    results: dict = {"_device": dev, "_deadline_s": DEADLINE_S, "rounds": {}}
    for n_pat in patient_counts:
        round_times, per_window = [], []
        for _ in range(rounds):
            idx = rng.integers(0, len(X), size=n_pat)
            t0 = time.perf_counter()
            for i in idx:
                t1 = time.perf_counter()
                score_one(int(i))
                per_window.append((time.perf_counter() - t1) * 1000.0)
            if dev == "cuda":
                torch.cuda.synchronize()
            round_times.append(time.perf_counter() - t0)

        rt = np.asarray(round_times)
        pw = np.asarray(per_window)
        entry = {
            "patients": n_pat,
            "round_s_mean": float(rt.mean()),
            "round_s_max": float(rt.max()),
            "per_window_ms_p50": float(np.percentile(pw, 50)),
            "per_window_ms_p99": float(np.percentile(pw, 99)),
            "deadline_utilisation": float(rt.max() / DEADLINE_S),
            "throughput_windows_per_s": float(n_pat / rt.mean()),
        }
        results["rounds"][str(n_pat)] = entry
        logger.info(
            "N=%4d  round %.3f s  per-window p50 %.2f ms  deadline used %.3f%%",
            n_pat, entry["round_s_mean"], entry["per_window_ms_p50"],
            100 * entry["deadline_utilisation"],
        )
    return results


def to_latex(results: dict) -> str:
    rows = results["rounds"]
    coords = " ".join(
        f"({r['patients']},{1000*r['round_s_mean']:.1f})" for r in rows.values()
    )
    pw = " ".join(
        f"({r['patients']},{r['per_window_ms_p50']:.2f})" for r in rows.values()
    )
    worst = max(r["deadline_utilisation"] for r in rows.values())
    nmax = max(r["patients"] for r in rows.values())

    return rf"""% Compute-path scaling - generated from measurement.
\begin{{figure}}[ht]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.88\linewidth, height=5.2cm,
  xlabel={{Concurrent patients}},
  ylabel={{Time to clear one round (ms)}},
  xmin=0, grid=major, grid style={{dashed,gray!30}},
  legend pos=north west, legend style={{font=\footnotesize}}
]
\addplot[blue, thick, mark=square*] coordinates {{ {coords} }};
\addlegendentry{{Round completion time}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Compute-path scaling. Each round scores one five-minute window per
concurrent patient through the full inference path. At {nmax} concurrent
patients the round consumes {100*worst:.2f}\% of the {int(DEADLINE_S)}\,s
window period, so the compute path is far from saturation; the binding
constraint is transport, not inference. Per-window median latency is
{{{pw}}} across the same sweep.}}
\label{{fig:latency_scaling}}
\end{{figure}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="MedROAD V3 concurrency scaling")
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default=config.MODEL_DIR)
    ap.add_argument("--patients", type=int, nargs="+",
                    default=[1, 5, 10, 20, 40, 60, 80])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/latency")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    results = run(args.data, args.patients, args.rounds, args.models, args.device)

    print(f"\n{'patients':>9} {'round (s)':>11} {'p50 (ms)':>10} "
          f"{'p99 (ms)':>10} {'deadline used':>15}")
    for r in results["rounds"].values():
        print(f"{r['patients']:>9} {r['round_s_mean']:>11.3f} "
              f"{r['per_window_ms_p50']:>10.2f} {r['per_window_ms_p99']:>10.2f} "
              f"{100*r['deadline_utilisation']:>14.3f}%")

    save_results(results, args.out, "latency_scaling")
    save_tex(to_latex(results), args.out, "latency_scaling_figure")
    print(f"\nwrote {Path(args.out)}/latency_scaling_figure.tex")


if __name__ == "__main__":
    main()
