"""
Transport latency against a live deployment.

The compute path is measured by ``latency.py`` without any infrastructure. The
remaining stages of the manuscript's latency budget — FHIR delivery, Kafka
produce and consume, and FHIR write-back — exist only when the stack is
running, and account for the larger share of the total.

Each stage is measured by round-tripping real requests and reporting
percentiles. Run against a deployment with OpenEMR, Kafka and the webhook up:

    docker compose up -d
    python -m medroad_v3.experiments.latency_live --n 200

Nothing here writes clinical content. Observations posted to FHIR carry a
synthetic patient reference and an explicit benchmark marker, and the
write-back probe targets a Flag on that same synthetic subject, so a live
clinical instance is never polluted with test alerts.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

from medroad_v3 import config
from medroad_v3.experiments.common import latex_table, save_results, save_tex

logger = logging.getLogger(__name__)

BENCH_PATIENT = "MEDROAD-BENCH-DO-NOT-USE-CLINICALLY"


def _pct(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def _observation() -> dict:
    return {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "vital-signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
        "subject": {"reference": f"Patient/{BENCH_PATIENT}"},
        "effectiveDateTime": datetime.now(UTC).isoformat(),
        "valueQuantity": {"value": 88.0, "unit": "/min"},
        "note": [{"text": "MedROAD latency benchmark - synthetic, not clinical"}],
    }


# ══════════════════════════════════════════════════════════════════════════

def measure_webhook(n: int) -> dict | None:
    """POST to the webhook receiver and time acceptance."""
    url = f"http://{config.WEBHOOK_HOST}:{config.WEBHOOK_PORT}/fhir/webhook"
    url = url.replace("0.0.0.0", "127.0.0.1")
    headers = {"Content-Type": "application/fhir+json",
               "X-MedROAD-Secret": config.WEBHOOK_SECRET,
               "X-MedROAD-Topic": "vitals"}
    body = json.dumps(_observation())
    try:
        requests.post(url, data=body, headers=headers, timeout=5)
    except requests.RequestException as e:
        logger.warning("webhook unreachable at %s: %s", url, e)
        return None

    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = requests.post(url, data=body, headers=headers, timeout=10)
        out.append((time.perf_counter() - t0) * 1000.0)
        if r.status_code >= 400:
            logger.warning("webhook returned %d", r.status_code)
            return None
    return _pct(out)


def measure_kafka(n: int) -> dict | None:
    """Produce to Kafka and await broker acknowledgement."""
    try:
        from kafka import KafkaProducer
        prod = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode(),
            acks="all", retries=0,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Kafka unreachable at %s: %s", config.KAFKA_BOOTSTRAP, e)
        return None

    obs = _observation()
    out = []
    try:
        for _ in range(n):
            t0 = time.perf_counter()
            prod.send(config.KAFKA_TOPIC_VITALS, value=obs).get(timeout=10)
            out.append((time.perf_counter() - t0) * 1000.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("Kafka produce failed: %s", e)
        return None
    finally:
        prod.close()
    return _pct(out)


def measure_fhir_writeback(n: int) -> dict | None:
    """Time a Flag write to OpenEMR, the dominant write-back resource."""
    try:
        from medroad_v3.fhir.client import FHIRClient
        from medroad_v3.fhir.writeback import build_flag
        client = FHIRClient()
        if not client.ping():
            logger.warning("FHIR server unreachable at %s", config.FHIR_BASE_URL)
            return None
    except Exception as e:  # noqa: BLE001
        logger.warning("FHIR client unavailable: %s", e)
        return None

    out = []
    for _ in range(n):
        flag = build_flag(BENCH_PATIENT, 0.80)
        flag["code"]["text"] = f"MedROAD latency benchmark {uuid.uuid4().hex[:8]}"
        t0 = time.perf_counter()
        try:
            client.post("Flag", flag)
        except Exception as e:  # noqa: BLE001
            logger.warning("Flag write failed: %s", e)
            return None
        out.append((time.perf_counter() - t0) * 1000.0)
    return _pct(out)


# ══════════════════════════════════════════════════════════════════════════

STAGES = [
    ("FHIR webhook ingestion", measure_webhook),
    ("Kafka produce (acks=all)", measure_kafka),
    ("FHIR write-back (Flag)", measure_fhir_writeback),
]


def to_latex(results: dict) -> str:
    rows = []
    for name, r in results.items():
        if name.startswith("_") or r is None:
            continue
        rows.append([name, f"{r['p50']:.1f}", f"{r['p95']:.1f}",
                     f"{r['p99']:.1f}", f"{r['max']:.1f}"])
    return latex_table(
        rows,
        ["Transport stage", "p50 (ms)", "p95 (ms)", "p99 (ms)", "max (ms)"],
        caption=(
            "Transport latency measured against a running deployment. Each "
            "stage is timed over repeated round trips with synthetic payloads."
        ),
        label="tab:latency_transport",
        col_spec="@{}lrrrr@{}",
        note=(
            "Measured on a single host with all services co-located; a "
            "distributed deployment adds network latency to every stage."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MedROAD V3 transport latency")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/latency")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    logger.warning(
        "This posts synthetic observations to the configured FHIR server and "
        "Kafka broker under patient id %s. Do not run against a clinical "
        "instance.", BENCH_PATIENT,
    )

    results: dict = {}
    for name, fn in STAGES:
        logger.info("measuring: %s", name)
        results[name] = fn(args.n)

    print(f"\n{'transport stage':>26}  {'p50':>8} {'p95':>8} {'p99':>8}   (ms)")
    any_ok = False
    for name, r in results.items():
        if r is None:
            print(f"{name:>26}  {'unavailable':>28}")
            continue
        any_ok = True
        print(f"{name:>26}  {r['p50']:8.1f} {r['p95']:8.1f} {r['p99']:8.1f}")

    if not any_ok:
        print("\nNo stage could be measured. Bring the stack up first:")
        print("    docker compose up -d")
        raise SystemExit(1)

    save_results(results, args.out, "latency_transport")
    save_tex(to_latex(results), args.out, "latency_transport_table")
    print(f"\nwrote {Path(args.out)}/latency_transport.json and _table.tex")


if __name__ == "__main__":
    main()
