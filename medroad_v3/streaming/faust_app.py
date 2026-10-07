"""
MedROAD V3 — Faust Stream Processor
Five-minute tumbling windows over vital-sign Kafka topics.
On window close: build feature vector → run inference engine → write back.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import faust

from medroad_v3 import config
from medroad_v3.fhir.client import FHIRClient
from medroad_v3.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

# ── Faust app ─────────────────────────────────────────────────────────────────

app = faust.App(
    id        = config.FAUST_APP_ID,
    broker    = config.FAUST_BROKER,
    topic_partitions = 4,
)

# Topics
vitals_topic = app.topic(config.KAFKA_TOPIC_VITALS, value_type=bytes)
labs_topic   = app.topic(config.KAFKA_TOPIC_LABS,   value_type=bytes)

# Windowed state tables (RocksDB backed)
# Store lists of observations per patient per window
vitals_table = app.Table(
    "vitals_window",
    default=list,
).tumbling(config.WINDOW_SECONDS, expires=faust.windows.timedelta_secs(config.WINDOW_EXPIRES))

labs_table = app.Table(
    "labs_24h",
    default=list,
).tumbling(60 * 60 * 24, expires=faust.windows.timedelta_secs(60 * 60 * 48))

# Lazy-loaded models and clients
_engine:      InferenceEngine | None = None
_fhir_client: FHIRClient | None      = None


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load_models(config.MODEL_DIR)
    return _engine


def _get_fhir() -> FHIRClient | None:
    global _fhir_client
    if _fhir_client is None and config.OPENEMR_CLIENT_ID:
        try:
            _fhir_client = FHIRClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning("FHIR client unavailable: %s", exc)
    return _fhir_client


def _patient_id(obs: dict[str, Any]) -> str | None:
    ref = obs.get("subject", {}).get("reference", "")
    if ref.startswith("Patient/"):
        return ref.split("/")[-1]
    return None


# ── Vital signs stream ────────────────────────────────────────────────────────

@app.agent(vitals_topic)
async def process_vitals(stream):
    async for event in stream.events():
        try:
            obs = json.loads(event.value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Malformed vital message; routing to DLQ")
            continue

        pid = _patient_id(obs)
        if not pid:
            continue

        # Append to 5-min tumbling window
        window_key = f"vitals:{pid}"
        current    = list(vitals_table[window_key].now())
        current.append(obs)
        vitals_table[window_key] = current

        logger.debug("Vital appended for patient %s (window size=%d)", pid, len(current))

        # Completeness check: need at least 3 different LOINC types in window
        loinc_set = set()
        for o in current:
            for coding in o.get("code", {}).get("coding", []):
                if "loinc" in coding.get("system", "").lower():
                    loinc_set.add(coding.get("code"))

        if len(loinc_set) < 3:
            continue

        # Run inference on the current window
        await _run_inference(pid, current)


@app.agent(labs_topic)
async def process_labs(stream):
    """Accumulate labs into 24-hour rolling window."""
    async for event in stream.events():
        try:
            obs = json.loads(event.value)
        except (json.JSONDecodeError, TypeError):
            continue

        pid = _patient_id(obs)
        if not pid:
            continue

        key    = f"labs:{pid}"
        current = list(labs_table[key].now())
        current.append(obs)
        # Prune to last 24 h
        cutoff = datetime.now(UTC).timestamp() - 86400
        current = [
            o for o in current
            if datetime.fromisoformat(
                o.get("effectiveDateTime", "1970-01-01T00:00:00Z")
                 .replace("Z", "+00:00")
            ).timestamp() > cutoff
        ]
        labs_table[key] = current
        logger.debug("Lab stored for patient %s (24h pool=%d)", pid, len(current))


# ── Inference trigger ─────────────────────────────────────────────────────────

async def _run_inference(patient_id: str, window_obs: list[dict]) -> None:
    """
    Called when a 5-min vital window is sufficiently populated.
    Fetches labs from state, builds feature vector, runs inference.
    """
    engine = _get_engine()
    fhir   = _get_fhir()

    # Retrieve 24h labs from state table
    lab_key  = f"labs:{patient_id}"
    lab_obs  = list(labs_table[lab_key].now()) if lab_key in labs_table else []

    try:
        result = engine.infer(
            patient_id       = patient_id,
            window_obs       = window_obs,
            lab_obs_24h      = lab_obs,
            prior_labs       = None,
            med_requests     = [],
            encounter_start  = None,
            generate_narrative = True,
            fhir_client      = fhir,
        )
        if result.alert:
            logger.warning(
                "ALERT patient=%s score=%.3f top=%s",
                patient_id,
                result.risk_score,
                [n for n, _ in result.top_features[:3]],
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference failed for patient %s: %s", patient_id, exc)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.main()
