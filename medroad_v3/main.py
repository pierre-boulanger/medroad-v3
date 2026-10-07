"""
MedROAD V3 — Main Entry Point
Usage:
    python -m medroad_v3.main webhook      # Start FHIR webhook receiver
    python -m medroad_v3.main stream       # Start Faust stream processor
    python -m medroad_v3.main setup        # Create FHIR Subscriptions on OpenEMR
    python -m medroad_v3.main teardown     # Remove FHIR Subscriptions
    python -m medroad_v3.main infer-test   # Run a synthetic inference sanity check
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("medroad_v3")


def cmd_webhook() -> None:
    from medroad_v3.webhook.server import serve
    serve()


def cmd_stream() -> None:
    from medroad_v3.streaming.faust_app import app
    app.main()


def cmd_setup() -> None:
    from medroad_v3.fhir.client import FHIRClient
    from medroad_v3.fhir.subscriptions import setup_subscriptions
    client = FHIRClient()
    if not client.ping():
        logger.error("Cannot reach OpenEMR FHIR server — check OPENEMR_BASE_URL")
        sys.exit(1)
    ids = setup_subscriptions(client)
    logger.info("Subscriptions created: %s", ids)


def cmd_teardown() -> None:
    from medroad_v3.fhir.client import FHIRClient
    from medroad_v3.fhir.subscriptions import teardown_subscriptions
    client = FHIRClient()
    teardown_subscriptions(client)


def cmd_infer_test() -> None:
    """
    Synthetic sanity check: build a random 55-element feature vector,
    run through all models (must be pre-trained), print result.
    """
    from medroad_v3 import config
    from medroad_v3.inference.engine import InferenceEngine

    engine = InferenceEngine()
    try:
        engine.load_models(config.MODEL_DIR)
    except FileNotFoundError as exc:
        logger.error("%s\nRun training first: python -m medroad_v3.training.train --data <csv>", exc)
        sys.exit(1)

    # Generate synthetic observation list (3 vital LOINC codes present)
    from datetime import datetime
    now = datetime.now(UTC).isoformat()
    obs_template = [
        {"resourceType": "Observation", "status": "final",
         "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]},
         "subject": {"reference": "Patient/TEST001"},
         "effectiveDateTime": now,
         "valueQuantity": {"value": 82.0, "unit": "/min"}},
        {"resourceType": "Observation", "status": "final",
         "code": {"coding": [{"system": "http://loinc.org", "code": "59408-5"}]},
         "subject": {"reference": "Patient/TEST001"},
         "effectiveDateTime": now,
         "valueQuantity": {"value": 96.0, "unit": "%"}},
        {"resourceType": "Observation", "status": "final",
         "code": {"coding": [{"system": "http://loinc.org", "code": "55284-4"}]},
         "subject": {"reference": "Patient/TEST001"},
         "effectiveDateTime": now,
         "valueQuantity": {"value": 118.0, "unit": "mmHg"}},
    ]

    result = engine.infer(
        patient_id      = "TEST001",
        window_obs      = obs_template,
        lab_obs_24h     = [],
        prior_labs      = None,
        med_requests    = [],
        encounter_start = None,
        generate_narrative = False,
        fhir_client     = None,
    )

    print("\n=== MedROAD V3 Inference Test ===")
    print(f"  XGBoost:      {result.xgb_score:.4f}")
    print(f"  LSTM:         {result.lstm_score:.4f}")
    print(f"  Transformer:  {result.transformer_score:.4f}")
    print(f"  Ensemble R:   {result.risk_score:.4f}")
    print(f"  Alert:        {result.alert}")
    print(f"  Top features: {result.top_features[:3]}")
    print("=================================\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

COMMANDS = {
    "webhook":    cmd_webhook,
    "stream":     cmd_stream,
    "setup":      cmd_setup,
    "teardown":   cmd_teardown,
    "infer-test": cmd_infer_test,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedROAD V3 — Real-Time FHIR Sensor Fusion CDS",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = "\n".join(f"  {cmd}" for cmd in COMMANDS),
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
