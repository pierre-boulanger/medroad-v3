"""
MedROAD V3 — FHIR Subscription Management
Creates FHIR R4 REST-hook Subscriptions on OpenEMR for Observation,
MedicationRequest, and Encounter resources.
"""
from __future__ import annotations

import logging
from typing import Any

from medroad_v3 import config
from medroad_v3.fhir.client import FHIRClient

logger = logging.getLogger(__name__)


SUBSCRIPTIONS: list[dict[str, Any]] = [
    {
        "id":      "medroad-vitals-sub",
        "status":  "requested",
        "reason":  "MedROAD V3 real-time vital sign streaming",
        "criteria": "Observation?category=vital-signs",
        "channel": {
            "type":    "rest-hook",
            "endpoint": f"{config.WEBHOOK_URL}",
            "payload":  "application/fhir+json",
            "header":  [f"X-MedROAD-Secret: {config.WEBHOOK_SECRET}",
                        "X-MedROAD-Topic: vitals"],
        },
    },
    {
        "id":      "medroad-labs-sub",
        "status":  "requested",
        "reason":  "MedROAD V3 real-time laboratory streaming",
        "criteria": "Observation?category=laboratory",
        "channel": {
            "type":    "rest-hook",
            "endpoint": f"{config.WEBHOOK_URL}",
            "payload":  "application/fhir+json",
            "header":  [f"X-MedROAD-Secret: {config.WEBHOOK_SECRET}",
                        "X-MedROAD-Topic: labs"],
        },
    },
    {
        "id":      "medroad-meds-sub",
        "status":  "requested",
        "reason":  "MedROAD V3 active medication tracking",
        "criteria": "MedicationRequest?status=active",
        "channel": {
            "type":    "rest-hook",
            "endpoint": f"{config.WEBHOOK_URL}",
            "payload":  "application/fhir+json",
            "header":  [f"X-MedROAD-Secret: {config.WEBHOOK_SECRET}",
                        "X-MedROAD-Topic: meds"],
        },
    },
]


def _build_subscription(definition: dict[str, Any]) -> dict[str, Any]:
    return {
        "resourceType": "Subscription",
        "id":           definition["id"],
        "status":       definition["status"],
        "reason":       definition["reason"],
        "criteria":     definition["criteria"],
        "channel":      definition["channel"],
    }


def setup_subscriptions(client: FHIRClient) -> list[str]:
    """
    Idempotently create all MedROAD Subscriptions on the FHIR server.
    Returns list of Subscription IDs that were created or already exist.
    """
    created: list[str] = []
    for defn in SUBSCRIPTIONS:
        sub_id = defn["id"]
        try:
            # Check if it already exists
            existing = client.get(f"Subscription/{sub_id}")
            logger.info("Subscription %s already exists (status=%s)",
                        sub_id, existing.get("status"))
            created.append(sub_id)
            continue
        except Exception:  # noqa: BLE001
            pass  # Does not exist yet; create it

        resource = _build_subscription(defn)
        try:
            result = client.put(f"Subscription/{sub_id}", resource)
            logger.info("Created Subscription %s → %s",
                        sub_id, result.get("id", "?"))
            created.append(sub_id)
        except Exception as exc:
            logger.error("Failed to create Subscription %s: %s", sub_id, exc)

    return created


def teardown_subscriptions(client: FHIRClient) -> None:
    """Remove all MedROAD Subscriptions (for clean shutdown or testing)."""
    for defn in SUBSCRIPTIONS:
        sub_id = defn["id"]
        try:
            # Set status to off
            resource = _build_subscription(defn)
            resource["status"] = "off"
            client.put(f"Subscription/{sub_id}", resource)
            logger.info("Deactivated Subscription %s", sub_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not deactivate %s: %s", sub_id, exc)
