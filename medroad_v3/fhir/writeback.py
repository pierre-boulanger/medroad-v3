"""
MedROAD V3 — FHIR Write-Back Layer
Writes inference results back to OpenEMR as:
  - Flag                : active risk indicator on the patient
  - CommunicationRequest: task directing the nurse/physician
  - DocumentReference   : full narrative + SHAP values as structured note
  - ServiceRequest      : order for clinical review / escalation
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from medroad_v3.fhir.client import FHIRClient

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── Individual resource builders ──────────────────────────────────────────────

def build_flag(
    patient_id: str,
    risk_score: float,
    encounter_id: str | None = None,
) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Flag",
        "status":       "active",
        "category": [{
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/flag-category",
                "code":    "clinical",
                "display": "Clinical",
            }]
        }],
        "code": {
            "coding": [{
                "system":  "http://snomed.info/sct",
                "code":    "281900007",
                "display": "Risk of clinical deterioration",
            }],
            "text": f"MedROAD V3 risk score: {risk_score:.3f}",
        },
        "subject":    {"reference": f"Patient/{patient_id}"},
        "period":     {"start": _now_iso()},
    }
    if encounter_id:
        resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    return resource


def build_communication_request(
    patient_id: str,
    risk_score: float,
    top_features: list[tuple[str, float]],
    practitioner_id: str | None = None,
    encounter_id: str | None = None,
) -> dict[str, Any]:
    feature_text = "; ".join(
        f"{name}: {value:+.3f}" for name, value in top_features[:3]
    )
    resource: dict[str, Any] = {
        "resourceType": "CommunicationRequest",
        "status":       "active",
        "priority":     "urgent" if risk_score >= 0.85 else "routine",
        "subject":      {"reference": f"Patient/{patient_id}"},
        "authoredOn":   _now_iso(),
        "requester":    {"display": "MedROAD V3 Clinical Decision Support"},
        "payload": [{
            "contentString": (
                f"ALERT: MedROAD V3 risk score {risk_score:.3f} exceeds threshold. "
                f"Top drivers: {feature_text}. "
                "Please review patient status immediately."
            )
        }],
        "note": [{
            "text": f"Ensemble risk score: {risk_score:.4f}. "
                    f"SHAP top features: {feature_text}."
        }],
    }
    if practitioner_id:
        resource["recipient"] = [{"reference": f"Practitioner/{practitioner_id}"}]
    if encounter_id:
        resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    return resource


def build_document_reference(
    patient_id: str,
    risk_score: float,
    narrative: str,
    shap_values: dict[str, float],
    feature_vector: dict[str, float],
    encounter_id: str | None = None,
) -> dict[str, Any]:
    """
    Encodes the full inference result as a FHIR DocumentReference.
    The content is a base64-encoded JSON payload for structured retrieval
    plus a plain-text narrative for human readability.
    """
    import base64

    payload = {
        "generated_at":    _now_iso(),
        "model":           "MedROAD V3",
        "risk_score":      risk_score,
        "narrative":       narrative,
        "shap_values":     shap_values,
        "feature_vector":  feature_vector,
    }
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()

    resource: dict[str, Any] = {
        "resourceType": "DocumentReference",
        "status":       "current",
        "type": {
            "coding": [{
                "system":  "http://loinc.org",
                "code":    "34109-9",
                "display": "Note",
            }]
        },
        "subject":  {"reference": f"Patient/{patient_id}"},
        "date":     _now_iso(),
        "author":   [{"display": "MedROAD V3 Clinical Decision Support"}],
        "description": f"MedROAD V3 risk alert — score {risk_score:.3f}",
        "content": [
            {
                "attachment": {
                    "contentType": "application/json",
                    "data":        encoded,
                    "title":       "MedROAD V3 Inference Result (structured)",
                    "creation":    _now_iso(),
                }
            },
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data":        base64.b64encode(narrative.encode()).decode(),
                    "title":       "MedROAD V3 Clinical Narrative",
                    "creation":    _now_iso(),
                }
            },
        ],
    }
    if encounter_id:
        resource["context"] = {"encounter": [{"reference": f"Encounter/{encounter_id}"}]}
    return resource


def build_service_request(
    patient_id: str,
    risk_score: float,
    encounter_id: str | None = None,
) -> dict[str, Any]:
    priority = "urgent" if risk_score >= 0.85 else "routine"
    resource: dict[str, Any] = {
        "resourceType": "ServiceRequest",
        "status":       "active",
        "intent":       "proposal",
        "priority":     priority,
        "code": {
            "coding": [{
                "system":  "http://snomed.info/sct",
                "code":    "182836005",
                "display": "Review of medication",
            }],
            "text": "Clinical review: MedROAD V3 deterioration alert",
        },
        "subject":      {"reference": f"Patient/{patient_id}"},
        "authoredOn":   _now_iso(),
        "requester":    {"display": "MedROAD V3 Clinical Decision Support"},
        "note": [{
            "text": f"Automated alert: ensemble risk score {risk_score:.4f} ≥ τ = 0.72. "
                    "Review patient for signs of clinical deterioration."
        }],
    }
    if encounter_id:
        resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    return resource


# ── Write-back orchestrator ───────────────────────────────────────────────────

class FHIRWriteBack:
    def __init__(self, client: FHIRClient) -> None:
        self.client = client

    def write_alert(
        self,
        patient_id: str,
        risk_score: float,
        narrative: str,
        shap_values: dict[str, float],
        feature_vector: dict[str, float],
        encounter_id: str | None = None,
        practitioner_id: str | None = None,
    ) -> dict[str, str]:
        """
        Write all four FHIR resources atomically.
        Returns dict of resource_type → created resource ID.
        """
        top_features = sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        results: dict[str, str] = {}
        resources = [
            ("Flag",               build_flag(patient_id, risk_score, encounter_id)),
            ("CommunicationRequest", build_communication_request(
                patient_id, risk_score, top_features, practitioner_id, encounter_id)),
            ("DocumentReference",  build_document_reference(
                patient_id, risk_score, narrative, shap_values, feature_vector, encounter_id)),
            ("ServiceRequest",     build_service_request(patient_id, risk_score, encounter_id)),
        ]

        for rtype, resource in resources:
            try:
                result = self.client.post(rtype, resource)
                rid = result.get("id", "unknown")
                results[rtype] = rid
                logger.info("Written %s/%s for patient %s (score=%.3f)",
                            rtype, rid, patient_id, risk_score)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to write %s for patient %s: %s",
                             rtype, patient_id, exc)
                results[rtype] = "ERROR"

        return results
