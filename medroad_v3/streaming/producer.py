"""
MedROAD V3 — Kafka Producer
Routes FHIR resources received by the webhook to the appropriate Kafka topic.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from kafka import KafkaProducer
from kafka.errors import KafkaError

from medroad_v3 import config

logger = logging.getLogger(__name__)


class FHIRKafkaProducer:
    def __init__(self) -> None:
        self._producer = KafkaProducer(
            bootstrap_servers = config.KAFKA_BOOTSTRAP,
            value_serializer  = lambda v: json.dumps(v).encode("utf-8"),
            acks              = "all",
            retries           = 5,
            retry_backoff_ms  = 200,
            linger_ms         = 5,
            compression_type  = "gzip",
        )
        logger.info("Kafka producer connected to %s", config.KAFKA_BOOTSTRAP)

    def _topic_for(self, resource_type: str, topic_hint: str | None) -> str:
        """
        Map FHIR resource type (or webhook hint) to Kafka topic.
        """
        if topic_hint == "vitals":
            return config.KAFKA_TOPIC_VITALS
        if topic_hint == "labs":
            return config.KAFKA_TOPIC_LABS
        if topic_hint == "meds":
            return config.KAFKA_TOPIC_MEDS
        # Fallback: classify by resourceType
        rt = resource_type.lower()
        if rt == "observation":
            # Further classify by category
            return config.KAFKA_TOPIC_VITALS
        if rt in ("medicationrequest", "medicationstatement"):
            return config.KAFKA_TOPIC_MEDS
        return config.KAFKA_TOPIC_DLQ

    def send(
        self,
        resource: dict[str, Any],
        topic_hint: str | None = None,
    ) -> None:
        """Send a FHIR resource to the appropriate Kafka topic."""
        rt    = resource.get("resourceType", "Unknown")
        topic = self._topic_for(rt, topic_hint)
        key   = self._extract_key(resource).encode("utf-8")

        try:
            self._producer.send(topic, key=key, value=resource)
            # Non-blocking; errors surface on flush
            logger.debug("Queued %s/%s → %s", rt, resource.get("id"), topic)
        except KafkaError as exc:
            logger.error("Kafka send error for %s/%s: %s", rt, resource.get("id"), exc)
            self._send_dlq(resource, str(exc))

    def _send_dlq(self, resource: dict[str, Any], reason: str) -> None:
        dlq_envelope = {"resource": resource, "error": reason}
        self._producer.send(config.KAFKA_TOPIC_DLQ, value=dlq_envelope)

    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()
        logger.info("Kafka producer closed")

    @staticmethod
    def _extract_key(resource: dict[str, Any]) -> str:
        """Use patient ID as partition key so all patient events go to same partition."""
        subject = resource.get("subject", {})
        ref = subject.get("reference", "")
        if ref.startswith("Patient/"):
            return ref.split("/")[-1]
        return resource.get("id", "unknown")
