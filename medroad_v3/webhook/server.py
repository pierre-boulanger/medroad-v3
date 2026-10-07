"""
MedROAD V3 — FHIR Webhook Server (FastAPI)
Receives REST-hook POST requests from OpenEMR FHIR Subscriptions,
validates HMAC-SHA256, and routes to Kafka.
"""
from __future__ import annotations

import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from medroad_v3 import config
from medroad_v3.streaming.producer import FHIRKafkaProducer

logger = logging.getLogger(__name__)

_producer: FHIRKafkaProducer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _producer
    _producer = FHIRKafkaProducer()
    logger.info("MedROAD V3 webhook server started on :%d", config.WEBHOOK_PORT)
    yield
    if _producer:
        _producer.close()
    logger.info("MedROAD V3 webhook server stopped")


app = FastAPI(
    title       = "MedROAD V3 FHIR Webhook",
    description = "REST-hook receiver for OpenEMR FHIR R4 Subscriptions",
    version     = "2.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["POST"],
    allow_headers  = ["*"],
)


# ── HMAC verification ─────────────────────────────────────────────────────────

def _verify_hmac(body: bytes, secret_header: str | None) -> bool:
    """
    OpenEMR sends X-MedROAD-Secret as a plain shared secret (not HMAC).
    For production, configure HMAC-SHA256 in the Subscription channel header.
    """
    if not config.WEBHOOK_SECRET:
        return True  # No secret configured — allow all (dev mode)
    return hmac.compare_digest(
        secret_header or "",
        config.WEBHOOK_SECRET,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "kafka":  _producer is not None,
        "ts":     time.time(),
    }


@app.post("/fhir/webhook")
async def fhir_webhook(request: Request) -> Response:
    """
    Receive a FHIR REST-hook notification from OpenEMR.
    The request body is a FHIR Resource (or Bundle) in application/fhir+json.
    """
    t0 = time.perf_counter()

    # Validate shared secret
    secret  = request.headers.get("X-MedROAD-Secret")
    body    = await request.body()
    if not _verify_hmac(body, secret):
        logger.warning("Webhook HMAC verification failed")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid webhook secret")

    # Parse body
    try:
        resource = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.error("Webhook body is not valid JSON: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Body must be application/fhir+json") from exc

    rt         = resource.get("resourceType", "Unknown")
    topic_hint = request.headers.get("X-MedROAD-Topic")

    # Handle Bundle: unwrap entries
    resources: list[dict[str, Any]] = []
    if rt == "Bundle":
        for entry in resource.get("entry", []):
            res = entry.get("resource")
            if res:
                resources.append(res)
    else:
        resources.append(resource)

    # Route to Kafka
    if _producer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Kafka producer not ready")

    dispatched = 0
    for res in resources:
        try:
            _producer.send(res, topic_hint=topic_hint)
            dispatched += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to route %s: %s", res.get("resourceType"), exc)

    _producer.flush()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Webhook: %d resource(s) dispatched in %.1f ms", dispatched, elapsed_ms)

    return Response(status_code=status.HTTP_200_OK)


@app.post("/fhir/webhook/test")
async def test_webhook(request: Request) -> dict[str, Any]:
    """Dev endpoint: echo back the parsed resource without routing to Kafka."""
    body = await request.body()
    try:
        resource = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid JSON") from exc
    return {
        "received":     resource.get("resourceType"),
        "id":           resource.get("id"),
        "patient":      resource.get("subject", {}).get("reference"),
        "kafka_ready":  _producer is not None,
    }


# ── Uvicorn entry point ───────────────────────────────────────────────────────

def serve() -> None:
    import uvicorn
    uvicorn.run(
        "medroad_v3.webhook.server:app",
        host  = config.WEBHOOK_HOST,
        port  = config.WEBHOOK_PORT,
        log_level = "info",
        reload    = False,
    )
