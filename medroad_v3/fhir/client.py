"""
MedROAD V3 — FHIR R4 Client
OAuth2 password-flow authentication against OpenEMR's HAPI FHIR R4 server.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from medroad_v3 import config

logger = logging.getLogger(__name__)


class FHIRAuthError(Exception):
    pass


class FHIRRequestError(Exception):
    pass


class FHIRClient:
    """
    Thin FHIR R4 REST client for OpenEMR.
    Handles OAuth2 token acquisition, automatic refresh, and retry logic.
    """

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._session = self._build_session()

    # ── Session ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ── Authentication ────────────────────────────────────────────────────────

    def _acquire_token(self) -> str:
        """OAuth2 Resource Owner Password Credentials flow."""
        logger.debug("Acquiring FHIR OAuth2 token from %s", config.OAUTH_TOKEN_URL)
        resp = requests.post(
            config.OAUTH_TOKEN_URL,
            data={
                "grant_type":    "password",
                "client_id":     config.OPENEMR_CLIENT_ID,
                "client_secret": config.OPENEMR_CLIENT_SECRET,
                "username":      config.OPENEMR_USERNAME,
                "password":      config.OPENEMR_PASSWORD,
                "scope":         config.FHIR_SCOPE,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise FHIRAuthError(
                f"Token request failed {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 60
        logger.info("FHIR token acquired, expires in %ds", data.get("expires_in", 3600))
        return self._token

    @property
    def token(self) -> str:
        if self._token is None or time.time() >= self._token_expiry:
            self._acquire_token()
        return self._token  # type: ignore

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/fhir+json",
            "Accept":        "application/fhir+json",
        }

    # ── CRUD helpers ─────────────────────────────────────────────────────────

    def get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = f"{config.FHIR_BASE_URL}/{path.lstrip('/')}"
        resp = self._session.get(url, headers=self._auth_headers, params=params, timeout=30)
        self._raise_for_status(resp)
        return resp.json()

    def post(self, path: str, resource: dict[str, Any]) -> dict[str, Any]:
        url = f"{config.FHIR_BASE_URL}/{path.lstrip('/')}"
        resp = self._session.post(url, headers=self._auth_headers, json=resource, timeout=30)
        self._raise_for_status(resp)
        return resp.json()

    def put(self, path: str, resource: dict[str, Any]) -> dict[str, Any]:
        url = f"{config.FHIR_BASE_URL}/{path.lstrip('/')}"
        resp = self._session.put(url, headers=self._auth_headers, json=resource, timeout=30)
        self._raise_for_status(resp)
        return resp.json()

    def search(
        self, resource_type: str, params: dict[str, str]
    ) -> list[dict[str, Any]]:
        bundle = self.get(resource_type, params=params)
        entries = bundle.get("entry", [])
        return [e["resource"] for e in entries]

    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            raise FHIRRequestError(
                f"FHIR {resp.request.method} {resp.url} → {resp.status_code}: {resp.text[:400]}"
            )

    # ── Convenience reads ─────────────────────────────────────────────────────

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        return self.get(f"Patient/{patient_id}")

    def get_observations(
        self,
        patient_id: str,
        loinc_code: str | None = None,
        date_from: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "patient": patient_id,
            "_sort":   "-date",
            "_count":  "50",
        }
        if loinc_code:
            params["code"] = f"http://loinc.org|{loinc_code}"
        if date_from:
            params["date"] = f"ge{date_from}"
        return self.search("Observation", params)

    def get_medication_requests(self, patient_id: str) -> list[dict[str, Any]]:
        return self.search("MedicationRequest", {
            "patient": patient_id,
            "status":  "active",
            "_count":  "100",
        })

    def get_encounters(self, patient_id: str) -> list[dict[str, Any]]:
        return self.search("Encounter", {
            "patient": patient_id,
            "status":  "in-progress",
        })

    # ── Capability check ──────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            resp = self._session.get(
                f"{config.FHIR_BASE_URL}/metadata", timeout=10
            )
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False
