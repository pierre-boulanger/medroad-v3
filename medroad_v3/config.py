"""
MedROAD V3 — Configuration
All settings loaded from environment variables / .env file.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.environ.get(key, default))


# ── OpenEMR / FHIR ──────────────────────────────────────────────────────────

OPENEMR_BASE_URL: str       = _env("OPENEMR_BASE_URL", "http://localhost:8300")
FHIR_BASE_URL: str          = _env("FHIR_BASE_URL",    f"{OPENEMR_BASE_URL}/apis/default/fhir")
OAUTH_TOKEN_URL: str        = _env("OAUTH_TOKEN_URL",   f"{OPENEMR_BASE_URL}/oauth2/default/token")
OPENEMR_CLIENT_ID: str      = _env("OPENEMR_CLIENT_ID")
OPENEMR_CLIENT_SECRET: str  = _env("OPENEMR_CLIENT_SECRET")
OPENEMR_USERNAME: str       = _env("OPENEMR_USERNAME",  "admin")
OPENEMR_PASSWORD: str       = _env("OPENEMR_PASSWORD")
FHIR_SCOPE: str             = _env("FHIR_SCOPE", "openid fhirUser offline_access user/*.read user/*.write")

# Webhook endpoint that OpenEMR will POST to
WEBHOOK_HOST: str  = _env("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT: int  = _env_int("WEBHOOK_PORT", 8001)
WEBHOOK_URL: str   = _env("WEBHOOK_URL", "http://host.docker.internal:8001/fhir/webhook")
WEBHOOK_SECRET: str = _env("WEBHOOK_SECRET", "change-me-in-production")

# ── Kafka ────────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP: str     = _env("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_VITALS: str  = _env("KAFKA_TOPIC_VITALS",  "fhir.vitals")
KAFKA_TOPIC_LABS: str    = _env("KAFKA_TOPIC_LABS",    "fhir.labs")
KAFKA_TOPIC_MEDS: str    = _env("KAFKA_TOPIC_MEDS",    "fhir.meds")
KAFKA_TOPIC_ALERTS: str  = _env("KAFKA_TOPIC_ALERTS",  "fhir.alerts")
KAFKA_TOPIC_DLQ: str     = _env("KAFKA_TOPIC_DLQ",     "fhir.dlq")
KAFKA_CONSUMER_GROUP: str = _env("KAFKA_CONSUMER_GROUP", "medroad-inference")

# ── Faust stream processing ──────────────────────────────────────────────────

FAUST_BROKER: str        = _env("FAUST_BROKER", f"kafka://{KAFKA_BOOTSTRAP}")
FAUST_APP_ID: str        = _env("FAUST_APP_ID", "medroad-v3")
WINDOW_SECONDS: int      = _env_int("WINDOW_SECONDS", 300)   # 5-minute tumbling window
WINDOW_EXPIRES: int      = _env_int("WINDOW_EXPIRES", 600)   # Keep state 10 min after close

# ── Models ───────────────────────────────────────────────────────────────────

MODEL_DIR: str           = _env("MODEL_DIR", "./models_saved")
N_FEATURES: int          = 86
SEQ_LEN: int             = _env_int("SEQ_LEN", 12)           # observations per window
# Weight-change thresholds (kg). A gain above WEIGHT_ALERT_72H is the
# classic heart-failure decompensation signal.
WEIGHT_ALERT_24H: float  = _env_float("WEIGHT_ALERT_24H", 1.0)
WEIGHT_ALERT_72H: float  = _env_float("WEIGHT_ALERT_72H", 2.0)
WEIGHT_DELTA_SCALE: float = _env_float("WEIGHT_DELTA_SCALE", 10.0)

RISK_THRESHOLD: float    = _env_float("RISK_THRESHOLD", 0.72) # F1-maximising τ

# XGBoost
XGB_N_ESTIMATORS: int    = _env_int("XGB_N_ESTIMATORS", 500)
XGB_MAX_DEPTH: int       = _env_int("XGB_MAX_DEPTH", 6)
XGB_LR: float            = _env_float("XGB_LR", 0.05)
XGB_SUBSAMPLE: float     = _env_float("XGB_SUBSAMPLE", 0.8)
XGB_COLSAMPLE: float     = _env_float("XGB_COLSAMPLE", 0.8)
XGB_EARLY_STOP: int      = _env_int("XGB_EARLY_STOP", 50)

# LSTM
LSTM_HIDDEN: int         = _env_int("LSTM_HIDDEN", 128)
LSTM_LAYERS: int         = _env_int("LSTM_LAYERS", 2)
LSTM_DROPOUT: float      = _env_float("LSTM_DROPOUT", 0.3)
# Lowered alongside the Transformer. At 1e-3 with a large positive class
# weight the recurrent net saturates and collapses to a constant
# prediction, which shows up as AUROC exactly 0.5.
LSTM_LR: float           = _env_float("LSTM_LR", 3e-4)
LSTM_EPOCHS: int         = _env_int("LSTM_EPOCHS", 50)
LSTM_BATCH: int          = _env_int("LSTM_BATCH", 256)

# Transformer
TF_D_MODEL: int          = _env_int("TF_D_MODEL", 64)
TF_NHEAD: int            = _env_int("TF_NHEAD", 4)
TF_NUM_LAYERS: int       = _env_int("TF_NUM_LAYERS", 2)
TF_DIM_FF: int           = _env_int("TF_DIM_FF", 256)
TF_DROPOUT: float        = _env_float("TF_DROPOUT", 0.1)
# Transformers tolerate far less learning rate than recurrent nets early in
# training. 3e-4 with pre-norm is stable without a warmup schedule.
TF_LR: float             = _env_float("TF_LR", 3e-4)
TF_EPOCHS: int           = _env_int("TF_EPOCHS", 50)
TF_BATCH: int            = _env_int("TF_BATCH", 256)

# ── Anthropic / Claude ───────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str   = _env("ANTHROPIC_API_KEY")
CLAUDE_MODEL: str        = _env("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_TOKENS: int   = _env_int("CLAUDE_MAX_TOKENS", 300)
# Well below the default. A clinical summary should be reproducible from
# the same inputs; linguistic variety is not a virtue here.
CLAUDE_TEMPERATURE: float = _env_float("CLAUDE_TEMPERATURE", 0.2)

# ── LOINC codes ──────────────────────────────────────────────────────────────

LOINC_VITALS: dict[str, str] = {
    "heart_rate":       "8867-4",
    "systolic_bp":      "55284-4",
    "diastolic_bp":     "8462-4",
    "spo2":             "59408-5",
    "respiratory_rate": "9279-1",
    "temperature":      "8310-5",
    "weight":           "29463-7",
    "gcs":              "59560-5",
    "pain_score":       "59574-4",
}

LOINC_LABS: dict[str, str] = {
    "troponin_i":  "10839-9",
    "bnp":         "30522-7",
    "creatinine":  "2160-0",
    "potassium":   "2823-3",
    "sodium":      "2090-9",
    "bicarbonate": "14682-9",
    "pco2":        "2019-8",
    "ph":          "59578-5",
    "lactate":     "2532-0",
    "hematocrit":  "4544-3",
    "wbc":         "6690-2",
    "magnesium":   "2601-3",
}

LOINC_REVERSE: dict[str, str] = {v: k for k, v in {**LOINC_VITALS, **LOINC_LABS}.items()}
