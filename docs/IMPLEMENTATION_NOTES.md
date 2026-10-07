# MedROAD V3

**Real-Time FHIR Sensor Fusion for Clinical Decision Support**

Multi-modal FHIR R4 streaming architecture integrating OpenEMR, Apache Kafka,
and an ensemble clinical risk inference engine (XGBoost + LSTM + Transformer)
with SHAP explainability and LLM-generated bedside narratives.

---

## Architecture

```
OpenEMR (HAPI FHIR R4)
       │  FHIR Subscription REST-hook
       ▼
Webhook Server (FastAPI :8001)
       │  HMAC-verified POST
       ▼
Apache Kafka  ─────────────────────────────────────
  vitals · labs · meds · encounters · dlq
       │
       ▼
Faust Stream Processor
  5-minute tumbling windows · RocksDB state
       │
       ▼
Inference Engine (55-element feature vector)
  ┌─────────────┐ ┌──────────┐ ┌─────────────┐
  │  XGBoost    │ │   LSTM   │ │ Transformer │
  │ + SHAP      │ │ hidden128│ │ d_model=64  │
  └─────┬───────┘ └────┬─────┘ └──────┬──────┘
        └──────────────┼──────────────┘
                       ▼
             Logistic Regression Meta-Learner
             Isotonic Calibration  τ = 0.72
                       │
              R ≥ 0.72?│
                       ▼
              Claude API Narrative (3 sentences)
                       │
                       ▼
OpenEMR Write-Back:  Flag · CommunicationRequest
                     DocumentReference · ServiceRequest
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Anthropic API key

### 2. Configuration

```bash
cp .env.example .env
# Edit .env with your OpenEMR credentials and Anthropic API key
```

### 3. Start infrastructure

```bash
docker compose up -d mariadb openemr zookeeper kafka
# Wait for OpenEMR to initialise (~2 min)
docker compose logs -f openemr
```

### 4. Register OAuth2 client in OpenEMR

Navigate to `http://localhost:8300/interface/smart/register-app.php` and register
a new client application. Copy the client_id and client_secret to `.env`.

### 5. Train models

```bash
pip install -r requirements.txt
python -m medroad_v3.training.train --data /path/to/mimic_iv_windows.csv
```

**Expected MIMIC-IV CSV columns:**
`patient_id, window_start, label, <55 feature columns in order>`

Feature column order matches `medroad_v3/features/engineering.py`:
- `vital_heart_rate` … `vital_pain_score` (8)
- `lab_troponin_i` … `lab_magnesium` (12)
- `meta_vital_cov` … `meta_icu_los` (8)
- `miss_vital_*` (8), `miss_lab_*` (12)
- `delta_troponin_i`, `delta_bnp`, `delta_lactate` (3)
- `sig`, `bcr` (2)
- `hour_sin`, `hour_cos` (2)

### 6. Start MedROAD V3 services

```bash
docker compose up -d webhook stream

# Create FHIR Subscriptions on OpenEMR
docker compose --profile setup up setup
```

### 7. Verify

```bash
# Health check
curl http://localhost:8001/health

# Synthetic inference test (requires trained models)
python -m medroad_v3.main infer-test
```

---

## Feature Vector (55 elements)

| Index | Group              | Count | Description                                      |
|-------|--------------------|-------|--------------------------------------------------|
| 0–7   | Vital signs        | 8     | HR, SBP, DBP, SpO₂, RR, Temp, GCS, Pain         |
| 8–19  | Laboratory values  | 12    | TropI, BNP, Creat, K, Na, HCO₃, pCO₂, pH, Lac, Hct, WBC, Mg |
| 20–27 | Metadata           | 8     | vital_cov, lab_cov, abg_present, doc_lag, vital_recency, med_burden, iv_drip, icu_los |
| 28–35 | Vital missingness  | 8     | Binary indicator per vital LOINC                 |
| 36–47 | Lab missingness    | 12    | Binary indicator per lab LOINC                   |
| 48–50 | Delta features     | 3     | ΔTropI, ΔBNP, ΔLactate (vs. prior 24h period)   |
| 51–52 | Interaction scores | 2     | SIG (Strong Ion Gap), BCR (BNP/Creatinine ratio) |
| 53–54 | Temporal context   | 2     | Hour-of-day sin/cos                              |

---

## Model Specifications

| Model       | Architecture                                        | Calibration       |
|-------------|-----------------------------------------------------|-------------------|
| XGBoost     | 500 trees, depth 6, lr 0.05, early stopping         | Platt scaling     |
| LSTM        | 2-layer, hidden=128, dropout=0.3, AdamW + cosine LR | Temperature scaling |
| Transformer | d_model=64, nhead=4, 2 encoder layers, sinusoidal PE | Temperature scaling |
| Meta-learner | LogisticRegression over (p_xgb, p_lstm, p_tf)      | Isotonic regression |

Alert threshold τ = 0.72 (F1-maximising on MIMIC-IV validation set).

---

## Latency Profile (20-patient simulated workload)

| Stage                        | Latency   |
|------------------------------|-----------|
| FHIR webhook → Kafka          | ~12 ms    |
| Feature engineering           | ~8 ms     |
| XGBoost + SHAP                | ~35 ms    |
| LSTM inference                | ~5 ms     |
| Transformer inference         | ~6 ms     |
| Meta-learner                  | <1 ms     |
| FHIR write-back (4 resources) | ~45 ms    |
| **Total (no narrative)**      | **~112 ms** |
| Claude API narrative          | ~1,800 ms |
| **Total (with narrative)**    | **~1,912 ms** |

---

## Regulatory

Positioned as Health Canada Class II Software as a Medical Device (SaMD)
under the Software Life Cycle Processes guidance (IVD / Non-IVD hybrid).
IEC 62304 compliant software life cycle documentation available on request.
MIMIC-IV validation cohort: 12,847 patient-hours, AUROC 0.87 (ensemble).

---

---

## Code → Paper Section Map

| Paper section | Module | Implements |
|---|---|---|
| §3.1 FHIR Event Source | `fhir/client.py`, `fhir/subscriptions.py` | OAuth2 client, R4B Subscription criteria, REST-hook registration |
| §3.2 Message Broker | `streaming/producer.py` | Kafka topic routing, `patient_id` partition keying, DLQ |
| §3.3 Stream Processing | `streaming/faust_app.py` | 5-min tumbling windows, RocksDB state, completeness guard |
| §3.4 Inference & Decision | `inference/engine.py` | Orchestration, latency instrumentation |
| §3.5 FHIR Write-Back | `fhir/writeback.py` | Flag, CommunicationRequest, DocumentReference, ServiceRequest + upsert |
| **§3.6 RPM Extension** | **`rpm/mapping.py`** | **Table 3 ICU→wearable mapping, window sizing, coverage estimation** |
| §4 Feature Engineering | `features/engineering.py` | 55-element vector: 8 vitals, 12 labs, 8 meta, 20 missingness, 3 deltas, SIG/BCR, temporal |
| §5 Ensemble & SHAP | `models/xgboost_model.py`, `models/deep_models.py`, `models/ensemble.py` | XGBoost+TreeExplainer, LSTM(2×128), Transformer(d=64), LR meta-learner |
| **§6 Model Calibration** | **`models/calibration.py`** | **ECE/MCE/Brier/Hosmer-Lemeshow, Platt, temperature, isotonic, shift detection, threshold selection** |
| §7 Narrative Generation | `narrative/generator.py` | Claude prompt contract, SHAP grounding, deterministic fallback |
| §8 FHIR Write-Back | `fhir/writeback.py` | Resource construction and POST |
| §9 Evaluation | `training/train.py`, `inference/engine.py` | Latency instrumentation, OOF training |

### Calibration module (§6)

```python
from medroad_v3.models.calibration import (
    calibration_report, IsotonicCalibrator, select_threshold_f1
)

report = calibration_report(y_true, y_prob)   # ECE, MCE, Brier, AUROC, H-L
cal = IsotonicCalibrator().fit(val_prob, val_y)
tau, f1 = select_threshold_f1(y_true, cal.transform(y_prob))
```

### RPM module (§3.6)

```python
from medroad_v3.rpm.mapping import DEVICE_PROFILES, describe_deployment

hf_kit = [p for p in DEVICE_PROFILES
          if p.name in ("Pulse oximeter", "BP cuff", "Smart scale")]
describe_deployment(hf_kit)
# -> window_seconds: 86400, expected_coverage: 0.556,
#    unavailable_features: ['troponin_i', 'ventilator_pressure']
```

Note that `recommended_window_seconds` returns 24 h for any kit containing a
daily-cadence device. The five-minute ICU window assumes dense telemetry; using
it unchanged on an RPM cohort produces mostly-empty windows. This is the single
most important configuration change when moving from bedside to remote
deployment.

---

## Citation

```bibtex
@article{boulanger2026medroad,
  author  = {Boulanger, Pierre},
  title   = {Multi-Modal {FHIR} Sensor Fusion for Real-Time Clinical Decision
             Support: From Hospital Bedside to Remote Patient Monitoring},
  journal = {Special Issue: Sensor Fusion for Telemedicine},
  year    = {2026},
}
```
