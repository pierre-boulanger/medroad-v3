# MedROAD V3

**Multi-modal FHIR sensor fusion for real-time clinical decision support — from hospital bedside to remote patient monitoring.**

[![CI](https://github.com/pboulanger/medroad-v3/actions/workflows/ci.yml/badge.svg)](https://github.com/pboulanger/medroad-v3/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation accompanying the paper submitted to the *Sensors*
Special Issue **"Sensor Fusion for Telemedicine: Advancing Remote Healthcare
Through Multi-Modal Data Integration."**

> **Research software.** This is a reference implementation for academic
> reproducibility. It is **not** a certified medical device and must not be
> used for clinical decision-making on real patients. See
> [Regulatory status](#regulatory-status).

---

## What this is

MedROAD V3 turns OpenEMR from a passive record store into an event-driven
clinical intelligence platform. Observations arriving at the FHIR server are
pushed to Kafka, windowed, fused into a 86-element feature vector, scored by a
calibrated three-model ensemble, explained with SHAP, narrated by an LLM, and
written back into the patient chart as native FHIR resources.

The architectural claim the paper makes is that because the FHIR R4
`Observation` resource is device-agnostic, the *same* pipeline ingests bedside
monitors and consumer wearables without modification. Only the feature mapping
and the window width change.

```
In-hospital sensors ─┐
                     ├─→ FHIR R4 Observation ─→ Kafka ─→ Faust 5-min window
Wearable / RPM ──────┘       (data-level fusion)               │
                                                               ▼
                                     86-element vector (feature-level fusion)
                                                               │
                              ┌────────────────┬───────────────┴──┐
                          XGBoost            LSTM           Transformer
                              └────────────────┴──────────────────┘
                                          │  (decision-level fusion)
                        LR meta-learner + isotonic calibration, tau = 0.72
                                          │
                       SHAP ─── LLM narrative ─── FHIR write-back → OpenEMR
```

---

> **Full installation manual:** [INSTALL.md](INSTALL.md) covers prerequisites,
> MIMIC-IV acquisition, the live stack, reproduction of every published result,
> and a troubleshooting section drawn from real deployment failures.

## Quickstart

### 1. Install

```bash
git clone https://github.com/pboulanger/medroad-v3.git
cd medroad-v3
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

Install subsets if you only need part of the stack:

| Extra | Pulls in | Use when |
|---|---|---|
| *(base)* | numpy, scipy, scikit-learn, pandas, requests | calibration + RPM mapping only |
| `[ml]` | xgboost, shap, torch | training or running inference |
| `[stream]` | kafka-python, faust-streaming | running the stream processor |
| `[serve]` | fastapi, uvicorn | running the FHIR webhook |
| `[llm]` | anthropic | LLM narrative generation |
| `[all]` | everything above | full pipeline |
| `[dev]` | pytest, ruff | contributing |

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your OpenEMR OAuth2 credentials and Anthropic API key.
**Never commit `.env`** — it is gitignored.

### 3. Bring up the stack

```bash
docker compose up -d              # MariaDB, OpenEMR, Zookeeper, Kafka
docker compose logs -f openemr    # wait for "successfully installed" (~3-5 min)
```

Register an OAuth2 client at
`http://localhost:8300/interface/smart/register-app.php` and copy the
`client_id` and `client_secret` into `.env`.

### 4. Train

```bash
python -m medroad_v3.training.train --data /path/to/mimic_iv_windows.csv
```

See [Data availability](#data-availability) for how to obtain MIMIC-IV and the
expected CSV schema.

### 5. Run

```bash
docker compose up -d webhook stream          # webhook receiver + stream processor
docker compose --profile setup up setup      # register FHIR Subscriptions
curl http://localhost:8001/health            # verify
python -m medroad_v3.main infer-test         # synthetic end-to-end check
```

---

## Repository layout

```
medroad-v3/
├── medroad_v3/
│   ├── config.py              # env-driven settings, LOINC code tables
│   ├── main.py                # CLI: webhook | stream | setup | infer-test
│   ├── fhir/                  # OAuth2 client, Subscriptions, write-back
│   ├── streaming/             # Kafka producer, Faust windowed processor
│   ├── features/              # 86-element feature vector assembly
│   ├── models/                # XGBoost, LSTM, Transformer, ensemble, calibration
│   ├── rpm/                   # remote patient monitoring extension
│   ├── inference/             # orchestration + latency instrumentation
│   ├── narrative/             # Claude prompt contract and fallback
│   ├── training/              # OOF training pipeline
│   └── webhook/               # FastAPI REST-hook receiver
├── tests/
├── docs/
├── docker-compose.yml
└── pyproject.toml
```

---

## Code to paper section map

| Paper section | Module | Implements |
|---|---|---|
| 3.1 FHIR Event Source | `fhir/client.py`, `fhir/subscriptions.py` | OAuth2, R4B Subscription criteria, REST-hook registration |
| 3.2 Message Broker | `streaming/producer.py` | Topic routing, `patient_id` partition keying, DLQ |
| 3.3 Stream Processing | `streaming/faust_app.py` | 5-min tumbling windows, RocksDB state, completeness guard |
| 3.4 Inference and Decision | `inference/engine.py` | Orchestration, per-stage latency instrumentation |
| 3.5 FHIR Write-Back | `fhir/writeback.py` | Flag, CommunicationRequest, DocumentReference, ServiceRequest + upsert |
| 3.6 RPM Extension | `rpm/mapping.py` | Table 3 mapping, window sizing, coverage estimation |
| 4 Feature Engineering | `features/engineering.py` | 9 vitals x 4 statistics, 12 labs, 8 metadata, 21 missingness, 3 lab deltas, 2 weight deltas, SIG/BCR, temporal |
| 5 Ensemble and SHAP | `models/xgboost_model.py`, `models/deep_models.py`, `models/ensemble.py` | XGBoost + TreeExplainer, LSTM 2x128, Transformer d=64, LR meta-learner |
| 6 Model Calibration | `models/calibration.py` | ECE/MCE/Brier/Hosmer-Lemeshow, Platt, temperature, isotonic, shift detection, threshold selection |
| 7 Narrative Generation | `narrative/generator.py` | Prompt contract, SHAP grounding, deterministic fallback |
| 9 Evaluation | `training/train.py` | 5-fold OOF training, latency measurement |

---

## Two modules worth reading first

### Calibration (Section 6)

Probability calibration is treated as a first-class concern rather than a
post-processing detail. An uncalibrated score of 0.72 does not mean a 72%
chance of deterioration, which would make the alert threshold meaningless.

```python
from medroad_v3.models.calibration import (
    calibration_report, IsotonicCalibrator, select_threshold_f1,
)

print(calibration_report(y_true, y_prob))
# ECE=0.0913  MCE=0.2140  Brier=0.0685  AUROC=0.8710  HL chi2=41.2 (p=0.000)

cal = IsotonicCalibrator().fit(val_prob, val_y)
tau, f1 = select_threshold_f1(y_true, cal.transform(y_prob))
```

Platt and temperature scaling are monotone, so they leave AUROC untouched while
improving ECE — a property the test suite asserts explicitly.

### Remote patient monitoring (Section 3.6)

```python
from medroad_v3.rpm.mapping import DEVICE_PROFILES, describe_deployment

hf_kit = [p for p in DEVICE_PROFILES
          if p.name in ("Pulse oximeter", "BP cuff", "Smart scale")]
describe_deployment(hf_kit)
# {'window_seconds': 86400, 'expected_coverage': 0.6,
#  'unavailable_features': ['troponin_i', 'ventilator_pressure'], ...}
```

**The single most important configuration change** when moving from bedside to
remote deployment: `recommended_window_seconds` returns 24 h for any kit
containing a daily-cadence device such as a smart scale. The five-minute ICU
window assumes dense telemetry; applying it unchanged to an RPM cohort produces
mostly-empty windows.

---

---

## Experiments

The `medroad_v3.experiments` package contains the analyses that turn the
architecture description into an empirical paper. Each writes a JSON result
file and a ready-to-paste LaTeX table. All accept `--synthetic` to verify the
pipeline without credentialed data; synthetic figures are for plumbing checks
only and must never be reported.

```bash
python -m medroad_v3.experiments.ablation        --data mimic_windows.csv
python -m medroad_v3.experiments.rpm_degradation --data mimic_windows.csv
python -m medroad_v3.experiments.baselines       --data mimic_windows.csv
```

**`ablation`** fits the three base learners, builds every leave-one-out
ensemble, and reports AUROC, ECE, MCE and Brier with stratified bootstrap
intervals. Each reduced ensemble is tested against the full ensemble with
DeLong's test for AUROC and a paired bootstrap for ECE. This answers whether
the Transformer, the weakest single model, earns its place.

**`rpm_degradation`** degrades MIMIC-IV windows to what a given home
monitoring kit could observe, using the Table 3 mapping: unobservable channels
are dropped, observable ones are coarsened to device cadence, and the scorer is
recalibrated on the degraded representation. It also reports which Table 3
channels have no corresponding feature column and therefore cannot contribute
regardless of sensor quality.

**`baselines`** scores NEWS and MEWS from the same engineered vector,
de-normalised to clinical units, and compares alert burden per patient per
8-hour shift at matched sensitivity. Matching on sensitivity is what makes the
comparison fair, since any system can reduce alerts by detecting less.

---

**`latency`** measures per-stage inference latency on windows from the
evaluation cohort, reported as p50, p95 and p99 rather than means, since for a
real-time guarantee the tail determines whether the deadline is met. It needs
no infrastructure. **`latency_live`** measures the transport stages — FHIR
delivery, Kafka produce, FHIR write-back — against a running deployment, and
degrades to a clear message when the stack is down.

```bash
python -m medroad_v3.experiments.latency      --data mimic_windows.csv
python -m medroad_v3.experiments.latency_live --n 200   # needs docker compose up
```

---

**Regenerating the two figures removed from the manuscript.** The reliability
diagram and the scaling curve were cut because their data had never been
measured. Both are now produced from runs:

```bash
# writes results/ablation/reliability_figure.tex, ready to paste
python -m medroad_v3.experiments.ablation --data mimic_windows.csv

# writes results/latency/latency_scaling_figure.tex, ready to paste
python -m medroad_v3.experiments.latency_scaling --data mimic_windows.csv
```

The scaling benchmark measures the compute path only. It answers whether N
concurrent patients can be scored within one five-minute window period, which
is the deadline that matters, but it does not include broker or FHIR transport
under load. Label the resulting figure as compute-path scaling rather than
end-to-end.

---

## Development

```bash
pip install -e ".[dev]"
pytest                       # full suite
pytest -m "not integration"  # skip tests needing a live stack
ruff check medroad_v3 tests
```

CI runs lint and tests on Python 3.11 and 3.12 for every push and pull request.

---

### Retraining after a feature change

A feature-vector change invalidates every saved checkpoint and the training CSV
along with them, because the per-vital statistics and weight deltas are derived
from timestamped observations that an aggregated CSV has already discarded.
Rebuild from the MIMIC-IV extracts rather than reusing the old matrix:

```bash
# 0. extract from MIMIC-IV (PostgreSQL); see sql/extract_mimic.sql for the
#    cohort, the itemid filters and the deterioration outcome definition
psql mimiciv -f sql/extract_mimic.sql

# 1. verify the whole pipeline on 50 stays before committing to a full run
python scripts/retrain.py --mimic-dir ./mimic_extract --quick --lenient

# 2. the real run: validate, ETL, train, and run the experiments
python scripts/retrain.py --mimic-dir ./mimic_extract
```

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
conda activate ndib            # or: .\.venv\Scripts\Activate.ps1
cd "C:\Users\pierr\OneDrive\Desktop\Medical AI\Special Issue Senors 2026\medroad_v3"

# extract from MIMIC-IV
psql -d mimiciv -f sql\extract_mimic.sql

# verify the pipeline on 50 stays first
.\scripts\retrain.ps1 -MimicDir .\mimic_extract -Quick -Lenient

# the real run
.\scripts\retrain.ps1 -MimicDir .\mimic_extract
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Or skip the wrapper and call Python directly, which needs no policy change:

```powershell
python scripts\retrain.py --mimic-dir .\mimic_extract --quick --lenient
python scripts\retrain.py --mimic-dir .\mimic_extract
```

Use `python`, not `python3`: on Windows `python3` normally resolves to the
Microsoft Store stub. Quote any path containing spaces.

</details>

### Running against the full MIMIC-IV download

The extractor chunks `chartevents`, `labevents` and `prescriptions`, so it runs
against the complete tables without decompressing them. Only seven files are
needed, roughly 6 GB compressed; the rest of MIMIC-IV can be skipped.

```powershell
New-Item -ItemType Directory -Force C:\mimic\full\hosp, C:\mimic\full\icu | Out-Null
cd C:\mimic\full

$u = "YOUR_PHYSIONET_USERNAME"
$base = "https://physionet.org/files/mimiciv/3.1"
foreach ($f in @(
    "hosp/admissions.csv.gz", "hosp/patients.csv.gz",
    "hosp/labevents.csv.gz",  "hosp/prescriptions.csv.gz",
    "icu/icustays.csv.gz",    "icu/chartevents.csv.gz",
    "icu/d_items.csv.gz")) {
  curl.exe -u "$u" -C - --create-dirs -o $f "$base/$f"
}
```

`-C -` resumes a partial download rather than restarting, which matters for
`chartevents.csv.gz`. Verify integrity before extracting, because a truncated
gzip is the failure mode that costs the most time:

```powershell
python verify_gz.py C:\mimic\full
```

Then extract and train as usual, pointing at the full download:

```powershell
cd "<your code directory>"
python -m medroad_v3.training.extract_raw --raw-dir C:\mimic\full --out-dir .\mimic_extract --blank-hours 6
python scripts\retrain.py --mimic-dir .\mimic_extract
```

Drop `--quick`: that flag caps the ETL at 50 stays and exists only for
pipeline verification.

Expect the cohort to be roughly 10,000 CCU and CVICU stays rather than 94,
the extract step to take 30 to 60 minutes dominated by the `chartevents`
scan, and the window matrix to reach several million rows. Check the
reported event rate against the 24.5% seen on the subset; a cohort this much
larger should be more stable, not less.

**No PostgreSQL?** If you downloaded a patient-filtered MIMIC-IV subset (the
`hosp/`, `icu/`, `note/` layout of gzipped CSVs), skip the SQL entirely and
build the extracts with pandas:

```powershell
python -m medroad_v3.training.extract_raw --raw-dir .\mimic_subset_250 --out-dir .\mimic_extract
python scripts\retrain.py --mimic-dir .\mimic_extract --quick --lenient
```

Subsets frequently omit `inputevents` and `procedureevents`, which is where
vasopressor and ventilation events normally come from. The extractor falls back
to drug orders in `prescriptions` and to ventilator-setting itemids appearing in
`chartevents`. Both are later and noisier than the true event time, and the
manuscript must state which definition was used. Add `--all-units` if the
cardiac cohort is too small.

`scripts/retrain.py` gates between stages, because every failure that matters
here is silent. A wrong itemid mapping, a mislabelled cohort and a constant
sequence tensor all train cleanly and produce plausible metrics. The gates
check that mapped itemids actually appear in your extract, that the window
event rate is in a plausible range, that the feature count matches
`config.N_FEATURES`, and that the sequences carry temporal variation. Any gate
failing stops the run and writes `results/retrain_report.json`.

Individual stages remain available:

```bash
python -m medroad_v3.training.etl --mimic-dir ./mimic_extract --out mimic_windows.csv
python -m medroad_v3.training.train --data mimic_windows.csv --output models_saved
python -m medroad_v3.experiments.ablation --data mimic_windows.csv
```

The ETL expects filtered extracts (`chartevents.csv`, `labevents.csv`,
`stays.csv`, `outcomes.csv`) rather than the full tables. Filter on the itemids
in `ITEMID_TO_LOINC` to keep the extract to a few GB.

**Check the itemid mapping against your MIMIC-IV release before trusting a
run.** Itemids are not stable across versions, and a wrong mapping trains
cleanly while meaning nothing.

Training logs a `sequence temporal variation` figure. If it is zero the
sequence models are receiving constant input and the LSTM and Transformer are
wasted capacity; pass `patient_ids` and `window_starts` to `build_sequences`.

---

## Data availability

Models are trained on **MIMIC-IV**, publicly available via
[PhysioNet](https://physionet.org/content/mimiciv/) subject to credentialing
and a data use agreement. **No clinical data is included in this repository**,
and `.gitignore` blocks `data/`, `*.csv`, and `*.parquet` to prevent accidental
commits.

Expected training CSV schema — one row per 5-minute window:

```
patient_id, window_start, label, <86 feature columns in vector order>
```

Column order must match `medroad_v3/features/engineering.py`: vitals, labs,
metadata, vital missingness, lab missingness, lab deltas, weight deltas,
SIG/BCR, temporal.

---

## Regulatory status

The paper positions MedROAD V3 as a candidate **Health Canada Class II Software
as a Medical Device**. It has not been submitted for, nor granted, any
regulatory clearance. The remote monitoring extension (Section 3.6) is a
documented design proposal validated only against retrospective in-hospital
MIMIC-IV data; prospective validation on wearable sensor streams is future work.

---

## Citing

If you use this software, please cite both the paper and the software. GitHub
renders [`CITATION.cff`](CITATION.cff) as a "Cite this repository" button.

```bibtex
@article{boulanger2026medroad,
  author  = {Boulanger, Pierre},
  title   = {Multi-Modal {FHIR} Sensor Fusion for Real-Time Clinical Decision
             Support: From Hospital Bedside to Remote Patient Monitoring},
  journal = {Sensors},
  year    = {2026},
  note    = {Special Issue: Sensor Fusion for Telemedicine}
}
```

---

## License

[MIT](LICENSE) © 2026 Pierre Boulanger
