# Contributing

Thanks for your interest in MedROAD V3.

## Reporting issues

Please include your Python version, OS, the command you ran, and the full
traceback. For pipeline issues, the output of `python -m medroad_v3.main
infer-test` is usually the most informative starting point.

## Development setup

```bash
git clone https://github.com/pboulanger/medroad-v3.git
cd medroad-v3
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Before opening a pull request

- `ruff check medroad_v3 tests` passes
- `pytest` passes
- New behaviour is covered by a test
- Public functions have type hints and a docstring

## Clinical safety

This repository implements a clinical decision support system. Two rules are
non-negotiable:

1. **Never commit patient data.** MIMIC-IV is credentialed and must not be
   redistributed. `.gitignore` blocks `data/`, `*.csv`, and `*.parquet`, but
   check your diff regardless.
2. **Changes affecting risk scoring, calibration, or the alert threshold must
   state their clinical rationale** in the pull request description. A change
   that improves AUROC while degrading calibration is not an improvement — the
   threshold is what determines whether a clinician is interrupted.

## Scope

Contributions that extend the RPM feature mapping to additional device classes,
add calibration diagnostics, or improve FHIR conformance are especially
welcome. Please open an issue to discuss substantial architectural changes
before starting work.
