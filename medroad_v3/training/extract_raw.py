"""
Build the four ETL input CSVs directly from raw MIMIC-IV files.

``sql/extract_mimic.sql`` assumes a PostgreSQL instance with the full
``mimiciv_hosp`` and ``mimiciv_icu`` schemas. This module is the alternative
for a downloaded subset: it reads the gzipped CSVs in place and writes
``stays.csv``, ``chartevents.csv``, ``labevents.csv`` and ``outcomes.csv``.

It also copes with a subset that lacks ``inputevents`` and ``procedureevents``,
which many patient-filtered downloads do. Those tables are where vasopressor
initiation and mechanical ventilation normally come from, so without them the
deterioration outcome has to be derived from substitutes:

    vasopressor start   prescriptions / emar drug names
    ventilation start   ventilator-setting itemids appearing in chartevents
    circulatory support unavailable, omitted
    cardiac arrest      unavailable, omitted
    death               admissions.deathtime

The substitutions are weaker than the originals. Drug orders are not
administrations, and ventilator settings appear only once charting begins, so
both are later and noisier than the true event time. The manuscript must state
which definition was actually used.

Usage:
    python -m medroad_v3.training.extract_raw \\
        --raw-dir ./mimic_subset_250 --out-dir ./mimic_extract
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CARDIAC_UNITS = (
    "Coronary Care Unit (CCU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
)

# chartevents itemids kept, matching ITEMID_TO_LOINC in etl.py
CHART_ITEMIDS = {
    220045, 220179, 220050, 220180, 220051, 220277,
    220210, 224690, 223761, 223762, 220739, 223900, 223901,
    223791, 226512, 224639,
}

LAB_ITEMIDS = {
    51003, 50963, 50912, 50971, 50983, 50882,
    50818, 50820, 50813, 51221, 51301, 50960,
}

# Presence of any of these implies invasive ventilation is running.
VENT_ITEMIDS = {
    223849,   # ventilator mode
    220339,   # PEEP set
    224685,   # tidal volume observed
    224684,   # tidal volume set
    224686,   # tidal volume spontaneous
    223848,   # ventilator type
}

VASOPRESSOR_DRUGS = {
    "norepinephrine",
    "norepinephrine (in ns)",
    "levophed",
    "epinephrine",
    "vasopressin",
    "pitressin",
    "dopamine",
    "dobutamine",
}
# Deliberately excluded: phenylephrine, ordered for nearly every patient in a
# cardiac cohort for transient hypotension; milrinone, usually started
# electively in heart failure; and the lidocaine/epinephrine, EpiPen,
# racepinephrine and nasal-spray products that substring matching on
# "epinephrine" or "phenylephrine" otherwise sweeps up. Matching is exact
# because no amount of pattern tuning separates these reliably.


def _find(raw: Path, *candidates: str) -> Path | None:
    """Locate a table under the hosp/icu layout, gzipped or not."""
    for c in candidates:
        for sub in ("", "hosp", "icu"):
            for ext in (".csv.gz", ".csv"):
                p = raw / sub / f"{c}{ext}" if sub else raw / f"{c}{ext}"
                if p.exists():
                    return p
    return None


class PlaceholderFileError(RuntimeError):
    """A file exists in the directory listing but its bytes are not local."""


def _check_readable(path: Path) -> None:
    """
    Fail early and legibly on cloud-placeholder files.

    OneDrive, Dropbox and iCloud all support files that appear in a listing at
    full size while their contents live only in the cloud. Reading one raises
    OSError errno 22 from deep inside the decompressor, which is an opaque way
    to learn that the download never happened.
    """
    try:
        with open(path, "rb") as fh:
            fh.read(4)
    except OSError as exc:
        if exc.errno == 22:
            raise PlaceholderFileError(
                f"{path.name} cannot be read: its contents are not on this "
                f"machine.\n\n"
                f"This is a cloud-storage placeholder. The usual cause is "
                f"OneDrive Files On-Demand.\n\n"
                f"Move the dataset off cloud-synced storage, which is the only "
                f"durable fix:\n"
                f"    Move-Item <dataset> C:\\mimic\\\n\n"
                f"Forcing a download also works but is temporary, since the "
                f"files get evicted again:\n"
                f"    attrib -U +P <dataset>\\*.* /s\n\n"
                f"Note also that syncing credentialed PhysioNet data to a "
                f"third-party cloud service may breach the data use agreement."
            ) from exc
        raise


def _read(path: Path, **kw) -> pd.DataFrame:
    logger.info("reading %s", path.name)
    _check_readable(path)
    df = pd.read_csv(path, low_memory=False, **kw)
    logger.info("  %d rows", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════

def build_cohort(raw: Path, units: tuple[str, ...] | None,
                 min_los_days: float) -> pd.DataFrame:
    p = _find(raw, "icustays")
    if p is None:
        raise FileNotFoundError("icustays not found under the raw directory")
    icu = _read(p, parse_dates=["intime", "outtime"])

    logger.info("care units present: %s", sorted(icu.first_careunit.unique()))
    cohort = icu
    if units:
        cohort = cohort[cohort.first_careunit.isin(units)]
        logger.info("after cardiac-unit filter: %d stays", len(cohort))
    cohort = cohort[cohort.los >= min_los_days]
    logger.info("after LOS >= %.2f d filter: %d stays", min_los_days, len(cohort))
    return cohort[["stay_id", "hadm_id", "subject_id", "intime", "outtime"]]


def build_chartevents(raw: Path, stay_ids: set[int]) -> pd.DataFrame:
    p = _find(raw, "chartevents")
    if p is None:
        raise FileNotFoundError("chartevents not found")
    _check_readable(p)
    keep = []
    for chunk in pd.read_csv(
        p, chunksize=1_000_000, low_memory=False,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
    ):
        m = chunk[chunk.stay_id.isin(stay_ids) & chunk.itemid.isin(CHART_ITEMIDS)]
        if not m.empty:
            keep.append(m)
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(
        columns=["stay_id", "charttime", "itemid", "valuenum"])
    df = df.dropna(subset=["valuenum"])
    logger.info("chartevents retained: %d rows, %d itemids",
                len(df), df.itemid.nunique())
    return df


def build_labevents(raw: Path, hadm_ids: set[int]) -> pd.DataFrame:
    p = _find(raw, "labevents")
    if p is None:
        raise FileNotFoundError("labevents not found")
    _check_readable(p)
    keep = []
    for chunk in pd.read_csv(
        p, chunksize=1_000_000, low_memory=False,
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
    ):
        m = chunk[chunk.hadm_id.isin(hadm_ids) & chunk.itemid.isin(LAB_ITEMIDS)]
        if not m.empty:
            keep.append(m)
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(
        columns=["hadm_id", "charttime", "itemid", "valuenum"])
    df = df.dropna(subset=["valuenum"])
    logger.info("labevents retained: %d rows, %d itemids",
                len(df), df.itemid.nunique())
    return df


def build_outcomes(raw: Path, cohort: pd.DataFrame,
                   blank_hours: float = 1.0) -> pd.DataFrame:
    """
    Composite deterioration event, from whichever sources the subset has.

    ``blank_hours`` suppresses events in the period immediately after ICU
    admission. One hour suffices for a medical unit, but surgical patients
    arrive from theatre already ventilated and on vasopressor support, so a
    short blanking period labels routine post-operative recovery as
    deterioration and inflates the event rate several-fold. Six to twelve hours
    is appropriate when the cohort includes a surgical ICU.
    """
    events: list[pd.DataFrame] = []
    stay_ids = set(cohort.stay_id)
    hadm_ids = set(cohort.hadm_id)
    by_hadm = cohort.set_index("hadm_id")[["stay_id", "intime", "outtime"]]

    # ── death ────────────────────────────────────────────────────────────
    p = _find(raw, "admissions")
    if p is not None:
        adm = _read(p, parse_dates=["deathtime"])
        adm = adm[adm.hadm_id.isin(hadm_ids) & adm.deathtime.notna()]
        if not adm.empty:
            j = adm.join(by_hadm, on="hadm_id", how="inner")
            j = j[(j.deathtime >= j.intime) & (j.deathtime <= j.outtime)]
            events.append(pd.DataFrame({
                "stay_id": j.stay_id, "event_time": j.deathtime,
                "event_type": "death"}))
            logger.info("death events: %d", len(j))

    # ── ventilation, inferred from charted ventilator settings ───────────
    p = _find(raw, "chartevents")
    if p is not None:
        vent = []
        for chunk in pd.read_csv(
            p, chunksize=1_000_000, low_memory=False,
            usecols=["stay_id", "charttime", "itemid"],
        ):
            m = chunk[chunk.stay_id.isin(stay_ids) & chunk.itemid.isin(VENT_ITEMIDS)]
            if not m.empty:
                vent.append(m)
        if vent:
            v = pd.concat(vent, ignore_index=True)
            v["charttime"] = pd.to_datetime(v.charttime)
            first = v.groupby("stay_id").charttime.min().reset_index()
            first = first.merge(cohort[["stay_id", "intime"]], on="stay_id")
            first = first[first.charttime > first.intime + pd.Timedelta(hours=blank_hours)]
            events.append(pd.DataFrame({
                "stay_id": first.stay_id, "event_time": first.charttime,
                "event_type": "ventilation"}))
            logger.info("ventilation events: %d", len(first))

    # ── vasopressor start, from drug orders ──────────────────────────────
    p = _find(raw, "prescriptions")
    if p is not None:
        try:
            # Chunked: the full MIMIC-IV prescriptions table runs to roughly
            # 17 million rows and will not fit comfortably in memory.
            _check_readable(p)
            keep = []
            for chunk in pd.read_csv(
                p, chunksize=1_000_000, low_memory=False,
                usecols=["hadm_id", "starttime", "drug"],
                parse_dates=["starttime"],
            ):
                m = chunk[chunk.hadm_id.isin(hadm_ids)]
                if m.empty:
                    continue
                m = m[m.drug.str.lower().str.strip().isin(VASOPRESSOR_DRUGS)]
                if not m.empty:
                    keep.append(m)
            rx = (pd.concat(keep, ignore_index=True) if keep
                  else pd.DataFrame(columns=["hadm_id", "starttime", "drug"]))
            logger.info("vasopressor orders matched: %d", len(rx))
            if not rx.empty:
                j = rx.join(by_hadm, on="hadm_id", how="inner")
                j = j[(j.starttime > j.intime + pd.Timedelta(hours=blank_hours))
                      & (j.starttime <= j.outtime)]
                g = j.groupby("stay_id").starttime.min().reset_index()
                events.append(pd.DataFrame({
                    "stay_id": g.stay_id, "event_time": g.starttime,
                    "event_type": "vasopressor"}))
                logger.info("vasopressor events: %d", len(g))
        except (KeyError, ValueError) as e:
            logger.warning("prescriptions unusable for vasopressors: %s", e)

    if not events:
        raise RuntimeError(
            "no deterioration events could be derived. Without inputevents, "
            "procedureevents, prescriptions or deaths there is no label."
        )

    allev = pd.concat(events, ignore_index=True)
    logger.info("event type breakdown:\n%s",
                allev.event_type.value_counts().to_string())
    out = allev.groupby("stay_id").event_time.min().reset_index()
    logger.info("stays with an event: %d of %d (%.1f%%)",
                len(out), len(cohort), 100 * len(out) / max(len(cohort), 1))
    return out


# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract ETL inputs from raw MIMIC-IV files")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default="mimic_extract")
    ap.add_argument("--all-units", action="store_true",
                    help="keep every ICU, not only CCU and CVICU. Use when the "
                         "cardiac cohort is too small to train on.")
    ap.add_argument("--min-los-days", type=float, default=0.25)
    ap.add_argument("--units", nargs="+", default=None,
                    help="explicit care-unit names; overrides the cardiac default. "
                         "Use 'Coronary Care Unit (CCU)' alone to exclude the "
                         "surgical unit.")
    ap.add_argument("--ccu-only", action="store_true",
                    help="keep only the medical coronary unit. Excludes CVICU, "
                         "whose post-operative admissions otherwise dominate the "
                         "event label.")
    ap.add_argument("--blank-hours", type=float, default=1.0,
                    help="suppress events within this many hours of ICU "
                         "admission (use 6-12 for surgical cohorts)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.all_units:
        units = None
    elif args.ccu_only:
        units = ("Coronary Care Unit (CCU)",)
    elif args.units:
        units = tuple(args.units)
    else:
        units = CARDIAC_UNITS
    cohort = build_cohort(raw, units, args.min_los_days)
    if cohort.empty:
        raise SystemExit(
            "cohort is empty. Try --all-units, or lower --min-los-days."
        )

    chart = build_chartevents(raw, set(cohort.stay_id))
    labs = build_labevents(raw, set(cohort.hadm_id))
    outcomes = build_outcomes(raw, cohort, blank_hours=args.blank_hours)

    cohort.to_csv(out / "stays.csv", index=False)
    chart.to_csv(out / "chartevents.csv", index=False)
    labs.to_csv(out / "labevents.csv", index=False)
    outcomes.to_csv(out / "outcomes.csv", index=False)

    print()
    print(f"  stays        {len(cohort):>8}")
    print(f"  chartevents  {len(chart):>8}")
    print(f"  labevents    {len(labs):>8}")
    print(f"  outcomes     {len(outcomes):>8}"
          f"  ({100*len(outcomes)/max(len(cohort),1):.1f}% of stays)")
    print(f"\nwritten to {out.resolve()}")

    rate = len(outcomes) / max(len(cohort), 1)
    if rate > 0.35:
        print(f"\nWARNING: {100*rate:.0f}% of stays carry an event. A cardiac ICU "
              f"deterioration rate above ~35% means the label is capturing "
              f"routine care rather than deterioration.\n"
              f"         Try --ccu-only (excludes post-operative CVICU "
              f"admissions) and --blank-hours 6.")

    if len(cohort) < 50:
        print("\nWARNING: fewer than 50 stays. This is enough to verify the "
              "pipeline but not to train a model worth reporting.")


if __name__ == "__main__":
    main()
