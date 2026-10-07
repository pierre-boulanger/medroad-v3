-- ============================================================================
-- MedROAD V3 — MIMIC-IV extraction
--
-- Produces the four CSVs that medroad_v3.training.etl expects. Written for
-- MIMIC-IV v2.2 on PostgreSQL with the `mimiciv_hosp` and `mimiciv_icu`
-- schemas. Adjust schema names for your installation.
--
-- Run each block with \copy, for example:
--   \copy (<query>) TO 'mimic_extract/stays.csv' WITH CSV HEADER
--
-- ---------------------------------------------------------------------------
-- A NOTE ON THE OUTCOME DEFINITION
--
-- The manuscript defines deterioration as unplanned ICU transfer, rapid
-- response activation, or in-hospital cardiac arrest. Two of those are not
-- recoverable from MIMIC-IV: rapid response activations are not recorded at
-- all, and cardiac arrest has no dedicated table. The third does not apply to
-- a cohort that is already in an ICU.
--
-- This script therefore uses the composite that the deterioration-prediction
-- literature actually uses for ICU cohorts, built from events MIMIC-IV does
-- record:
--
--   1. first initiation of a vasopressor or inotrope infusion
--   2. first initiation of invasive mechanical ventilation
--   3. initiation of mechanical circulatory support (IABP, ECMO, Impella)
--   4. bolus epinephrine, as the available proxy for cardiac arrest
--   5. death in the ICU
--
-- Each is an escalation of care representing acute haemodynamic or respiratory
-- decompensation. The manuscript must be amended to state this definition
-- rather than the current one, and to acknowledge item 4 as a proxy. Reporting
-- a definition the data cannot support is the kind of thing a reviewer with
-- MIMIC-IV experience will catch immediately.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 1. COHORT — adult cardiac ICU stays
-- ----------------------------------------------------------------------------
-- CVICU is the cardiac surgical unit, CCU the medical coronary unit. Stays
-- under 6 hours are excluded: they yield too few windows to form a sequence
-- and are dominated by immediate post-operative recovery.

DROP TABLE IF EXISTS medroad_cohort;
CREATE TEMP TABLE medroad_cohort AS
SELECT
    icu.stay_id,
    icu.hadm_id,
    icu.subject_id,
    icu.intime,
    icu.outtime,
    icu.first_careunit
FROM mimiciv_icu.icustays icu
JOIN mimiciv_hosp.patients pat ON pat.subject_id = icu.subject_id
JOIN mimiciv_hosp.admissions adm ON adm.hadm_id = icu.hadm_id
WHERE icu.first_careunit IN (
        'Coronary Care Unit (CCU)',
        'Cardiac Vascular Intensive Care Unit (CVICU)'
      )
  AND pat.anchor_age >= 18
  AND icu.los >= 0.25;                      -- at least 6 hours

-- stays.csv
SELECT stay_id, hadm_id, subject_id, intime, outtime
FROM medroad_cohort
ORDER BY stay_id;


-- ----------------------------------------------------------------------------
-- 2. CHARTEVENTS — vital signs, filtered to mapped itemids
-- ----------------------------------------------------------------------------
-- These itemids must match ITEMID_TO_LOINC in medroad_v3/training/etl.py.
-- Verify against mimiciv_icu.d_items for your release before trusting a run;
-- itemids are not stable across MIMIC versions.

-- chartevents.csv
SELECT
    ce.stay_id,
    ce.charttime,
    ce.itemid,
    ce.valuenum
FROM mimiciv_icu.chartevents ce
JOIN medroad_cohort c ON c.stay_id = ce.stay_id
WHERE ce.valuenum IS NOT NULL
  AND ce.itemid IN (
        220045,                     -- heart rate
        220179, 220050,             -- systolic BP (NIBP, arterial)
        220180, 220051,             -- diastolic BP
        220277,                     -- SpO2
        220210, 224690,             -- respiratory rate
        223761, 223762,             -- temperature (F, C)
        220739, 223900, 223901,     -- GCS components
        223791,                     -- pain level
        226512, 224639              -- admission weight, daily weight
      )
  AND ce.charttime BETWEEN c.intime AND c.outtime
ORDER BY ce.stay_id, ce.charttime;


-- ----------------------------------------------------------------------------
-- 3. LABEVENTS — the twelve laboratory channels
-- ----------------------------------------------------------------------------
-- Note that MIMIC-IV records troponin T (51003) whereas the manuscript
-- specifies troponin I. They are different assays with different reference
-- ranges and are not interchangeable. Either change the manuscript to say
-- troponin T or document the substitution.

-- labevents.csv
SELECT
    le.hadm_id,
    le.charttime,
    le.itemid,
    le.valuenum
FROM mimiciv_hosp.labevents le
JOIN (SELECT DISTINCT hadm_id FROM medroad_cohort) c ON c.hadm_id = le.hadm_id
WHERE le.valuenum IS NOT NULL
  AND le.itemid IN (
        51003,   -- troponin T
        50963,   -- NTproBNP
        50912,   -- creatinine
        50971,   -- potassium
        50983,   -- sodium
        50882,   -- bicarbonate
        50818,   -- pCO2
        50820,   -- pH
        50813,   -- lactate
        51221,   -- hematocrit
        51301,   -- WBC
        50960    -- magnesium
      )
ORDER BY le.hadm_id, le.charttime;


-- ----------------------------------------------------------------------------
-- 4. OUTCOMES — composite deterioration events
-- ----------------------------------------------------------------------------

-- outcomes.csv
WITH vasopressors AS (
    -- First vasopressor or inotrope infusion in the stay. Patients already on
    -- one at admission are excluded below, since for them the infusion is not
    -- a deterioration event.
    SELECT ie.stay_id, MIN(ie.starttime) AS event_time
    FROM mimiciv_icu.inputevents ie
    JOIN medroad_cohort c ON c.stay_id = ie.stay_id
    WHERE ie.itemid IN (
            221906,   -- norepinephrine
            221289,   -- epinephrine (infusion)
            221662,   -- dopamine
            221653,   -- dobutamine
            222315,   -- vasopressin
            221749    -- phenylephrine
          )
      AND ie.starttime > c.intime + INTERVAL '1 hour'
    GROUP BY ie.stay_id
),
ventilation AS (
    -- First invasive mechanical ventilation
    SELECT pe.stay_id, MIN(pe.starttime) AS event_time
    FROM mimiciv_icu.procedureevents pe
    JOIN medroad_cohort c ON c.stay_id = pe.stay_id
    WHERE pe.itemid IN (225792, 225794)     -- invasive / non-invasive vent
      AND pe.starttime > c.intime + INTERVAL '1 hour'
    GROUP BY pe.stay_id
),
mcs AS (
    -- Mechanical circulatory support
    SELECT pe.stay_id, MIN(pe.starttime) AS event_time
    FROM mimiciv_icu.procedureevents pe
    JOIN medroad_cohort c ON c.stay_id = pe.stay_id
    WHERE pe.itemid IN (225752, 228178, 229268)   -- IABP, ECMO, Impella
    GROUP BY pe.stay_id
),
arrest_proxy AS (
    -- Bolus epinephrine, the available proxy for cardiac arrest. MIMIC-IV
    -- records no arrest event directly; this over-counts, because bolus
    -- epinephrine is also given for profound hypotension without arrest.
    SELECT ie.stay_id, MIN(ie.starttime) AS event_time
    FROM mimiciv_icu.inputevents ie
    JOIN medroad_cohort c ON c.stay_id = ie.stay_id
    WHERE ie.itemid = 221289
      AND ie.ordercategoryname ILIKE '%bolus%'
    GROUP BY ie.stay_id
),
icu_death AS (
    SELECT c.stay_id, adm.deathtime AS event_time
    FROM medroad_cohort c
    JOIN mimiciv_hosp.admissions adm ON adm.hadm_id = c.hadm_id
    WHERE adm.deathtime IS NOT NULL
      AND adm.deathtime BETWEEN c.intime AND c.outtime
),
all_events AS (
    SELECT stay_id, event_time, 'vasopressor'  AS event_type FROM vasopressors
    UNION ALL
    SELECT stay_id, event_time, 'ventilation'  FROM ventilation
    UNION ALL
    SELECT stay_id, event_time, 'mcs'          FROM mcs
    UNION ALL
    SELECT stay_id, event_time, 'arrest_proxy' FROM arrest_proxy
    UNION ALL
    SELECT stay_id, event_time, 'icu_death'    FROM icu_death
)
SELECT stay_id, MIN(event_time) AS event_time
FROM all_events
WHERE event_time IS NOT NULL
GROUP BY stay_id
ORDER BY stay_id;


-- ----------------------------------------------------------------------------
-- 5. SANITY CHECKS — run these before committing to a full ETL
-- ----------------------------------------------------------------------------
-- Expect an event rate in the region of 15 to 30 percent of stays for a
-- cardiac ICU cohort under this composite. A rate near zero means the itemids
-- are wrong for your release; a rate near one means the one-hour exclusion is
-- not filtering admission-time infusions.

-- SELECT COUNT(*) AS n_stays FROM medroad_cohort;
-- SELECT first_careunit, COUNT(*) FROM medroad_cohort GROUP BY 1;
-- SELECT COUNT(DISTINCT stay_id) AS n_with_event FROM all_events;
--
-- Per-channel coverage: any vital below ~80 percent of stays, or any lab
-- below ~30 percent, should be investigated before training rather than
-- silently absorbed by imputation.
--
-- SELECT ce.itemid, COUNT(DISTINCT ce.stay_id) AS n_stays
-- FROM mimiciv_icu.chartevents ce
-- JOIN medroad_cohort c ON c.stay_id = ce.stay_id
-- GROUP BY ce.itemid ORDER BY n_stays DESC;
