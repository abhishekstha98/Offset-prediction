# GraphCast Fog Baseline Handoff

Date: 2026-05-05

## 1. Purpose

This document is the execution brief for the local `GraphCast` repo.

The job of the `GraphCast` repo in this study is:

- run GraphCast inference over the study region and time window
- export station-aligned meteorological predictions
- produce files that can be evaluated by the local `Offset-prediction` repo

This is **not** a pollution workflow and **not** a direct fog-prediction workflow.

## 2. Study framing

The main study has moved away from pollution and is now strictly focused on:

- fog formation
- low-visibility forecasting
- station-scale sub-grid effects

Within that framing, `GraphCast` is being used as:

`a coarse meteorological baseline`

More specifically:

- GraphCast provides coarse-grid atmospheric predictions
- those predictions are interpolated to station locations
- the local repo compares them against station observations and later against fog labels

Do **not** frame GraphCast as a direct fog model.

Do **not** add pollution-specific logic, emissions logic, or PM-specific outputs in this phase.

## 3. Immediate milestone

The first milestone is:

`export station-aligned daily Tmax/Tmin predictions for 2024-01-01 to 2025-12-31`

This milestone exists to validate:

- local GraphCast inference setup
- variable extraction
- time handling
- spatial interpolation
- compatibility with the downstream evaluation scripts

The second milestone is:

`export station-aligned hourly meteorological precursor fields for later fog evaluation`

## 4. Required deliverables

Produce these three files:

### A. Daily validation export

`station_daily_tmax_tmin_graphcast.csv`

### B. Hourly precursor export

`station_hourly_precursors_graphcast.parquet`

### C. Run metadata

`run_metadata_graphcast.json`

## 5. Output schema

### Daily CSV schema

Required columns:

- `station`
- `date`
- `pred_tmax`
- `pred_tmin`

Column meanings:

- `station`: must exactly match the station IDs from the station manifest
- `date`: daily valid date
- `pred_tmax`: predicted daily maximum 2m temperature at station location
- `pred_tmin`: predicted daily minimum 2m temperature at station location

### Hourly precursor schema

Required columns:

- `station`
- `time`
- `pred_t2m`
- `pred_u10`
- `pred_v10`
- `pred_pressure`
- best available humidity-compatible field
- `source_model`

Required conventions:

- `time` must be UTC
- one row per `station x hour`
- `source_model` should be the literal string `graphcast`

### Metadata JSON schema

At minimum include:

- `source_model`
- `graphcast_repo_commit`
- `model_variant`
- `checkpoint_or_weights`
- `inference_entrypoint`
- `input_data_source`
- `forecast_period_start`
- `forecast_period_end`
- `variables_exported`
- `units`
- `interpolation_method`
- `daily_aggregation_rule`
- `humidity_field_used`
- `notes`

## 6. Execution workflow

Follow this order.

### Step 1. Identify the exact local inference entrypoint

Record the actual local GraphCast runner:

- notebook
- script
- wrapper

Do not leave this implicit. Write it into `run_metadata_graphcast.json`.

### Step 2. Record the model/checkpoint variant

Before producing outputs, explicitly record:

- GraphCast variant used
- checkpoint/weights source
- native spatial resolution
- native temporal resolution

### Step 3. Run a short Netherlands sample first

Before the full `2024-01-01` to `2025-12-31` export, run a short sample window.

Recommended first smoke window:

- a few consecutive days over the Netherlands station domain

Use this to verify:

- variable names
- unit conventions
- coordinate names
- time spacing
- whether the output contains the needed fields

### Step 4. Inspect outputs before station export

Confirm:

- which variable represents 2m temperature
- whether temperature is Celsius or Kelvin
- whether time stamps are forecast valid times
- whether lat/lon are standard geographic coordinates
- whether pressure is available as mean sea level pressure or surface pressure
- which humidity-compatible field is available

### Step 5. Export station-level results

Use the station manifest from the local `Offset-prediction` repo.

The export path should either:

- interpolate gridded GraphCast output to station coordinates

or:

- directly write station-level predictions if the local GraphCast workflow already supports that

For interpolation, use one consistent method and record it in metadata.

Default:

- linear / bilinear interpolation first

### Step 6. Aggregate to daily Tmax/Tmin

From the sub-daily station-aligned temperature series:

- compute daily `pred_tmax`
- compute daily `pred_tmin`

Use one explicit day-boundary convention and record it.

Default convention:

- aggregate by the same local-day definition used downstream in `Offset-prediction`

### Step 7. Write the shared deliverables

Write:

- `station_daily_tmax_tmin_graphcast.csv`
- `station_hourly_precursors_graphcast.parquet`
- `run_metadata_graphcast.json`

The hourly precursor file can be filled after the daily validation pass succeeds, but the final repo handoff should support both products.

## 7. Variable and unit requirements

Minimum required meteorological outputs:

- 2m temperature
- 10m u wind
- 10m v wind
- pressure field

Preferred additional output:

- direct humidity-related field

Units must be documented clearly.

Examples:

- temperature: Kelvin or Celsius
- wind: m/s
- pressure: Pa or hPa

The exported station-level files should use stable, documented units. If upstream units are kept, they must be stated explicitly in metadata.

## 8. Humidity handling

Humidity should follow the best available rule, consistent with the fog-data contract in the local repo.

Preference order:

1. direct `2m dewpoint`
2. direct near-surface relative humidity
3. lowest-level humidity-compatible field that can act as a fog precursor proxy

Requirements:

- export the exact field name used
- document the unit
- explain whether it is direct dewpoint, RH, or a proxy
- do not silently substitute one type for another

If no usable humidity-compatible field is available in the chosen GraphCast path, record that explicitly in metadata rather than inventing one.

## 9. Validation checklist

Before handing results back, verify:

- the station IDs exactly match the station manifest
- the date range covers `2024-01-01` to `2025-12-31`
- the daily CSV has all required columns
- the hourly precursor file has all required columns that are actually available
- `pred_tmax >= pred_tmin` for all valid daily rows
- no obvious unit errors exist
- interpolation method is documented
- metadata JSON is complete

Also check that the exported daily CSV can be consumed without schema edits by the downstream evaluator.

## 10. Handoff back to `Offset-prediction`

The local `Offset-prediction` repo is the evaluation hub.

Downstream evaluation happens with:

- `scripts/export_station_manifest.py`
- `scripts/evaluate_gridded_baseline.py`
- `scripts/validate_mpt_benchmarks.py`

The GraphCast repo does **not** need to reimplement those benchmark comparisons.

Its responsibility is to return clean exports that the local repo can consume directly.

## Final note

The target narrative is:

`GraphCast as a coarse meteorological baseline for a fog-focused station-level study`

Not:

- pollution baseline
- direct fog model
- separate standalone benchmark paper
