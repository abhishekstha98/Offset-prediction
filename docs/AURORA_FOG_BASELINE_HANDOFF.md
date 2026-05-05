# Aurora Fog Baseline Handoff

Date: 2026-05-05

## 1. Purpose

This document is the execution brief for the local `Aurora` repo.

The job of the `Aurora` repo in this study is:

- run Aurora inference over the study region and time window
- export station-aligned meteorological predictions
- produce files that can be evaluated by the local `Offset-prediction` repo

This is **not** a pollution workflow and **not** a direct fog-prediction workflow.

## 2. Study framing

The main study is now strictly about:

- fog formation
- low visibility
- station-scale meteorological correction and forecasting

Within that framing, `Aurora` is being used as:

`a coarse meteorological precursor baseline`

That means:

- Aurora provides coarse atmospheric predictions
- those predictions are aligned to station locations
- the local repo evaluates them first against station daily temperature targets and later against fog labels

Do **not** frame Aurora as a direct fog predictor.

Do **not** add pollution-specific outputs or PM-specific evaluation logic in this phase.

## 3. Immediate milestone

The first milestone is:

`export station-aligned daily Tmax/Tmin predictions for 2024-01-01 to 2025-12-31`

This is the validation milestone for the external-baseline pipeline.

The second milestone is:

`export station-aligned hourly meteorological precursor fields for later fog evaluation`

## 4. Required deliverables

Produce these three files:

### A. Daily validation export

`station_daily_tmax_tmin_aurora.csv`

### B. Hourly precursor export

`station_hourly_precursors_aurora.parquet`

### C. Run metadata

`run_metadata_aurora.json`

## 5. Output schema

### Daily CSV schema

Required columns:

- `station`
- `date`
- `pred_tmax`
- `pred_tmin`

Definitions:

- `station`: exact station ID from the shared manifest
- `date`: daily valid date
- `pred_tmax`: predicted daily station Tmax
- `pred_tmin`: predicted daily station Tmin

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
- `source_model` should be the literal string `aurora`

### Metadata JSON schema

At minimum include:

- `source_model`
- `aurora_repo_commit`
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

### Step 1. Use Aurora’s official ERA5 example as the starting reference

Start from the documented Aurora ERA5 flow and identify the exact local script or notebook used in the clone.

Record that entrypoint in metadata.

### Step 2. Record the actual Aurora configuration

Before running the full export, record:

- Aurora model variant
- checkpoint/weights source
- required input bundle
- native time spacing
- native spatial resolution

### Step 3. Run a short Netherlands sample first

Before the full `2024-01-01` to `2025-12-31` run, execute a short sample over the Netherlands station region.

Use the sample to confirm:

- variable names
- shape conventions
- units
- coordinate names
- time handling

### Step 4. Inspect the exported fields

Confirm which internal Aurora fields correspond to:

- 2m temperature
- 10m u wind
- 10m v wind
- pressure
- humidity-compatible signal

Do this before building the final station export.

### Step 5. Convert Aurora outputs to the shared station-level contract

The local `Offset-prediction` repo expects a fixed contract.

So even if Aurora uses different internal variable names, convert the final export to:

- `pred_t2m`
- `pred_u10`
- `pred_v10`
- `pred_pressure`
- humidity-compatible field(s)

For spatial mapping:

- interpolate gridded Aurora output to station coordinates

or:

- directly write station-level outputs if your local Aurora workflow already supports it

Use one explicit interpolation method and record it.

Default:

- linear / bilinear interpolation first

### Step 6. Aggregate to daily Tmax/Tmin

From the station-aligned temperature time series:

- compute daily `pred_tmax`
- compute daily `pred_tmin`

Use one documented day-boundary rule and write it into metadata.

### Step 7. Write the final deliverables

Write:

- `station_daily_tmax_tmin_aurora.csv`
- `station_hourly_precursors_aurora.parquet`
- `run_metadata_aurora.json`

Daily validation is the first required milestone. Hourly precursor export is the required extension after that milestone works.

## 7. Variable and unit requirements

Minimum required outputs:

- 2m temperature
- 10m u wind
- 10m v wind
- pressure field

Preferred additional output:

- direct humidity-related field

All exported units must be documented clearly.

Examples:

- temperature: Kelvin or Celsius
- wind: m/s
- pressure: Pa or hPa

If the internal Aurora outputs use different names or conventions, convert or document them explicitly in metadata.

## 8. Humidity handling

Humidity should follow the best available rule, consistent with the local fog-data contract.

Preference order:

1. direct `2m dewpoint`
2. direct near-surface relative humidity
3. lowest-level humidity-compatible field that can be used as a documented proxy

Requirements:

- export the exact humidity-related field used
- document the unit
- document whether it is direct dewpoint, RH, or a proxy
- describe any limitations

If Aurora’s chosen run path does not expose a direct humidity-compatible field cleanly, record that explicitly rather than hiding the gap.

## 9. Validation checklist

Before handing results back, verify:

- station IDs exactly match the station manifest
- the date range covers `2024-01-01` to `2025-12-31`
- the daily CSV has the required columns
- the hourly precursor file has the required available columns
- `pred_tmax >= pred_tmin` for valid rows
- variable units are documented
- interpolation method is documented
- metadata JSON is complete

Also verify that the exported daily CSV can be consumed directly by the downstream local evaluator without schema edits.

## 10. Handoff back to `Offset-prediction`

The local `Offset-prediction` repo is the benchmark and evaluation hub.

Downstream evaluation happens with:

- `scripts/export_station_manifest.py`
- `scripts/evaluate_gridded_baseline.py`
- `scripts/validate_mpt_benchmarks.py`

The Aurora repo does **not** need to reimplement those local comparison scripts.

Its job is to produce clean station-aligned exports that match the shared contract.

## Final note

The target narrative is:

`Aurora as a coarse meteorological precursor baseline for a fog-focused station-level study`

Not:

- pollution workflow
- direct fog model
- independent downstream benchmark framework
