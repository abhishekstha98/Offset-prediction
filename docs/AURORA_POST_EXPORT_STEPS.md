# Aurora Post-Export Steps

This file starts after Aurora outputs already exist on disk.

Use it when you already have one of these:

- `station_daily_tmax_tmin_aurora.csv`
- a gridded Aurora NetCDF/Zarr export with sub-daily `2t`
- a gridded Aurora daily Tmax/Tmin export

## Goal

Validate Aurora as an external baseline against the local station observations in `merged.csv`.

## Files in this repo that handle the next stage

- `scripts/evaluate_gridded_baseline.py`
- `baseline_assets/netherlands_station_manifest.csv`
- `outputs/external_baselines/`

The evaluation script does not run Aurora. It only evaluates exported predictions after they already exist.

## Step 1. Confirm the station manifest

From the repo root:

```bash
python scripts/export_station_manifest.py
```

This refreshes:

- `baseline_assets/netherlands_station_manifest.csv`

## Step 2. Pick the correct evaluation mode

Use exactly one mode:

- `csv_daily_tmax_tmin` if you already have `station_daily_tmax_tmin_aurora.csv`
- `subdaily_t2m` if you have gridded sub-daily `2t`
- `daily_tmax_tmin` if you have gridded daily Tmax/Tmin

## Step 3. Run the evaluator

### Option A. Aurora station daily CSV

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model aurora \
  --prediction-path /path/to/station_daily_tmax_tmin_aurora.csv \
  --mode csv_daily_tmax_tmin \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

### Option B. Aurora gridded sub-daily temperature

Use this if the export still contains the gridded `2t` field:

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model aurora \
  --prediction-path /path/to/aurora_predictions.nc \
  --mode subdaily_t2m \
  --temp-var 2t \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --kelvin
```

### Option C. Aurora gridded daily Tmax/Tmin

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model aurora \
  --prediction-path /path/to/aurora_daily_predictions.nc \
  --mode daily_tmax_tmin \
  --tmax-var tmax \
  --tmin-var tmin \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

If your file uses different names for time, latitude, longitude, or variables, pass:

- `--time-name`
- `--lat-name`
- `--lon-name`
- `--temp-var`
- `--tmax-var`
- `--tmin-var`

## Step 4. Review outputs

The evaluator writes to:

- `outputs/external_baselines/aurora/matched_station_predictions.csv`
- `outputs/external_baselines/aurora/station_mae.csv`
- `outputs/external_baselines/aurora/summary.json`

Check `summary.json` first.

The first numbers to inspect are:

- `mae_tmax`
- `mae_tmin`
- `era5_mae_tmax_on_same_subset`
- `era5_mae_tmin_on_same_subset`

## Step 5. Sanity-check the matched rows

Open `matched_station_predictions.csv` and verify:

- the date range is correct
- the station count is correct
- `pred_tmax` and `pred_tmin` are in Celsius if you intended Celsius outputs
- the row count is not suspiciously small

If the output is empty or nearly empty, the usual causes are:

- wrong station ids
- wrong date column semantics
- timezone/day-boundary mismatch
- no overlap between export dates and the requested evaluation window

## Step 6. Decide whether the export is good enough

Use this rule:

- if the evaluator runs cleanly and the summary numbers are plausible, freeze that Aurora daily CSV as the baseline artifact for this iteration
- if the evaluator fails or the numbers look structurally wrong, fix the export stage first rather than patching around it in evaluation

## Step 7. Archive the run cleanly

Keep these together:

- Aurora export files
- `run_metadata_aurora.json`
- evaluation outputs from `outputs/external_baselines/aurora/`

That makes the result reproducible later.

## Optional next step

After the daily baseline is stable, extend the Aurora export to the hourly precursor file:

- `station_hourly_precursors_aurora.parquet`

That is the natural follow-up for later fog-feature analysis.
