# Aurora Output Generation

This file only covers generating Aurora model outputs. It does not cover downstream evaluation.

Use this when the goal is to produce the Aurora deliverables that the `Offset-prediction` repo can consume:

- `station_daily_tmax_tmin_aurora.csv`
- `station_hourly_precursors_aurora.parquet`
- `run_metadata_aurora.json`

## Scope

Aurora is being used here as a coarse meteorological baseline, not as a direct fog model.

The immediate milestone is:

- generate station-aligned daily `pred_tmax` and `pred_tmin` for `2024-01-01` to `2025-12-31`

## Local entrypoints

Use the local Aurora clone already present in this repo:

- `aurora/README.md`
- `aurora/docs/example_era5.ipynb`
- `aurora/docs/example_hres_t0.ipynb`
- `aurora/aurora/foundry/demo/hres_t0_data.py`

Those are the best local references for building an Aurora `Batch`, loading checkpoints, and running rollout/inference.

## Recommended model choice

Pick the model variant based on the input source:

- Use `AuroraPretrained()` for ERA5-style 0.25° inputs.
- Use `Aurora()` for HRES T0 style inputs.
- Use `AuroraSmallPretrained()` only for smoke tests and debugging.

The local Aurora docs explicitly describe:

- required surface variables: `2t`, `10u`, `10v`, `msl`
- static variables: `lsm`, `slt`, `z`
- atmospheric variables: `t`, `u`, `v`, `q`, `z`
- pressure levels: `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`

## Input contract you need to satisfy

Aurora expects a `Batch` with:

- two history timesteps
- gridded latitude/longitude coordinates
- surface variables with keys `2t`, `10u`, `10v`, `msl`
- static variables `lsm`, `slt`, `z`
- atmospheric variables `t`, `u`, `v`, `q`, `z`

Units expected by Aurora:

- `2t` in Kelvin
- `10u` and `10v` in `m/s`
- `msl` in `Pa`

## Environment setup

Work inside the local Aurora clone:

```bash
cd aurora
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install xarray netcdf4 scipy pandas pyarrow fsspec gcsfs "zarr<3" huggingface_hub cdsapi jupyter
```

If you already manage environments another way, keep the same package set.

## Shared inputs from `Offset-prediction`

Before running Aurora, make sure the station manifest exists:

```bash
cd ..
python scripts/export_station_manifest.py
```

This produces:

- `baseline_assets/netherlands_station_manifest.csv`

Aurora should use that manifest as the station list for interpolation/export.

## Recommended execution order

### 1. Smoke test the model locally

Start with the smallest possible run:

- 2 to 7 days only
- one model variant only
- one export path only

Verify:

- checkpoint download works
- input batch creation works
- rollout returns a prediction object
- predicted surface fields include at least `2t`
- timestamps behave as forecast valid times

### 2. Decide the input source

Use one of these and record it in metadata:

- ERA5-style inputs with `AuroraPretrained`
- HRES T0 style inputs with `Aurora`

Do not mix sources implicitly. Write the choice into `run_metadata_aurora.json`.

### 3. Build a reusable Aurora runner

The local repo does not contain a dedicated station-export script for this study yet.

So create or adapt a small runner in the `aurora/` repo based on the example notebooks that does the following:

1. Load the initial-condition dataset for the target time window.
2. Convert it into an Aurora `Batch`.
3. Load the chosen checkpoint.
4. Run inference / rollout.
5. Convert predictions to `xarray` or tabular form.
6. Keep at least these predicted fields if available:
   - `2t`
   - `10u`
   - `10v`
   - `msl`
   - one humidity-compatible field if available in your chosen setup

### 4. Export gridded predictions first

For debugging, first save the raw gridded predictions for a short window:

- NetCDF is easiest for inspection
- Zarr is fine if that is easier in your workflow

Recommended debugging checks:

- confirm latitude and longitude coordinate names
- confirm time coordinate name
- confirm `2t` is still in Kelvin unless you intentionally convert it
- confirm field shapes match `time x lat x lon` after squeezing batch/history axes

### 5. Interpolate to the 23 stations

Using `baseline_assets/netherlands_station_manifest.csv`:

- interpolate gridded outputs to station latitude/longitude
- use one explicit interpolation rule
- default to `linear`
- if `linear` fails at edges, fall back to `nearest` and record that choice

The station-level hourly export should contain:

- `station`
- `time`
- `pred_t2m`
- `pred_u10`
- `pred_v10`
- `pred_pressure`
- one humidity-compatible field if available
- `source_model`

Use:

- UTC timestamps
- one row per `station x hour`
- `source_model = aurora`

### 6. Aggregate hourly station temperature to daily Tmax/Tmin

From the interpolated station time series:

- compute daily `pred_tmax`
- compute daily `pred_tmin`

Use one day-boundary rule and record it in metadata.

Recommended rule for this project:

- aggregate by local day in `Europe/Amsterdam`

### 7. Write the required final files

Write:

- `station_daily_tmax_tmin_aurora.csv`
- `station_hourly_precursors_aurora.parquet`
- `run_metadata_aurora.json`

Required daily CSV columns:

- `station`
- `date`
- `pred_tmax`
- `pred_tmin`

## Minimum metadata to record

Include at least:

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

## Practical advice

- Get the daily CSV working first. That validates most of the pipeline quickly.
- Keep raw gridded sample outputs from the smoke test. They make debugging much faster.
- Do not hand-convert variable names at evaluation time. Normalize them during Aurora export.
- If your Aurora workflow only gives you station daily CSV directly, that is acceptable. The downstream evaluator supports it.

## Success condition

This phase is complete when all three files exist and the daily CSV has the exact schema:

```text
station,date,pred_tmax,pred_tmin
```
