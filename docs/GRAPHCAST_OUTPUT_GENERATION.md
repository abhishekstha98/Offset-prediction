# GraphCast Output Generation

This file only covers generating GraphCast model outputs. It does not cover downstream evaluation.

Use this when the goal is to produce the GraphCast deliverables that the `Offset-prediction` repo can consume:

- `station_daily_tmax_tmin_graphcast.csv`
- `station_hourly_precursors_graphcast.parquet`
- `run_metadata_graphcast.json`

## Scope

GraphCast is being used here as a coarse meteorological baseline, not as a direct fog model.

The immediate milestone is:

- generate station-aligned daily `pred_tmax` and `pred_tmin` for `2024-01-01` to `2025-12-31`

## Local entrypoints

Use the local GraphCast clone already present in this repo:

- `graphcast/README.md`
- `graphcast/graphcast_demo.ipynb`
- `graphcast/docs/cloud_vm_setup.md`

Those are the main local references for loading a checkpoint, selecting example input data, running rollout, and saving predictions.

## Recommended model choice

Use one model choice explicitly and record it:

- `GraphCast_small` for lower-memory debugging and local experimentation
- `GraphCast` for the main ERA5-style high-resolution baseline
- `GraphCast_operational` only if you are intentionally using HRES-style initialization

For this project, start with:

- `GraphCast_small` for the smoke test
- `GraphCast` for the real export if compute allows

## Input source

The local GraphCast repo describes these sources:

- ERA5 data
- WeatherBench2 ERA5 Zarr access
- HRES / HRES-fc0 for operational-style workflows

Do not leave the input source implicit. Record it in `run_metadata_graphcast.json`.

## Environment and compute expectations

GraphCast is the heavier setup.

The local docs point to notebook-driven inference and note that:

- TPU/GPU acceleration is strongly preferred
- the notebook is the main supported entrypoint
- cloud TPU or large GPU setups are often needed for non-trivial runs

For a simple local start:

```bash
cd graphcast
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install xarray netcdf4 scipy pandas pyarrow fsspec gcsfs "zarr<3" jupyter
```

If you plan to use the full notebook flow, expect to add the JAX stack required by the local GraphCast package.

## Shared inputs from `Offset-prediction`

Before running GraphCast, make sure the station manifest exists:

```bash
cd ..
python scripts/export_station_manifest.py
```

This produces:

- `baseline_assets/netherlands_station_manifest.csv`

GraphCast should use that manifest as the station list for interpolation/export.

## Recommended execution order

### 1. Start with the local notebook entrypoint

Use `graphcast/graphcast_demo.ipynb` first.

That notebook is the clearest local reference for:

- loading checkpoints
- selecting example datasets
- running prediction rollout
- inspecting variables such as `2m_temperature`

### 2. Run a short smoke test

Start with:

- a short window only
- one checkpoint only
- a single export path only

Verify:

- checkpoint loading works
- the model returns forecast outputs
- `2m_temperature` is present
- time stamps are valid forecast times
- coordinate names are usable for later interpolation

### 3. Choose the compute path explicitly

Use one path and document it:

- local notebook with enough GPU
- cloud TPU / VM path from `graphcast/docs/cloud_vm_setup.md`

If you move to GPU-specific inference, keep that fact in metadata because attention implementation and runtime behavior may differ.

### 4. Save gridded predictions first

Before any station export, save raw gridded predictions for a short run:

- Zarr is the most natural choice in the GraphCast notebook flow
- NetCDF is fine if you convert after inference

At minimum preserve:

- `2m_temperature`
- any wind fields you plan to export
- any pressure field you plan to export
- a humidity-compatible field if available in your chosen output contract

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
- `source_model = graphcast`

### 6. Aggregate hourly station temperature to daily Tmax/Tmin

From the interpolated station time series:

- compute daily `pred_tmax`
- compute daily `pred_tmin`

Use one day-boundary rule and record it in metadata.

Recommended rule for this project:

- aggregate by local day in `Europe/Amsterdam`

### 7. Write the required final files

Write:

- `station_daily_tmax_tmin_graphcast.csv`
- `station_hourly_precursors_graphcast.parquet`
- `run_metadata_graphcast.json`

Required daily CSV columns:

- `station`
- `date`
- `pred_tmax`
- `pred_tmin`

## Minimum metadata to record

Include at least:

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

## Practical advice

- Use `GraphCast_small` first unless you already know the larger model fits your hardware.
- Keep the first successful smoke-test export on disk. It will save time when the full run fails later.
- Do not postpone schema normalization. Convert the final station export to the shared field names during GraphCast export.
- If you only manage to produce station daily CSV initially, that is enough for the downstream evaluator.

## Success condition

This phase is complete when all three files exist and the daily CSV has the exact schema:

```text
station,date,pred_tmax,pred_tmin
```
