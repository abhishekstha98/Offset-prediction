# GraphCast / Aurora Baseline Prep

This note prepares the external-baseline comparison that Alicia asked for:

- run GraphCast and/or Aurora on the 23 Netherlands stations
- interpolate their outputs to station locations
- compute station-level Tmin / Tmax error

The goal is to answer one immediate research question:

`How large is the gap between the current repo results and strong external coarse-grid baselines?`

## 1. Official sources

GraphCast:

- Google DeepMind repo: https://github.com/google-deepmind/graphcast
- Best starting point from the repo: `graphcast_demo.ipynb`

Aurora:

- Microsoft repo: https://github.com/microsoft/aurora
- ERA5 example docs: https://microsoft.github.io/aurora/example_era5.html

## 2. Recommended evaluation protocol

Use the repo's final held-out test period first:

- primary window: `2024-01-01` to `2025-12-31`

Why:

- this aligns with the current repo's held-out test split
- it avoids comparing external baselines on the training years that were used for local model selection

Optional secondary analysis:

- evaluate `2020-01-01` to `2023-12-31` as a larger validation-style window
- keep that separate from the primary test result

## 3. What output you need from GraphCast / Aurora

For a first-pass comparison, you only need a gridded temperature field with valid timestamps:

Preferred:

- sub-daily 2m temperature on the model grid

Acceptable:

- daily Tmax and daily Tmin already derived and stored in a file

If you use sub-daily 2m temperature, this repo provides an evaluation script that:

- interpolates grid values to the 23 station coordinates
- aggregates sub-daily temperatures to daily Tmax / Tmin
- computes MAE against the station observations in `merged.csv`

## 4. Files added for this workflow

### Station manifest export

Script:

- `scripts/export_station_manifest.py`

Generated manifest:

- `baseline_assets/netherlands_station_manifest.csv`

### External baseline evaluator

Script:

- `scripts/evaluate_gridded_baseline.py`

This script evaluates exported GraphCast/Aurora outputs once they exist on disk.

It now supports three input styles:

- gridded sub-daily temperature file
- gridded daily Tmax/Tmin file
- station-level daily CSV with `station`, `date`, `pred_tmax`, `pred_tmin`

## 5. Step-by-step workflow

### Step 1. Export the station manifest

From the repo root:

```bash
python scripts/export_station_manifest.py
```

This creates:

- `baseline_assets/netherlands_station_manifest.csv`

The file contains:

- station id
- station name
- latitude
- longitude
- height
- start date
- end date
- row count

### Step 2. Run GraphCast or Aurora externally

Produce a file containing either:

- sub-daily 2m temperature
- or daily Tmax / Tmin

Practical note:

- Aurora is currently easier to get running locally because it is packaged as `microsoft-aurora` and has documented ERA5 examples.
- GraphCast is available officially, but its workflow is more notebook / JAX oriented and usually needs more setup.

### Step 3. Evaluate the exported predictions

#### Option A: sub-daily temperature field

Example for Aurora if the exported variable is `2t` in Kelvin:

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model aurora \
  --prediction-path /path/to/aurora_predictions.nc \
  --mode subdaily_t2m \
  --temp-var 2t \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --kelvin
```

Example for GraphCast if the exported variable is `2m_temperature` in Kelvin:

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model graphcast \
  --prediction-path /path/to/graphcast_predictions.nc \
  --mode subdaily_t2m \
  --temp-var 2m_temperature \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --kelvin
```

#### Option B: daily Tmax / Tmin already exported

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model aurora \
  --prediction-path /path/to/daily_predictions.nc \
  --mode daily_tmax_tmin \
  --tmax-var tmax \
  --tmin-var tmin \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

#### Option C: station-level daily CSV

If you can export GraphCast or Aurora predictions directly to a CSV indexed by station and date, you do not need NetCDF/Zarr at all.

Expected CSV columns by default:

- `station`
- `date`
- `pred_tmax`
- `pred_tmin`

Example:

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model aurora \
  --prediction-path /path/to/aurora_station_daily.csv \
  --mode csv_daily_tmax_tmin \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

If your CSV uses different column names:

```bash
python scripts/evaluate_gridded_baseline.py \
  --source-model graphcast \
  --prediction-path /path/to/graphcast_station_daily.csv \
  --mode csv_daily_tmax_tmin \
  --station-col station_id \
  --date-col valid_date \
  --pred-tmax-col tmax_pred \
  --pred-tmin-col tmin_pred \
  --start-date 2024-01-01 \
  --end-date 2025-12-31
```

## 6. Outputs from the evaluation script

For each source model, the script writes:

- `outputs/external_baselines/<model>/matched_station_predictions.csv`
- `outputs/external_baselines/<model>/station_mae.csv`
- `outputs/external_baselines/<model>/summary.json`

The key number to look at first is in `summary.json`:

- `mae_tmax`
- `mae_tmin`

Also compare against:

- `era5_mae_tmax_on_same_subset`
- `era5_mae_tmin_on_same_subset`

## 7. Important methodological caveats

### Time aggregation

GraphCast / Aurora typically work with sub-daily fields, while this repo's targets are daily Tmax / Tmin.

So the evaluation requires a daily aggregation choice.

Current script behavior:

- it groups sub-daily times by `Europe/Amsterdam` local day by default
- it then computes daily max and daily min from the interpolated 2m temperature values

This is the right default for a first pass, but it should still be mentioned explicitly in the paper.

### Interpolation

Current script supports:

- `linear`
- `nearest`

Use `linear` first unless the baseline output format forces something else.

### Variable naming

GraphCast and Aurora exports may use different variable names and coordinate names.

The evaluation script is flexible, but you may need to specify:

- `--temp-var`
- `--tmax-var`
- `--tmin-var`
- `--time-name`
- `--lat-name`
- `--lon-name`

## 8. What a successful first comparison should produce

For both GraphCast and Aurora, you want one concise table:

| Model | Test window | Interp | Daily aggregation | MAE Tmax | MAE Tmin |
|---|---|---|---|---:|---:|
| ERA5 | 2024-2025 | station CSV direct | already daily | ? | ? |
| GraphCast | 2024-2025 | linear | local-day max/min from 2m T | ? | ? |
| Aurora | 2024-2025 | linear | local-day max/min from 2m T | ? | ? |
| Baseline ST-LOBO model | current repo | native | native | `1.7848` test Tmax only currently | TBD for Tmin test if evaluated |

That table is the fastest way to decide whether this comparison materially strengthens the paper.

## 9. Recommended next move

If only one external baseline can be prepared first, start with Aurora:

- the packaging is easier
- the docs include an ERA5 example
- it is the fastest route to obtaining a first external MAE number

Once one baseline MAE exists, you can decide whether GraphCast setup is worth the extra effort immediately or whether the paper direction already looks promising.
