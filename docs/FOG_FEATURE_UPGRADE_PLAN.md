# Fog Feature Upgrade Plan

Date: 2026-04-25

## Summary

This plan defines the next training setup for fog-formation-oriented feature development in the existing offset-prediction project.

The immediate goal is **not** to build a full fog classifier. The current repo predicts station-level daily temperature offsets:

```text
offset_tmax = TX - mx2t
offset_tmin = TN - mn2t
```

The next step is to upgrade the model inputs with ERA5-Land-derived humidity, wind, and stability proxies that are physically relevant to fog, inversion, and low-visibility conditions.

## Recommended Model Direction

Continue from:

```text
baseline_stlobo
```

Do not continue from the ablated multi-channel variants as the main model. The ablations were useful diagnostics, but none outperformed the full multi-channel model, and the full multi-channel model did not outperform the simpler baseline in the completed comparisons.

Use this sequence:

1. Continue with the **baseline ST-LOBO architecture** as the primary model.
2. Add the fog/inversion proxy features to the input dataset.
3. Rerun baseline ST-LOBO with the expanded features.
4. Later, rerun full multi-channel with the same expanded features as a comparison.
5. Do not use a channel-removed ablation as the main next model.

## Past Experiment Result Summary

Completed results from `docs/EXPERIMENTS_AND_RESULTS.md` and current logs:

```text
Baseline ST-LOBO:
  Val MAE Tmax: 1.6688
  Val MAE Tmin: 1.8009
  Test MAE Tmax: 1.7848

Multi-channel ST-LOBO:
  Val MAE Tmax: 1.7126
  Val MAE Tmin: 1.8125
  Test MAE Tmax: 1.7968

Full multi-channel SLOBO:
  Val MAE Tmax: 1.7582
  Val MAE Tmin: 1.8707

Best ablation, remove temperature:
  Val MAE Tmax: 1.7668
  Val MAE Tmin: 1.8823

Worst ablation, remove pressure:
  Val MAE Tmax: 1.7795
  Val MAE Tmin: 1.9123
```

## Main Conclusion From Experiments

The best completed overall result is:

```text
baseline_stlobo
```

The best completed purely spatial result is:

```text
baseline_slobo
```

The best multi-channel setup is:

```text
full multi-channel
```

not a channel ablation.

Important interpretation for fog work:

- Removing the pressure/humidity channel caused the worst ablation result.
- This suggests humidity, pressure, seasonality, and stability-related variables should be preserved and expanded.
- The next setup should add fog-relevant variables rather than remove physical channels.

## Feature Upgrade

Expand the input dataset from the current 6-feature setup to this 17-feature setup:

```text
mx2t
mn2t
era5_t2m
era5_d2m
UG_era5
dewpoint_spread_2m
rh_2m
era5_u10
era5_v10
wind_speed_10m
theta_v_2m
theta_v_delta_1d
t2m_delta_1d
dewpoint_spread_delta_1d
height
sin_doy
cos_doy
```

Highest-priority fog/inversion features:

```text
dewpoint_spread_2m
rh_2m
wind_speed_10m
theta_v_2m
theta_v_delta_1d
```

### Derived Feature Definitions

```text
dewpoint_spread_2m = era5_t2m - era5_d2m
```

```text
rh_2m = 100 * es(era5_d2m) / es(era5_t2m)
```

```text
wind_speed_10m = sqrt(era5_u10^2 + era5_v10^2)
```

```text
theta = T * (1000 / p)^0.286
theta_v_2m = theta * (1 + 0.61q)
```

```text
theta_v_delta_1d = theta_v_2m_today - theta_v_2m_previous_day
t2m_delta_1d = era5_t2m_today - era5_t2m_previous_day
dewpoint_spread_delta_1d = dewpoint_spread_2m_today - dewpoint_spread_2m_previous_day
```

For the first day of each station time series, fill lag deltas with `0.0`.

### Pressure Handling For `theta_v_2m`

Use this priority order:

1. If `surface_pressure`, `era5_sp`, or `sp` exists in the CSV, use it.
2. If pressure is absent, estimate pressure from station `height` using a standard-atmosphere approximation.

This keeps the feature usable with the current `merged.csv`, which already contains:

```text
era5_t2m
era5_d2m
era5_u10
era5_v10
era5_ws10
```

## Code Changes

### `src/data/dataset.py`

Add a feature-engineering helper that:

- standardizes column aliases
- computes dewpoint spread
- computes relative humidity
- computes wind speed
- computes virtual potential temperature
- computes station-wise daily lag features
- exposes one canonical feature-list constant

The dataset should use the canonical feature list by default.

Preserve legacy behavior where practical:

- If an old scaler contains 6 features and has no `feature_columns`, use the old 6-feature list.
- If a new scaler contains `feature_columns`, use exactly those columns.

### `src/config.py`

Change:

```text
in_features: 6
```

to:

```text
in_features: 17
```

Update comments so the config documents the new feature list.

### `src/models/factory.py`

If the multi-channel model is rerun, update the channel registry to match the expanded feature list:

```text
temperature:
  mx2t, mn2t, era5_t2m, era5_d2m, t2m_delta_1d

humidity_stability:
  UG_era5, dewpoint_spread_2m, rh_2m, theta_v_2m,
  theta_v_delta_1d, dewpoint_spread_delta_1d, sin_doy, cos_doy

wind:
  era5_u10, era5_v10, wind_speed_10m

terrain:
  height
```

Default `active_channels` should use all channels.

### `src/train.py`

Save the feature list in the checkpoint:

```text
model_config["feature_columns"]
```

Also save the feature list in the scaler object:

```text
scaler["feature_columns"]
```

### `src/inference.py`

Load feature columns from the scaler or checkpoint.

Fallback behavior:

- new checkpoint: use saved `feature_columns`
- old checkpoint/scaler: infer 6-feature legacy mode

### Tests

Update synthetic test data to include:

```text
era5_t2m
era5_d2m
era5_u10
era5_v10
era5_ws10
```

Replace hard-coded `6` input-feature assumptions with the canonical feature-list length.

Add a test that asserts:

```text
dataset[0]["x"] has no NaN
dataset[0]["x"] has no Inf
dataset[0]["x"].shape[1] == 17
```

## Validation Commands

Run the unit tests:

```bash
python -m pytest src/tests/test_pipeline.py
```

Run a one-fold smoke training pass:

```bash
python src/train.py --epochs 1 --fold 0
```

Verify:

```text
batch["x"].shape[1] == 17
scaler["mean"].shape[0] == 17
scaler["feature_columns"] matches the new feature list
model forward pass returns shape (N, 2)
no NaN or Inf appears in engineered inputs
```

## Acceptance Criteria

The implementation is complete when:

- the expanded dataset builds without NaN/Inf features
- tests pass
- a 1-epoch smoke training run completes
- the checkpoint records the expanded input dimension and feature names
- old checkpoints can still be loaded or fail with a clear feature-dimension explanation

## Assumptions

- The input file remains `merged.csv`.
- The target remains daily `Delta Tmax` and `Delta Tmin` offset correction.
- The first implementation is a daily feature upgrade, not an hourly fog classifier.
- True 2-3 hour pre-fog virtual-potential-temperature tendency requires a later hourly ERA5-Land dataset.
- Airport visibility or fog labels will be added in a later phase.
