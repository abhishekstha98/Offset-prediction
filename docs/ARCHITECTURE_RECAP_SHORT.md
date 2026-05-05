# Architecture Recap Short

Date: 2026-05-02

## One-line summary

The current project implements a **shared spatiotemporal graph backbone** that predicts station-level temperature offsets and can optionally attach a fog / low-visibility head on top of the same latent state.

## Core idea

The model is:

```text
input meteorology -> temporal encoder -> spatial graph message passing -> shared latent state
                   -> offset head
                   -> optional fog head
```

This keeps **offset prediction as the backbone** while preparing the system for fog forecasting.

## Inputs

Current default input is a `17`-feature station-aligned meteorological vector:

```text
mx2t, mn2t, era5_t2m, era5_d2m, UG_era5,
dewpoint_spread_2m, rh_2m,
era5_u10, era5_v10, wind_speed_10m,
theta_v_2m, theta_v_delta_1d,
t2m_delta_1d, dewpoint_spread_delta_1d,
height, sin_doy, cos_doy
```

The model accepts either:

- `(N, F)` single-step graph input
- `(T, N, F)` temporal graph input

## Graph structure

- nodes = stations
- edges = directed `k`-nearest-neighbor connections
- edge features:

```text
distance_km, delta_lat, delta_lon, delta_height
```

These edge features let the model use spatial distance and terrain gradients during message passing.

## Backbone

The shared backbone has three stages:

1. **Node encoder**
   - MLP projects raw features into hidden space

2. **Temporal encoder**
   - Transformer self-attention over previous timesteps for each station
   - learned temporal position embeddings
   - attention pooling or last-step pooling

3. **Spatial encoder**
   - stacked `TransformerConv` layers over the station graph
   - residual connections and layer normalization

The result is one shared latent vector per station.

## Outputs

### Offset head

Primary implemented task:

```text
[Delta Tmax, Delta Tmin]
```

### Fog head

Optional downstream head:

```text
fog logit or low-visibility logit
```

This is enabled only when fog labels are available.

## Loss

Current training supports:

- masked MAE for temperature offsets
- optional BCE-with-logits loss for fog labels

Combined form:

```text
L_total = L_offset + lambda_fog * L_fog
```

Missing labels are masked, but stations remain in the graph.

## Current status

What is already implemented:

- fog-relevant feature engineering
- temporal sequence support
- temporal self-attention
- spatial edge-aware message passing
- shared backbone with offset head
- optional fog head
- multitask loss path

What is still missing for a real fog paper:

- hourly fog / visibility labels
- full hourly merged dataset
- end-to-end fog training runs
- baseline comparisons

## Correct description today

The project is best described as:

```text
a fog-ready spatiotemporal graph backbone with offset prediction as the main implemented task
and fog prediction as the attached downstream head
```

## Recommended paper framing

Use this framing:

```text
shared spatiotemporal backbone
-> offset correction head
-> fog / low-visibility head
```
   