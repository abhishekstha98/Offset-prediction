# NeurIPS Fog Positioning

Date: 2026-05-02

## Recommended narrative

Use **fog-related low-visibility forecasting** as the primary narrative, not pollution as the main task.
Keep the **offset-prediction mechanism as the backbone** of the architecture.

Why:

- Fog and low visibility are more directly tied to the station-scale and sub-grid argument than pollution.
- Pollution forecasting is scientifically important, but current Earth-system foundation models already cover air-quality forecasting at coarser resolutions, which makes it a more crowded benchmark story.
- Visibility is operationally meaningful for airports, transport, and public safety, and it stays closer to the core meteorological and boundary-layer physics that this project already models.

So the preferred first claim is:

`Can a lightweight physics-aware spatiotemporal graph model predict station-level fog or low-visibility risk from coarse meteorological forcing, especially in sparse observation networks?`

Pollution should remain a secondary extension, not the title-level narrative for v1.

## Title options

Recommended title:

`Physics-Aware Spatiotemporal Graph Transformers for Station-Level Fog and Low-Visibility Forecasting in Sparse Meteorological Networks`

Strong shorter alternative:

`Station-Level Fog and Low-Visibility Forecasting with Physics-Aware Spatiotemporal Graph Transformers`

If you want to emphasize the scale gap explicitly:

`Bridging the Scale Gap in Fog Forecasting with Physics-Aware Spatiotemporal Graph Transformers`

If you want to keep the transfer-learning angle visible:

`Physics-Aware Spatiotemporal Graph Transformers for Fog Forecasting under Sparse Meteorological Observations`

## Backbone framing

The cleanest research framing is:

`shared spatiotemporal backbone -> offset correction head + fog/visibility head`

Interpretation:

- the shared backbone learns station-scale meteorological structure from coarse forcing
- the offset head preserves the original correction task and gives an interpretable physics-aware intermediate target
- the fog head turns the same latent state into an event prediction task

This is stronger than replacing offset prediction entirely, because it makes the model easier to defend scientifically and easier to transfer to sparse networks.

## Architecture changes already aligned with this framing

The current architecture is now better prepared for a fog paper because it has:

- temporal self-attention over station histories
- learned temporal position embeddings
- learned temporal attention pooling instead of only using the last time step
- spatial message passing with edge features:
  - distance
  - delta latitude
  - delta longitude
  - delta elevation
- a shared-backbone path that can emit:
  - offset predictions
  - fog / low-visibility logits

These choices better match fog forecasting, where the timing of cooling, moistening, and mixing changes matters as much as the latest snapshot.

## Next architecture steps for the actual fog model

For the first fog-specific experiment, the model should evolve in this order:

1. Keep the current station graph.
2. Move from daily to hourly inputs.
3. Use temporal windows such as `6 h`, `12 h`, or `24 h`.
4. Keep the offset head, and add fog supervision using one of:
   - binary fog occurrence
   - low-visibility event
   - visibility class
5. Add strong non-neural baselines:
   - persistence
   - threshold/rule-based fog baseline
   - station-wise logistic regression or gradient boosting
6. Keep GraphCast / Aurora as coarse-grid comparison baselines after station interpolation.

## Minimal defensible NeurIPS story

The strongest defensible v1 story is:

`Large AI weather models are powerful at synoptic scales, but fog and low visibility remain strongly local, terrain-sensitive, and boundary-layer-driven. A lightweight spatiotemporal graph transformer can exploit sparse station networks and local relational structure to improve station-level fog-risk prediction from coarse forcing.`

## Current limitation to state clearly

Until direct visibility or fog labels are integrated, the repo should still be described as a **fog-ready precursor architecture**, not a completed fog forecasting system.

## Useful external references

- GraphCast: https://deepmind.google/research/publications/22598/
- Aurora: https://www.nature.com/articles/s41586-025-09005-y
- Review of ML for atmospheric extremes including low-visibility forecasting:
  https://link.springer.com/article/10.1007/s00704-023-04571-5
- Recent low-visibility prediction example:
  https://link.springer.com/article/10.4209/aaqr.240145
