# Nepal-Focused Implementation Plan

Date: 2026-04-13

## 1. Recommended research framing

The strongest version of the project is not "general downscaling for everything in Nepal".

It should be:

`A lightweight graph-based offset correction model that learns station-level Tmin/Tmax corrections for coarse global products (ERA5, GraphCast, Aurora) and remains robust when the station network is sparse, with Nepal as the motivating deployment case.`

Recommended application niche:

- Primary niche: nocturnal cold bias, local cold pooling, and inversion-sensitive Tmin errors in Nepal-like terrain
- First downstream application: fog / visibility risk or frost-risk warning
- Secondary extension only if time permits: air-quality concentration risk under low-wind inversion conditions

Why this framing is stronger:

- It directly addresses the coarse-resolution limitation of GraphCast and Aurora.
- It matches the current repo, which already predicts temperature offsets.
- It gives a Nepal-specific motivation without requiring the whole paper to become an air-quality paper.
- It supports a clean story around sparse-station robustness and transfer learning.

## 2. Core research questions

### Main question

Can a lightweight MPT/GNN trained on historical station data correct coarse-grid temperature products at individual stations better than raw ERA5 and coarse AI weather models?

### Secondary questions

1. Does the model outperform raw ERA5, GraphCast, and Aurora at station-level Tmin/Tmax on the same stations and dates?
2. Does performance remain strong as station density decreases?
3. Does pretraining on a dense network and fine-tuning on a sparse target network help?
4. Do inversion-sensitive settings show larger gains than ordinary conditions?
5. Does improved Tmin correction improve a simple downstream fog/frost proxy?

## 3. Hypotheses

1. Station-level correction will beat raw coarse-grid products because sub-grid terrain, local exposure, and cold-pooling effects are not resolved at 0.25 degrees.
2. Gains should be strongest for Tmin, nighttime, winter, post-monsoon, and low-wind stable-boundary-layer conditions.
3. A lightweight graph model should stay competitive under sparse sampling because message passing encodes spatial structure better than purely local regression.
4. Transfer from a denser source region should help if the target task is framed as bias correction rather than direct weather generation.

## 4. Required literature review

The literature review should be written in five streams. Do not write one generic "related work" section.

### Stream A. Global AI weather models and their spatial limits

Purpose:

- Establish GraphCast and Aurora as strong but coarse baselines.
- Motivate why station-level correction is still needed.

Required reading:

1. GraphCast: `Learning skillful medium-range global weather forecasting`
   Link: https://doi.org/10.1126/science.adi2336
   Use for:
   - state-of-the-art status
   - global medium-range skill
   - coarse-grid limitation relative to station applications

2. Aurora: `A foundation model for the Earth system`
   Link: https://www.nature.com/articles/s41586-025-09005-y
   Use for:
   - foundation-model framing
   - multi-task fine-tuning
   - official resolution facts and benchmark context

3. MAUSAM: `An Observations-focused assessment of Global AI Weather Prediction Models During the South Asian Monsoon`
   Link: https://arxiv.org/abs/2509.01879
   Use for:
   - South Asia-specific evidence
   - observation-based evaluation
   - proof that reanalysis-only evaluation can overstate performance

Key review question:

- If GraphCast and Aurora are strong globally, what failure modes remain at station scale in South Asia and complex terrain?

### Stream B. Regional and high-resolution AI weather modeling

Purpose:

- Show that the field recognizes the need for higher-resolution or observation-driven regional methods.
- Position your work as lightweight correction instead of full regional forecasting.

Required reading:

1. OMG-HD: `A High-Resolution AI Weather Model for End-to-End Forecasts from Observations`
   Link: https://arxiv.org/abs/2412.18239
   Use for:
   - observation-driven regional AI forecasting
   - high-resolution regional motivation
   - contrast with your smaller post-processing approach

2. `A regional high resolution AI weather model for the prediction of atmospheric rivers and extreme precipitation`
   Link: https://www.nature.com/articles/s41612-025-01265-9
   Use for:
   - regional high-resolution AI modeling
   - evidence that coarse global systems still leave regional gaps

Key review question:

- Why is lightweight station correction a useful alternative to building a full regional model?

### Stream C. Bias correction and station-level post-processing of reanalysis / forecast products

Purpose:

- Ground the paper in bias correction, not only in AI-weather hype.
- Show the need for station-aware correction in complex terrain.

Required reading:

1. `Bias corrections of ERA5 and ERA5-Land temperature using automatic weather station data in the Higher Central Himalaya`
   Link: https://doi.org/10.1016/j.ejrh.2025.103079
   Use for:
   - direct Himalayan relevance
   - evidence that ERA5/ERA5-Land temperature biases are materially correctable

2. `Evaluation of Daily Temperature Extremes in the ECMWF Operational Weather Forecasts and ERA5 Reanalysis`
   Link: https://www.mdpi.com/2073-4433/15/1/93
   Use for:
   - Tmin/Tmax evaluation framing
   - extreme-temperature bias context

3. `Multivariate Bias Correction of ERA5 Using in-situ Observations`
   Link: https://d197for5662m48.cloudfront.net/documents/publicationstatus/266414/preprint_pdf/54872f40c6315e2274b3d786d7fca250.pdf
   Use for:
   - multi-variable correction framing
   - benchmark ideas for classical/statistical baselines

Key review question:

- What do classical correction methods fix well, and where do graph models have an advantage?

### Stream D. Graph learning / spatiotemporal interpolation over station networks

Purpose:

- Justify the model class itself.
- Compare graph message passing against interpolation or local regression.

Required reading:

1. `Spatiotemporal Interpolation of Meteorological Fields in Complex Terrain Using Deep Graph Neural Networks`
   Link: https://www.mdpi.com/2076-3417/16/4/1755
   Use for:
   - graph learning in complex terrain
   - interpolation versus learned spatial propagation

Also review 2-4 additional papers on:

- station-network GNNs
- graph transformers for environmental sensing
- sparse-sensor interpolation

Key review question:

- What exactly is the MPT/GNN buying you beyond kriging, nearest-neighbor interpolation, or per-station ML?

### Stream E. Nepal-specific application context

Purpose:

- Make the Nepal case scientifically and socially credible.
- Choose one downstream application and support it with Nepal literature.

Required reading:

1. WeatherWave: `A Machine Learning-Integrated Web Application for Localized Weather Forecasting in Nepal`
   Link: https://doi.org/10.65091/icicset.v2i1.13
   Use for:
   - direct Nepal peer
   - lightweight localized forecasting context
   - contrast district-level RF against station-level graph correction

2. `Trends in winter fog events in the Terai region of Nepal`
   Link: https://doi.org/10.1016/j.agrformet.2018.04.018
   Use for:
   - fog application motivation
   - visibility and agricultural impacts

3. `Dynamics of PM2.5 concentrations in Kathmandu Valley, Nepal`
   Link: https://doi.org/10.1016/j.jhazmat.2009.02.086
   Use for:
   - inversion and wind-air-quality relationship
   - justification if air quality becomes a secondary extension

4. `Ambient air quality in the Kathmandu Valley, Nepal, during the NAMaSTE campaign`
   Link: https://doi.org/10.5194/acp-20-2927-2020
   Use for:
   - valley-scale pollution and meteorology context

Key review question:

- Which Nepal impact story is strongest and still feasible with available data?

## 5. Literature review output structure

The literature review deliverable should contain:

1. A one-page comparison table with columns:
   - paper
   - region
   - resolution
   - target variable
   - input data
   - model family
   - station-based evaluation yes/no
   - sparse-network relevance
   - direct lesson for this project

2. A gap statement with three claims:
   - global AI weather models are strong but too coarse for station-scale Nepal use cases
   - Nepal-localized work exists, but it is mostly district-level and not station-graph correction
   - there is an opening for lightweight, sparse-network, station-level correction

3. A benchmark matrix listing:
   - raw ERA5 / ERA5-Land
   - GraphCast
   - Aurora
   - persistence
   - per-station linear model
   - per-station random forest / XGBoost
   - quantile mapping or delta correction

## 6. Implementation plan

### Phase 1. Lock scope and benchmark definition

Deliverable:

- final one-sentence problem statement
- final application choice
- benchmark list
- evaluation split policy

Decisions to lock:

1. Primary prediction targets: daily Tmin and Tmax
2. Primary claim: station-level correction of coarse products
3. Primary geography story: Nepal deployment case, Netherlands as source pretraining environment unless Nepal station data is immediately available
4. Primary downstream application: fog/frost proxy, not full air-quality forecasting

### Phase 2. Data inventory and harmonization

Required datasets:

- station observations
  - Netherlands KNMI data already used in repo
  - Nepal station data if available
- coarse predictors
  - ERA5 or ERA5-Land
  - GraphCast output
  - Aurora output
- static spatial covariates
  - elevation
  - slope / aspect if available
  - land cover or vegetation proxy if available
- dynamic covariates
  - humidity
  - wind speed
  - pressure
  - cloud cover / radiation
  - precipitation

Recommended first-pass data policy:

- Start with ERA5 or ERA5-Land plus existing station observations.
- Add GraphCast and Aurora only for evaluation, not as mandatory training inputs in the first experiment.
- Do not add NASA evapotranspiration in the first round.

Reason:

- It adds temporal and spatial alignment work before the core claim is tested.
- ERA5-Land already includes evaporation / latent-heat-related variables, but these need careful use because ECMWF documents known issues for some evapotranspiration components.

If evapotranspiration is tested later:

- prefer a clearly aligned product with explicit temporal aggregation
- compare "with ET" versus "without ET" as an ablation
- keep it optional, not part of the main contribution

### Phase 3. Baselines

You need external and classical baselines.

Minimum baseline set:

1. Raw ERA5 / ERA5-Land at station locations
2. GraphCast interpolated to station locations
3. Aurora interpolated to station locations
4. Persistence baseline
5. Per-station linear regression
6. Per-station Random Forest or XGBoost
7. Quantile mapping or monthly bias-correction baseline

The repo already has scripts prepared for GraphCast/Aurora evaluation:

- `scripts/export_station_manifest.py`
- `scripts/evaluate_gridded_baseline.py`

### Phase 4. Model work

Model ladder:

1. Existing baseline graph model
2. Existing multi-channel model
3. Simplified MPT variant
4. MPT ablation against plain message passing

Recommended feature groups:

- local meteorology: ERA5 temperature, humidity, pressure, wind
- static terrain: elevation and topographic context
- temporal context: lagged values, diurnal / seasonal encoding
- neighborhood context: nearby station states and gradients

Do not overbuild at first.

The first target is:

- a clean MPT-versus-baseline comparison under the same split

### Phase 5. Sparse-network and transfer experiments

This is where the paper becomes interesting.

Experiments:

1. Station withholding curve
   - use current withholding protocol
   - report degradation versus number of removed stations

2. Density-stratified evaluation
   - compare dense versus sparse subgraphs
   - report error by neighbor count and distance to nearest station

3. Cross-region transfer
   - pretrain on Netherlands
   - fine-tune on Nepal or another sparse target region
   - compare against training from scratch on the small target set

4. Regime-specific evaluation
   - nighttime vs daytime
   - winter vs monsoon
   - low-wind vs high-wind
   - inversion-like conditions vs non-inversion conditions

### Phase 6. Downstream application experiment

Choose one and keep it simple.

Recommended option:

- fog / frost proxy from corrected Tmin and humidity / wind conditions

Possible operationalization:

- define fog-risk days from visibility observations if available
- if not available, define a proxy rule using low Tmin, high RH, low wind, and stable nighttime conditions
- compare event detection using raw ERA5 versus corrected outputs

Avoid in the first paper:

- full PM2.5 forecasting
- multi-task joint air-quality modeling

Those are separate papers unless the data is already easy to obtain.

## 7. Experimental protocol

### Primary metrics

- MAE for Tmin
- MAE for Tmax
- RMSE for Tmin/Tmax
- bias
- station-wise median MAE

### Secondary metrics

- improvement over raw ERA5 on the same subset
- improvement over GraphCast and Aurora on the same subset
- performance under withheld-station settings
- regime-specific metrics for low-wind nighttime conditions

### Split policy

For source-region development:

- keep ST-LOBO as the strict main protocol
- keep final held-out test window separate

For Nepal fine-tuning:

- use strictly time-ordered splits
- if station count is low, add leave-one-station-out evaluation

### Statistical reporting

- confidence intervals by fold or bootstrap
- paired significance tests on matched station-date predictions
- error distributions, not only mean metrics

## 8. Recommended immediate task order

The next actions should be:

1. Run GraphCast and Aurora station-level evaluation on the Netherlands test period.
2. Build the external-baseline comparison table.
3. Finalize the literature matrix.
4. Decide on the primary Nepal application: fog/frost or inversion-sensitive Tmin correction.
5. Run a clean MPT-vs-message-passing ablation.
6. Reframe withholding as sparse-network robustness.
7. Only then start transfer-learning experiments.

## 9. Ten-week execution schedule

### Weeks 1-2

- finish literature review matrix
- finalize problem framing
- verify GraphCast/Aurora evaluation pipeline
- collect benchmark outputs on current Netherlands station set

### Weeks 3-4

- run classical baselines
- run clean MPT ablation
- produce first comparison table against raw ERA5, GraphCast, Aurora, and classical baselines

### Weeks 5-6

- run sparse-network withholding analysis
- add density-stratified diagnostics
- identify where gains are largest: Tmin, nighttime, winter, low-wind

### Weeks 7-8

- obtain Nepal or Himalayan target data
- harmonize metadata and covariates
- run small-data fine-tuning or transfer experiments

### Weeks 9-10

- run downstream fog/frost proxy analysis
- prepare figures and writing skeleton
- decide whether air-quality extension is worth keeping as future work only

## 10. Expected figures for the paper

1. Study design diagram: coarse model to station correction
2. Map of station network and terrain
3. Main benchmark table: ERA5 vs GraphCast vs Aurora vs classical baselines vs MPT
4. Station-wise error map
5. Sparse-network degradation curve
6. Regime-specific bar chart: nighttime / low-wind / winter
7. Optional downstream event-detection comparison for fog/frost proxy

## 11. Risks and mitigation

### Risk 1. Nepal station data is hard to obtain

Mitigation:

- use the Netherlands as the source benchmark system
- use publicly available Himalayan AWS studies for limited transfer tests
- state Nepal as the motivating deployment case if full data access lags

### Risk 2. GraphCast / Aurora do not lose by much on Tmin/Tmax

Mitigation:

- evaluate specifically in regimes where coarse models should fail most
- focus on station sparsity and inversion-sensitive cases
- add classical baselines so the paper is not only "our model vs two giant models"

### Risk 3. The current multi-channel model is not better than baseline

Mitigation:

- simplify the story
- prioritize the best-performing lightweight model
- present MPT only if it adds measurable value under strict evaluation

### Risk 4. Evapotranspiration or extra variables create complexity without gain

Mitigation:

- treat them as optional ablations
- keep the main model on variables already supported by the current pipeline

## 12. Success criteria

The project is strong enough for a paper if you can show all of the following:

1. Clear station-level improvement over raw ERA5.
2. Competitive or better performance than GraphCast and Aurora at the same stations.
3. Robustness when stations are withheld.
4. A credible Nepal-specific motivation tied to terrain, sparsity, and one concrete impact area.
5. A lightweight model size / runtime story that makes deployment realistic.

## 13. Bottom-line recommendation

Do not pitch this as a generic downscaling paper for Nepal.

Pitch it as:

`lightweight station-level correction of coarse global AI weather products for sparse Himalayan networks, with inversion-sensitive Tmin errors as the scientifically grounded entry point and fog/frost risk as the applied demonstration.`
