# Team Workflow Guideline for Nepal-Focused Research Direction

Date: 2026-04-16

## 1. Project goal

Our working research direction is:

`Build a lightweight graph-based offset correction model that improves station-level Tmin/Tmax from coarse global products such as ERA5, GraphCast, and Aurora, with Nepal as the motivating deployment case.`

This means:

- we are not trying to build a full weather model from scratch
- we are not trying to solve all Nepal weather problems at once
- we are trying to show that a small model can correct coarse products at station level
- we want the method to remain useful even when the station network is sparse

## 2. Core paper story

The team should align on this story before starting more experiments:

1. Global AI weather models are strong, but too coarse for station-level use in complex terrain.
2. A lightweight graph model can learn station-level corrections from historical observations.
3. The gains should be strongest for Tmin, nighttime, low-wind, and inversion-sensitive settings.
4. Nepal is the motivating deployment case because terrain and station sparsity make this problem especially relevant.
5. The first applied demonstration should be fog/frost risk, not full air-quality forecasting.

## 3. What we should lock now

Before expanding the scope, the team should confirm these decisions:

- Primary targets: daily `Tmin` and `Tmax`
- Main claim: station-level correction of coarse products
- Main benchmark products: `ERA5`, `GraphCast`, `Aurora`
- Main application story: `fog/frost proxy`
- Air quality: keep as future work unless data becomes easy to integrate
- Evapotranspiration: do not include in the first experiment round

## 4. Recommended workflow

### Phase 1. Finalize scope and benchmark definition

Objective:

- make sure everyone is aligned on the same problem statement and evaluation goal

Steps:

1. Agree on the one-sentence project framing.
2. Confirm that the first paper focuses on `Tmin/Tmax correction`, not generic downscaling.
3. Confirm that `fog/frost proxy` is the first Nepal-facing impact story.
4. Freeze the benchmark list so experiments do not keep shifting.

Deliverables:

- final one-sentence problem statement
- final benchmark list
- final evaluation split policy
- short written scope statement for the team

### Phase 2. External baseline preparation

Objective:

- answer the most important open question: how our model compares against strong coarse external models

Steps:

1. Export the station manifest for the Netherlands station set.
2. Run `Aurora` on the held-out Netherlands test period first.
3. Interpolate predictions to station locations.
4. Aggregate to daily `Tmin/Tmax` if needed.
5. Compute station-level MAE and bias.
6. Repeat for `GraphCast` after Aurora if setup time is reasonable.

Deliverables:

- `Aurora` station-level evaluation
- `GraphCast` station-level evaluation if available
- summary table: `ERA5 vs Aurora vs GraphCast vs current best model`

Why this phase matters:

- this is the strongest missing comparison requested in the discussion
- without this, the paper story is still incomplete

### Phase 3. Classical baseline comparison

Objective:

- make sure the project is not only compared against giant foundation models

Steps:

1. Run raw `ERA5` or `ERA5-Land` baseline at station level.
2. Add persistence baseline.
3. Add per-station linear regression.
4. Add per-station `Random Forest` or `XGBoost`.
5. Add a simple bias-correction baseline such as quantile mapping or monthly delta correction.

Deliverables:

- benchmark matrix with all external and classical baselines
- first comparison table for the paper

Why this phase matters:

- if the graph model beats only ERA5 but not simple baselines, the paper story weakens
- if it beats both classical and external baselines, the claim becomes much stronger

### Phase 4. Model simplification and ablation

Objective:

- identify the strongest lightweight model rather than assuming the more complex version is better

Steps:

1. Use the current best strict protocol as the main evaluation setup.
2. Compare the baseline graph model against the multi-channel model.
3. Run the planned `MPT vs plain message passing` comparison.
4. Keep the simplest model that gives the strongest and most stable result.

Deliverables:

- ablation table
- decision on the main model for the paper

Why this phase matters:

- current evidence suggests the simpler baseline is still the strongest completed model
- the team should not over-commit to a more complex architecture without clear gains

### Phase 5. Sparse-network robustness analysis

Objective:

- turn the station withholding idea into one of the central scientific contributions

Steps:

1. Reframe station withholding as sparse-network robustness.
2. Run the withholding curve cleanly for the chosen main model.
3. Report seen-station and unseen-station error as station density decreases.
4. Add diagnostics such as neighbor count and nearest-station distance if possible.

Deliverables:

- sparse-network degradation curve
- robustness summary table
- interpretation of where performance starts to break down

Why this phase matters:

- this is a strong differentiator for Nepal and other sparse-network regions

### Phase 6. Regime-specific analysis

Objective:

- show that improvements are not uniform, and are most relevant in scientifically meaningful regimes

Steps:

1. Evaluate `nighttime vs daytime`.
2. Evaluate `winter vs monsoon` or relevant seasonal groups.
3. Evaluate `low-wind vs high-wind`.
4. Evaluate inversion-like or cold-pooling-sensitive conditions if proxies are available.

Deliverables:

- regime-specific metrics table
- figure showing where the model helps most

Why this phase matters:

- it connects the technical model result to the Nepal motivation
- it strengthens the inversion-sensitive Tmin correction narrative

### Phase 7. Nepal transfer or fine-tuning stage

Objective:

- move from a strong source-region benchmark to the Nepal deployment case

Steps:

1. Obtain Nepal or Himalayan station data if available.
2. Harmonize metadata, coordinates, timestamps, and covariates.
3. Fine-tune the best source-region model on the target data.
4. Compare transfer learning against training from scratch on the small target dataset.

Deliverables:

- Nepal-target experiment or transfer-learning pilot
- comparison of pretrained vs scratch models

Why this phase matters:

- this is what makes the paper truly Nepal-facing rather than only Nepal-motivated

### Phase 8. Downstream application demo

Objective:

- demonstrate that better temperature correction improves a simple operationally meaningful task

Steps:

1. Define a fog/frost risk rule using corrected `Tmin` and available humidity/wind conditions.
2. If visibility observations are available, evaluate event detection directly.
3. If not, use a simple proxy and compare raw ERA5 against corrected predictions.

Deliverables:

- fog/frost proxy experiment
- simple event-detection or risk-comparison table

Why this phase matters:

- it makes the work easier to communicate beyond model MAE
- it keeps the application practical without turning the paper into a full air-quality study

## 5. Recommended order of execution

The team should work in this order:

1. Lock the scope and final framing.
2. Run `Aurora` benchmark on the Netherlands held-out period.
3. Add `GraphCast` benchmark if feasible.
4. Build the external and classical baseline table.
5. Run model ablations and choose the main lightweight model.
6. Run sparse-network robustness experiments.
7. Run regime-specific analysis.
8. Start Nepal transfer-learning or fine-tuning experiments.
9. Run the fog/frost proxy demonstration.

## 6. Suggested team split

The work can be split across the team as follows:

- Person 1: literature matrix and benchmark paper comparison
- Person 2: Aurora and GraphCast pipeline
- Person 3: classical baselines and benchmark table
- Person 4: model ablations and sparse-network analysis
- Person 5: Nepal data sourcing and transfer-learning preparation
- Shared task: final framing, interpretation, and writing

## 7. Weekly check-in structure

Each weekly check-in should answer these questions:

1. What benchmark results were added this week?
2. Did the current best model improve or stay the same?
3. Are we still aligned on the same paper story?
4. What is the biggest technical blocker right now?
5. Does any new idea belong in the main paper, or should it be deferred?

## 8. What to avoid

The team should avoid these common mistakes:

- changing the research question every week
- adding too many extra variables before baseline comparisons are complete
- turning the first paper into a full Nepal forecasting system
- jumping into air-quality modeling too early
- assuming the most complex architecture is automatically the best one
- running many experiments without a fixed benchmark table

## 9. Minimum success criteria

The project is on a strong path if we can show:

1. clear improvement over raw `ERA5`
2. competitive or better station-level performance than `Aurora` and ideally `GraphCast`
3. robustness under reduced station density
4. a clear Nepal-specific motivation tied to terrain and sparse observations
5. one simple and convincing downstream demo such as fog/frost risk

## 10. Short version to present to the team

If you want to explain the workflow quickly in a meeting, use this:

`First we lock the scope. Then we benchmark against Aurora and GraphCast. Then we compare against simple classical baselines. After that we choose the strongest lightweight model, test how well it works when stations are sparse, and only then move into Nepal transfer-learning and a fog/frost application demo.`
