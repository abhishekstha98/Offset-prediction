# Experiment Suite and Results

This document explains the experiment suite currently defined in the repository, summarizes the completed runs found in `outputs/`, and interprets the results in one place.

The numbers below are taken from the current log files in `outputs/` at the time this report was generated. Where a run is incomplete, that is stated explicitly rather than treated as a final result.

## 1. What the project is evaluating

The project trains graph-based offset-prediction models over a 23-station network using ERA5 and station metadata. The target is the temperature offset relative to ERA5 for both Tmax and Tmin.

At a high level:

- Data source used by the training scripts: `merged.csv`
- Graph structure: 23 nodes, 69 edges, `k=3`
- Train/validation years used by the current config: `2020, 2021, 2022, 2023`
- Final held-out test split starts at `test_year=2024`, so the test set is `2024-2025`
- Maximum epochs: `1000`
- Early stopping patience: `10`
- Loss monitors both Tmax and Tmin

The relevant configuration comes from `src/config.py`, `src/train.py`, and `src/station_withholding_test.py`.

## 2. Cross-validation and evaluation protocols

### 2.1 Random CV

Stations are assigned randomly into 4 balanced blocks. One block is used for validation and the remaining blocks are used for training. This is the least spatially strict protocol.

### 2.2 SLOBO

SLOBO means Spatial Leave-One-Block-Out. Stations are clustered geographically into 4 K-means spatial blocks. Each fold withholds one spatial block for validation while training on the other three blocks.

Important detail: the withheld validation stations remain in the graph for message passing, but they do not contribute to the training loss.

### 2.3 ST-LOBO

ST-LOBO adds a temporal constraint on top of the SLOBO spatial split.

- Spatial blocks: the same 4 SLOBO K-means blocks
- Temporal windows: 2 windows
  - Window 0: `2020-2021`
  - Window 1: `2022-2023`

This produces `4 x 2 = 8` folds. Validation points are only those nodes that are both:

- in the withheld spatial block
- in the withheld temporal window

After cross-validation, the best validation-selected ST-LOBO checkpoint is evaluated on the final held-out test set.

### 2.4 Station withholding

The withholding study is a spatial generalization stress test.

Protocol:

1. Train on a subset of stations.
2. Hold out `m` entire stations.
3. Measure seen-station MAE and unseen-station MAE.
4. Repeat for multiple values of `m`.

Current `m` schedule:

- `m = 0`
- `m = 1`
- `m = 3`
- `m = 5`
- `m = 10`
- `m = 15`

Iteration counts:

- `m = 0`: 1 run
- `m = 1`: leave-one-out over all 23 stations
- `m >= 3`: Monte Carlo repeated runs, capped at 10 iterations

## 3. Experiment inventory

The experiment runner defines three groups: baseline, multi-channel, and ablation.

### 3.1 Baseline experiments

| Key | Purpose | Command shape | Log |
|---|---|---|---|
| `baseline_random` | Baseline model under random CV | `python run_all_experiments.py --experiment baseline_random` | `outputs/baseline_random.log` |
| `baseline_slobo` | Baseline model under SLOBO | `python run_all_experiments.py --experiment baseline_slobo` | `outputs/baseline_slobo.log` |
| `baseline_stlobo` | Baseline model under ST-LOBO | `python run_all_experiments.py --experiment baseline_stlobo` | `outputs/baseline_stlobo.log` |
| `baseline_withholding` | Baseline model under station withholding | `python run_all_experiments.py --experiment baseline_withholding` | `outputs/baseline_withholding.log` |

### 3.2 Multi-channel experiments

| Key | Purpose | Active channels | Log |
|---|---|---|---|
| `mc_slobo` | Full multi-channel model under SLOBO | `all` | `outputs/mc_slobo.log` |
| `mc_stlobo` | Full multi-channel model under ST-LOBO | `all` | `outputs/mc_stlobo.log` |
| `mc_withholding` | Full multi-channel model under station withholding | `all` | `outputs/mc_withholding.log` |

### 3.3 Channel ablations

These ablations are all run under SLOBO.

| Key | Channel removed | Active channels | Log |
|---|---|---|---|
| `ablate_terrain` | Terrain | `temperature,pressure` | `outputs/mc_ablate_terrain.log` |
| `ablate_pressure` | Pressure | `temperature,terrain` | `outputs/mc_ablate_pressure.log` |
| `ablate_temperature` | Temperature | `pressure,terrain` | `outputs/mc_ablate_temperature.log` |

## 4. Completed result summary

The table below summarizes the completed experiments with their median cross-validation performance.

Lower MAE is better.

| Experiment | Val MAE Tmax | Val MAE Tmin | ERA5 Baseline Tmax | ERA5 Baseline Tmin | Extra result |
|---|---:|---:|---:|---:|---|
| Baseline Random | `1.7404 ± 0.0548` | `1.8497 ± 0.1006` | `2.0780 ± 0.0671` | `1.9662 ± 0.1710` | None |
| Baseline SLOBO | `1.7357 ± 0.0541` | `1.9018 ± 0.1572` | `2.0938 ± 0.1054` | `2.1064 ± 0.2768` | None |
| Baseline ST-LOBO | `1.6688 ± 0.1076` | `1.8009 ± 0.2005` | `2.1045 ± 0.1078` | `2.0980 ± 0.2787` | Test MAE Tmax `1.7848` |
| Multi-channel SLOBO | `1.7582 ± 0.0670` | `1.8707 ± 0.1507` | `2.0938 ± 0.1054` | `2.1064 ± 0.2768` | None |
| Multi-channel ST-LOBO | `1.7126 ± 0.0741` | `1.8125 ± 0.1652` | `2.1045 ± 0.1078` | `2.0980 ± 0.2787` | Test MAE Tmax `1.7968` |
| Ablation: no terrain | `1.7687 ± 0.0777` | `1.8926 ± 0.1504` | `2.0938 ± 0.1054` | `2.1064 ± 0.2768` | SLOBO only |
| Ablation: no pressure | `1.7795 ± 0.0882` | `1.9123 ± 0.1710` | `2.0938 ± 0.1054` | `2.1064 ± 0.2768` | SLOBO only |
| Ablation: no temperature | `1.7668 ± 0.0554` | `1.8823 ± 0.1651` | `2.0938 ± 0.1054` | `2.1064 ± 0.2768` | SLOBO only |

## 5. Main findings

### 5.1 Best overall completed validation result

The best completed validation result in the current logs is:

- `Baseline ST-LOBO`
- Val MAE Tmax: `1.6688`
- Val MAE Tmin: `1.8009`

This is also the best completed test result currently available:

- Test MAE Tmax: `1.7848`

### 5.2 The models consistently beat the ERA5 baseline

Every completed learned model outperforms the ERA5 baseline on Tmax.

Representative improvements in Val MAE Tmax relative to the baseline:

- Baseline Random: `2.0780 - 1.7404 = 0.3376 C`
- Baseline SLOBO: `2.0938 - 1.7357 = 0.3581 C`
- Baseline ST-LOBO: `2.1045 - 1.6688 = 0.4357 C`
- Multi-channel ST-LOBO: `2.1045 - 1.7126 = 0.3919 C`

So the project is learning a useful correction signal beyond raw ERA5.

### 5.3 The full multi-channel model does not beat the baseline model in the completed runs

This is the most important result in the current experiment set.

Relative to the baseline model:

- On SLOBO, multi-channel is worse on Tmax by `+0.0225 C`
  - Baseline SLOBO Tmax: `1.7357`
  - Multi-channel SLOBO Tmax: `1.7582`

- On ST-LOBO, multi-channel is worse on Tmax by `+0.0438 C`
  - Baseline ST-LOBO Tmax: `1.6688`
  - Multi-channel ST-LOBO Tmax: `1.7126`

- On ST-LOBO test Tmax, multi-channel is worse by `+0.0120 C`
  - Baseline ST-LOBO test Tmax: `1.7848`
  - Multi-channel ST-LOBO test Tmax: `1.7968`

There is one partial counterpoint:

- On SLOBO Tmin, multi-channel is slightly better than baseline
  - Baseline SLOBO Tmin: `1.9018`
  - Multi-channel SLOBO Tmin: `1.8707`
  - Delta: `-0.0311 C`

But the main headline remains that the full multi-channel variant has not yet produced a better overall Tmax result than the baseline in the completed evaluations.

### 5.4 Ablation results suggest no single removed channel improves the full multi-channel model

Compared with full multi-channel SLOBO (`1.7582` Tmax):

- No terrain: `1.7687` Tmax, worse by `+0.0105 C`
- No pressure: `1.7795` Tmax, worse by `+0.0213 C`
- No temperature: `1.7668` Tmax, worse by `+0.0086 C`

Among the ablations:

- Best ablation on Tmax: `no temperature` at `1.7668`
- Worst ablation on Tmax: `no pressure` at `1.7795`

Interpretation:

- Pressure appears to be the most damaging channel to remove in this setup.
- None of the ablations produces a better Tmax score than the full multi-channel SLOBO model.
- Even the best ablation remains worse than the baseline SLOBO model.

## 6. Detailed experiment-by-experiment interpretation

### 6.1 Baseline Random CV

Log: `outputs/baseline_random.log`

Purpose:

- Establish a lower-rigor baseline where station splits are random rather than geographically structured.

Results:

- Val MAE Tmax: `1.7404 ± 0.0548`
- Val MAE Tmin: `1.8497 ± 0.1006`

Interpretation:

- This run performs well, but it is the least spatially strict evaluation.
- It is useful for sanity checking optimization and model capacity, but it is less convincing as a spatial generalization benchmark than SLOBO or ST-LOBO.

### 6.2 Baseline SLOBO

Log: `outputs/baseline_slobo.log`

Purpose:

- Evaluate the baseline model under purely spatial block holdout.

Results:

- Val MAE Tmax: `1.7357 ± 0.0541`
- Val MAE Tmin: `1.9018 ± 0.1572`

Interpretation:

- This is a stronger baseline than random CV because it forces validation on unseen geographic blocks.
- It remains better than the full multi-channel SLOBO run on Tmax.

### 6.3 Baseline ST-LOBO

Log: `outputs/baseline_stlobo.log`

Purpose:

- Evaluate the baseline model under the strictest completed protocol in the repo: spatial plus temporal holdout.

Results:

- Val MAE Tmax: `1.6688 ± 0.1076`
- Val MAE Tmin: `1.8009 ± 0.2005`
- Final held-out test MAE Tmax: `1.7848`

Interpretation:

- This is the strongest completed result in the current experiment set.
- It suggests the baseline architecture generalizes better than the current multi-channel version under the most demanding evaluation scheme.

### 6.4 Baseline station withholding

Log: `outputs/baseline_withholding.log`

Purpose:

- Measure how performance degrades as entire stations are removed from training.

Final degradation curve from the log:

| Withheld stations `m` | Seen MAE Tmax | Unseen MAE Tmax | Baseline Tmax |
|---|---:|---:|---:|
| 0 | `1.6528` | `nan` | `nan` |
| 1 | `1.6875` | `1.7086` | `2.0890` |
| 3 | `1.6805` | `1.7111` | `2.0796` |
| 5 | `1.7040` | `1.7262` | `2.0489` |
| 10 | `1.6803` | `1.7515` | `2.1017` |
| 15 | `1.6864` | `1.7575` | `2.0957` |

Key pattern:

- Unseen-station MAE rises as more stations are withheld.
- Degradation is small at `m=1` and becomes more noticeable by `m=10` and `m=15`.

Interpretation:

- The baseline model retains reasonable robustness even as spatial support is reduced.
- The degradation trend is real, but not catastrophic, which is encouraging for spatial transfer.

### 6.5 Multi-channel SLOBO

Log: `outputs/mc_slobo.log`

Purpose:

- Evaluate the full multi-channel model using all channels under spatial block holdout.

Results:

- Val MAE Tmax: `1.7582 ± 0.0670`
- Val MAE Tmin: `1.8707 ± 0.1507`

Interpretation:

- This run improves Tmin relative to baseline SLOBO.
- It does not improve Tmax, which is the more prominent comparison target in the logs and plots.
- In the current state, the multi-channel model is not yet a clear upgrade over baseline.

### 6.6 Multi-channel ST-LOBO

Log: `outputs/mc_stlobo.log`

Purpose:

- Evaluate the full multi-channel model under the strict spatial-temporal split.

Results:

- Val MAE Tmax: `1.7126 ± 0.0741`
- Val MAE Tmin: `1.8125 ± 0.1652`
- Final held-out test MAE Tmax: `1.7968`

Interpretation:

- This is a solid result and still clearly better than the ERA5 baseline.
- However, it remains behind the baseline ST-LOBO model on both validation Tmax and test Tmax.

### 6.7 Multi-channel withholding

Log: `outputs/mc_withholding.log`

Status:

- Incomplete
- Not suitable for final comparison yet

What completed before the run stopped:

- `m=0` finished
- Aggregated seen MAE Tmax at `m=0`: `1.6319`

What had started but did not finish:

- `m=1`
- Iteration `1/23`
- Progress reached at least epoch 20, with monitor MAE `1.5293`

Interpretation:

- The completed `m=0` number is promising and is lower than the baseline withholding `m=0` seen MAE (`1.6528`).
- But the run did not finish the actual unseen-station evaluations, so no conclusion should be drawn yet about whether the multi-channel model is better under withholding.

### 6.8 SLOBO channel ablations

Logs:

- `outputs/mc_ablate_terrain.log`
- `outputs/mc_ablate_pressure.log`
- `outputs/mc_ablate_temperature.log`

Purpose:

- Test whether any one channel is hurting the full multi-channel SLOBO setup.

Results:

| Variant | Val MAE Tmax | Val MAE Tmin | Reading |
|---|---:|---:|---|
| Full multi-channel | `1.7582` | `1.8707` | Best Tmax among multi-channel SLOBO variants |
| Remove terrain | `1.7687` | `1.8926` | Slightly worse |
| Remove pressure | `1.7795` | `1.9123` | Worst ablation |
| Remove temperature | `1.7668` | `1.8823` | Best ablation, but still worse than full multi-channel |

Interpretation:

- The full multi-channel configuration is already the strongest of the tested multi-channel SLOBO variants.
- Pressure seems to matter most among the three channels in this ablation set.
- No ablation closes the gap to the baseline SLOBO model.

## 7. Recommended reading of the current evidence

If the current goal is to identify the strongest model from completed runs, the evidence supports:

1. `baseline_stlobo` as the strongest completed overall result
2. `baseline_slobo` as the strongest completed purely spatial result
3. `mc_slobo` and `mc_stlobo` as useful experiments, but not yet superior to the baseline

If the current goal is to justify the multi-channel architecture, the current logs do not yet support that claim strongly. The architecture still beats the raw ERA5 baseline, but it does not outperform the simpler baseline neural model in the completed comparisons.

## 8. Available plots

Plots have already been generated in `outputs/plots/` from the current logs:

- `01_random_ablation.png`
- `02_slobo_ablation.png`
- `03_stlobo_ablation.png`
- `05_withholding_training_curves.png`
- `06_withholding_degradation.png`
- `07_cv_method_comparison.png`
- `08_best_fold_curves.png`
- `09_all_folds_overlay.png`
- `10_best_overall_bar.png`

These are the fastest visual way to inspect fold dynamics and method comparisons.

## 9. Caveats

- `mc_withholding` is incomplete, so the withholding comparison is currently asymmetric.
- The report reflects the logs currently present in `outputs/`, not every historical run ever performed.
- Since the current config uses only the most recent 4 pre-test years, these results are specifically tied to the `2020-2023` train/validation subset.

## 10. Bottom line

The completed experiments show a consistent and useful learned correction over ERA5, but they do not yet show a compelling advantage for the multi-channel architecture over the simpler baseline model.

The strongest completed result in the current repository state is the baseline model under ST-LOBO.
