# LLM Research Handoff

Date: 2026-03-27  
Repo: `Offset-prediction`  
Branch: `main`

This file is designed to be pasted into a fresh Claude/ChatGPT/Codex session so another model can continue the research without rebuilding context from scratch.

## 1. Strategic Research Handoff

### Who is involved

- Researcher: based in Birgunj, Nepal; ML background; no agricultural domain expertise yet
- Mentor / co-author: Alicia Lupidi, PhD researcher at Oxford University, also works at Meta
- Target venue: NeurIPS 2026, with flexibility on venue and roughly 12 months of runway

### What already exists

This repo is a working research codebase for station-level weather correction.

Current data context:

- 23 Netherlands KNMI stations
- consistent station network across multiple decades
- stations are mostly lowland / flat, so there is not much elevation diversity
- ERA5 is used as input

Current model context:

- graph neural network over a 23-node, 69-edge station graph with `k=3`
- predicts Tmin / Tmax correction offsets over ERA5
- includes baseline and multi-channel variants
- includes channel ablations over terrain, pressure, and temperature

Completed experiments:

- random CV
- SLOBO
- ST-LOBO
- channel ablation study
- baseline withholding study
- multi-channel withholding study is incomplete

Best completed result so far:

- `baseline_stlobo`
- validation MAE Tmax: `1.6688 ± 0.1076`
- test MAE Tmax: `1.7848`

Important current empirical result:

- the full multi-channel model does not yet beat the simpler baseline model on the completed evaluations

### Alicia's key feedback

- the paper cannot remain architecture-first
- it needs a clear problem statement
- it needs external baselines, not just ERA5 and internal ablations
- GraphCast and Aurora are the obvious external baselines to compare against
- the value proposition should be: smaller, faster, and better for a specific niche
- the work should target an underexplored niche rather than a saturated broad forecasting task
- transfer learning across regions is a promising story if framed carefully

### Core technical gap being exploited

GraphCast and Aurora operate at approximately `0.25°` resolution, roughly `28 km` per grid cell. That is structurally too coarse for:

- station-level decision making
- sub-grid phenomena like frost pockets, inversions, and local cold pooling
- sparse-network countries that cannot rely on dense correction infrastructure

This spatial-resolution mismatch is the main technical motivation for the paper.

### Current proposed framing

Problem:

- GraphCast and Aurora are strong large-scale forecasters but too coarse for station-level correction
- classical post-processing baselines are spatially weak
- data-scarce countries often have no strong correction layer beyond raw NWP output

Method:

- a Message Passing Transformer-style graph model for station-level Tmin / Tmax correction
- designed to be lightweight, topology-agnostic, and deployable on sparse station networks

Planned baselines to beat:

- raw ERA5
- interpolated GraphCast output at station locations
- interpolated Aurora output at station locations
- classical statistical post-processing like quantile mapping / delta correction

Core experimental contributions being targeted:

1. station-level offset correction that beats raw ERA5 and external baselines
2. ablations isolating attention vs standard message passing
3. robustness-under-sparsity evidence via the withholding curve
4. few-shot transfer / fine-tuning protocol for sparse target networks

Deployment niche:

- sparse meteorological networks in developing countries
- Nepal is the motivating deployment case
- the Netherlands is the current validation ground

Transfer learning story:

- what should transfer is not local geography itself, but the physics of near-surface bias correction and graph-based spatial reasoning
- pretraining on the Netherlands and fine-tuning on a sparse target network is the intended story
- current withholding experiments are already a proxy for sparse-target adaptation

### Current paper shape

Working title:

`Lightweight Graph-Based NWP Correction for Data-Scarce Meteorological Networks`

Current contribution set:

1. a Message Passing Transformer GNN for station-level ERA5 Tmin / Tmax correction
2. a benchmark against GraphCast / Aurora at station level
3. robustness-under-sparsity analysis via withholding
4. a few-shot fine-tuning protocol for adaptation to sparse networks

Optional strengthening contribution:

- one concrete downstream application, such as frost risk, cold-wave warning, fog, or agricultural decision support

### Major unresolved questions

- what exact niche application will anchor the introduction
- whether Nepal data becomes available
- what the attention mechanism should attend over:
  - neighboring stations
  - time steps
  - pressure levels

### Priority next steps

1. Run GraphCast and/or Aurora on the 23 Netherlands stations and compute station-level Tmin / Tmax MAE.
2. Formalize the MPT architecture and run a direct ablation against standard message passing.
3. Reframe the withholding study as robustness-under-sparsity rather than just a diagnostic.
4. Choose one concrete downstream application niche for the paper framing.
5. Start outreach for Nepal or similar sparse-network data.

### Bottom line for the next research session

The identity of the work is now:

`A lightweight Message Passing Transformer trained to correct ERA5 Tmin/Tmax at station level, intended for data-scarce meteorological networks, with the claim that it can beat raw ERA5 and external coarse-grid baselines while remaining robust under sparse network conditions.`

The single most important next action is:

- run GraphCast / Aurora on the Netherlands stations and record the resulting station-level MAE

## 2. Repo-Local Operational State

### Current branch / recent history

Recent commits:

- `256c98a` `Track experiment output logs`
- `cb7c85a` `logs`
- `0240c63` `Update experiment runner and training pipeline`
- `867130b` `multi channel attention implementation`
- `2aebe27` `Fix progress log capture in experiment runner`

### Current uncommitted files

At the time this file was updated, the worktree contains local changes:

- modified: `plot_results.py`
- untracked: `EXPERIMENTS_AND_RESULTS.md`
- untracked: `outputs/plots/`
- untracked: `LLM_HANDOFF.md`

If a future agent is asked to commit / push, it should review and include these intentionally rather than treating them as accidental noise.

## 3. Current Experiment Inventory

The main experiment orchestrator is:

- `run_all_experiments.py`

Experiment groups:

- `baseline`
- `multi_channel`
- `ablation`

Important single experiment keys:

- `baseline_random`
- `baseline_slobo`
- `baseline_stlobo`
- `baseline_withholding`
- `mc_slobo`
- `mc_stlobo`
- `mc_withholding`
- `ablate_terrain`
- `ablate_pressure`
- `ablate_temperature`

## 4. Current Configuration Facts

From the current repo config and logs:

- data file: `merged.csv`
- graph: 23 nodes, 69 edges, `k=3`
- train/validation years currently used: `2020-2023`
- final held-out test split begins at `2024`
- ST-LOBO windows:
  - `2020-2021`
  - `2022-2023`
- spatial blocks: `4`
- max epochs: `1000`
- early stopping patience: `10`

Relevant code files:

- `src/config.py`
- `src/train.py`
- `src/data/split.py`
- `src/station_withholding_test.py`
- `plot_results.py`

## 5. Best Completed Results So Far

| Experiment | Val MAE Tmax | Val MAE Tmin | Extra |
|---|---:|---:|---|
| `baseline_random` | `1.7404 ± 0.0548` | `1.8497 ± 0.1006` | none |
| `baseline_slobo` | `1.7357 ± 0.0541` | `1.9018 ± 0.1572` | none |
| `baseline_stlobo` | `1.6688 ± 0.1076` | `1.8009 ± 0.2005` | test Tmax `1.7848` |
| `mc_slobo` | `1.7582 ± 0.0670` | `1.8707 ± 0.1507` | none |
| `mc_stlobo` | `1.7126 ± 0.0741` | `1.8125 ± 0.1652` | test Tmax `1.7968` |
| `mc_ablate_terrain` | `1.7687 ± 0.0777` | `1.8926 ± 0.1504` | SLOBO only |
| `mc_ablate_pressure` | `1.7795 ± 0.0882` | `1.9123 ± 0.1710` | SLOBO only |
| `mc_ablate_temperature` | `1.7668 ± 0.0554` | `1.8823 ± 0.1651` | SLOBO only |

Interpretation:

- all learned models beat raw ERA5
- the baseline model currently beats the full multi-channel model on the completed Tmax comparisons
- among multi-channel SLOBO variants, the full multi-channel model still beats all three single-channel ablations

Important deltas:

- `mc_slobo` vs `baseline_slobo` on Tmax: `+0.0225 C` worse
- `mc_stlobo` vs `baseline_stlobo` on Tmax: `+0.0438 C` worse
- `mc_stlobo` vs `baseline_stlobo` on test Tmax: `+0.0120 C` worse

## 6. Withholding Study Status

### Baseline withholding

`baseline_withholding.log` completed successfully.

Final curve from the log:

| m withheld | Seen MAE Tmax | Unseen MAE Tmax | Baseline Tmax |
|---|---:|---:|---:|
| 0 | `1.6528` | `nan` | `nan` |
| 1 | `1.6875` | `1.7086` | `2.0890` |
| 3 | `1.6805` | `1.7111` | `2.0796` |
| 5 | `1.7040` | `1.7262` | `2.0489` |
| 10 | `1.6803` | `1.7515` | `2.1017` |
| 15 | `1.6864` | `1.7575` | `2.0957` |

### Multi-channel withholding

`mc_withholding.log` is incomplete.

What completed:

- `m=0` finished
- aggregated seen MAE Tmax at `m=0`: `1.6319`

Where it stopped:

- started `m=1`
- started iteration `1/23`
- reached at least epoch 20
- latest printed monitor MAE: `1.5293`

Important implementation caveat:

- `src/station_withholding_test.py` does not currently support mid-run resume
- it saves withholding checkpoints, but does not load them to skip completed iterations
- rerunning starts the study from the beginning

## 7. Existing Documentation and Plots

Detailed experiment report:

- `EXPERIMENTS_AND_RESULTS.md`

Generated plots currently available in `outputs/plots/`:

- `01_random_ablation.png`
- `02_slobo_ablation.png`
- `03_stlobo_ablation.png`
- `05_withholding_training_curves.png`
- `06_withholding_degradation.png`
- `07_cv_method_comparison.png`
- `08_best_fold_curves.png`
- `09_all_folds_overlay.png`
- `10_best_overall_bar.png`
- `11_ablate_terrain.png`
- `12_ablate_pressure.png`
- `13_ablate_temperature.png`
- `14_channel_ablation_comparison.png`

## 8. Recommended Files for a New LLM to Read First

1. `LLM_HANDOFF.md`
2. `EXPERIMENTS_AND_RESULTS.md`
3. `run_all_experiments.py`
4. `src/train.py`
5. `src/station_withholding_test.py`
6. `plot_results.py`
7. `outputs/baseline_stlobo.log`
8. `outputs/mc_stlobo.log`
9. `outputs/baseline_slobo.log`
10. `outputs/mc_slobo.log`

## 9. Useful Commands

Run one experiment:

```bash
python run_all_experiments.py --experiment baseline_slobo
python run_all_experiments.py --experiment baseline_stlobo
python run_all_experiments.py --experiment mc_slobo
python run_all_experiments.py --experiment mc_stlobo
python run_all_experiments.py --experiment ablate_terrain
python run_all_experiments.py --experiment ablate_pressure
python run_all_experiments.py --experiment ablate_temperature
```

Run withholding directly:

```bash
python src/station_withholding_test.py --model_type multi_channel --active_channels all
```

Regenerate plots:

```bash
python plot_results.py
```

## 10. Suggested Prompt for a New Session

```text
Please read LLM_HANDOFF.md and EXPERIMENTS_AND_RESULTS.md first, then inspect the current git status before making changes.

Important current state:
- baseline_stlobo is the best completed result
- the full multi-channel model beats ERA5 but not the baseline model
- mc_withholding is incomplete and not resumable yet
- plot_results.py has local updates for current logs and ablation plots
- there are uncommitted docs / plots that may need to be committed

My next goal is: [replace with the exact next objective].
```

## 11. Bottom Line

The repo has a clear current research state:

- the learning setup works
- the baseline model is strong
- the multi-channel story is not yet empirically winning
- the biggest missing external comparison is GraphCast / Aurora
- the biggest unfinished engineering item is resumable multi-channel withholding

If another LLM starts from this file, it should treat running GraphCast / Aurora station-level baselines as the top-priority next research action.
