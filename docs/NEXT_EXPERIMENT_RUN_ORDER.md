# Next Experiment Run Order

This runbook is for manually rerunning the highest-priority experiments on the fog-ready dataset from the `Offset-prediction` repo root. It is ordered by research value, not by runtime. The goal is to refresh the main baseline vs multi-channel comparison on `datasets/fog_ready/era5_trainable_daily.csv` under the strictest protocols first.

## Current Research Status

- Best completed historical result: `baseline_stlobo`
- Main evaluation protocol: `ST-LOBO`
- Full multi-channel has not yet beaten the baseline in the completed runs
- Current objective: rerun the strongest comparisons on `datasets/fog_ready/era5_trainable_daily.csv`

## Recommended Run Order

1. `baseline_stlobo`
2. `mc_stlobo`
3. `baseline_slobo`
4. `mc_slobo`
5. optional `mc_withholding`

Run these from the repo root:

```bash
cd "/media/peridot/2TB1/Documents/Abhishek/offset prediction research/Offset-prediction"
```

## Fresh Reruns

### 1. Baseline ST-LOBO

```bash
python -u src/train.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --cv_mode st_lobo \
  --model_type baseline \
  --epochs 1000 \
  --no_resume \
  2>&1 | tee outputs/baseline_stlobo_fogready_$(date +%Y%m%d_%H%M%S).log
```

### 2. Multi-channel ST-LOBO

```bash
python -u src/train.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --cv_mode st_lobo \
  --model_type multi_channel \
  --active_channels all \
  --epochs 1000 \
  --no_resume \
  2>&1 | tee outputs/mc_stlobo_fogready_$(date +%Y%m%d_%H%M%S).log
```

### 3. Baseline SLOBO

```bash
python -u src/train.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --cv_mode slobo \
  --model_type baseline \
  --epochs 1000 \
  --no_resume \
  2>&1 | tee outputs/baseline_slobo_fogready_$(date +%Y%m%d_%H%M%S).log
```

### 4. Multi-channel SLOBO

```bash
python -u src/train.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --cv_mode slobo \
  --model_type multi_channel \
  --active_channels all \
  --epochs 1000 \
  --no_resume \
  2>&1 | tee outputs/mc_slobo_fogready_$(date +%Y%m%d_%H%M%S).log
```

### 5. Optional: Multi-channel Withholding

Run this after the main ST-LOBO/SLOBO refresh is complete.

```bash
python -u src/station_withholding_test.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --model_type multi_channel \
  --active_channels all \
  --epochs 1000 \
  --no_resume \
  2>&1 | tee outputs/mc_withholding_fogready_$(date +%Y%m%d_%H%M%S).log
```

## How To Confirm A Run Finished Correctly

Look for all of the following in the log:

- `RUN_STATUS: COMPLETED`
- a summary block with MAE metrics
- `Saved checkpoint`
- `Saved scaler`

For `ST-LOBO`, also expect:

- `Final Test Evaluation`

## Resume / Restart

### Resume the same run

Re-run the exact same command but remove `--no_resume`.

Example:

```bash
python -u src/train.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --cv_mode st_lobo \
  --model_type baseline \
  --epochs 1000 \
  2>&1 | tee outputs/baseline_stlobo_fogready_resume_$(date +%Y%m%d_%H%M%S).log
```

### Inspect saved run state

```bash
ls -lt outputs/train_state/*.status.json
```

```bash
cat "$(ls -t outputs/train_state/*.status.json | head -n1)"
```

For withholding runs:

```bash
ls -lt outputs/withholding_state/
```

### Force a fresh restart

Use the same command with `--no_resume`.

Generic training example:

```bash
python -u src/train.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --cv_mode slobo \
  --model_type baseline \
  --epochs 1000 \
  --no_resume
```

Generic withholding example:

```bash
python -u src/station_withholding_test.py \
  --data_path datasets/fog_ready/era5_trainable_daily.csv \
  --model_type multi_channel \
  --active_channels all \
  --epochs 1000 \
  --no_resume
```

## Lower Priority / Not First

- `baseline_random` is useful as a sanity check, but it is not the main scientific result.
- Channel ablations are secondary until the main `ST-LOBO` and `SLOBO` refresh is complete.
- Do not spend time on random CV or ablations before refreshing:
  - `baseline_stlobo`
  - `mc_stlobo`
  - `baseline_slobo`
  - `mc_slobo`
