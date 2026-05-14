# Environment Setup

This repo currently expects a Python environment that can run:

- the core graph-learning pipeline in `src/`
- the plotting utilities in `plot_results.py`
- the optional external-baseline evaluation helpers in `scripts/`

## Core training environment

The fastest way to reproduce the core training stack used in this repo is:

```bash
python -m pip install -r requirements.txt
```

This installs:

- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `matplotlib`, `tqdm`, `pytest`
- `torch==2.3.0`
- `torch-geometric` and the matching compiled PyG extensions for `torch 2.3.0 + cu121`

## External baseline tooling

If you want to evaluate exported GraphCast or Aurora gridded outputs with:

- `scripts/evaluate_gridded_baseline.py`

install the optional dependencies too:

```bash
python -m pip install -r requirements-external-baselines.txt
```

If your exported files are in GRIB format, you will also need `cfgrib` and `eccodes`.

## Smoke checks

After installing dependencies, run:

```bash
python -m pytest src/tests/test_pipeline.py
python validate_architecture.py
python src/train.py --epochs 1 --cv_mode st_lobo --model_type baseline
```

## Notes

- The repo currently has no separate environment lockfile beyond the pinned requirements above.
- The `requirements.txt` file is pinned to the CUDA-enabled PyTorch Geometric wheel set for `torch==2.3.0`.
- If you change the local PyTorch or CUDA version, update the PyG wheel URL in `requirements.txt` to the matching build from `https://data.pyg.org/`.
