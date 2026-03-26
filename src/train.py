"""
train.py — Training script for the ERA5 Offset MPT.

Pipeline:
  1. Load full comparison_all_years.csv.
  2. Apply temporal split (hold out test_year).
  3. Build static k-NN graph from all unique stations.
  4. Build folds based on cv_mode (random, slobo, st_lobo).
  5. For each fold: 
       - Fit scaler on fold's training data.
       - Train model with early stopping.
       - Evaluate with masked MAE loss.
  6. If cv_mode is st_lobo:
       - evaluate all best models on the test set to find the best fold.
  7. Report median MAE across folds.
  8. Save best checkpoint + scaler.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from src.config import cfg
from src.data.dataset import ERA5LandDataset, fit_scaler, save_scaler, standardize_input_columns
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.split import (
    temporal_split, restrict_train_years, build_slobo_folds, get_fold_masks, summarize_folds,
    build_random_station_folds, build_temporal_windows, get_st_fold_masks
)
from src.models.factory import build_model
from src.utils.loss import OffsetLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROGRESS_ENABLED = sys.stdout.isatty()


def progress_message(message):
    """Write log messages without breaking active tqdm bars."""
    if PROGRESS_ENABLED:
        tqdm.write(message)
    else:
        print(message, flush=True)


def selection_score(metrics):
    """Single scalar used for early stopping and checkpoint selection."""
    weighted = []
    weights = []
    for key, weight in (
        ("val_mae_tmax", cfg.loss.lambda_tmax),
        ("val_mae_tmin", cfg.loss.lambda_tmin),
    ):
        value = metrics.get(key, float("nan"))
        if not np.isnan(value):
            weighted.append(weight * value)
            weights.append(weight)
    if not weights:
        return float("inf")
    return sum(weighted) / sum(weights)

def evaluate_fold(model, dataset, get_mask_fn, device):
    """
    Evaluate model on the validation subset defined by get_mask_fn.
    get_mask_fn(batch) -> (train_mask, val_mask)
    """
    model.eval()
    preds_tmax, targets_tmax = [], []
    preds_tmin, targets_tmin = [], []
    era5_tmax_list, era5_tmin_list = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            x = batch["x"].to(device)
            y = batch["y"]
            valid_mask = batch["valid_mask"]

            train_mask, val_mask = get_mask_fn(batch)
            if val_mask.sum() == 0:
                continue

            pred = model(x, dataset.edge_index.to(device), dataset.edge_attr.to(device))  # (N, 2)
            pred = pred.cpu()

            val_valid_tmax = val_mask & valid_mask[:, 0].numpy()
            val_valid_tmin = val_mask & valid_mask[:, 1].numpy()

            if val_valid_tmax.any():
                preds_tmax.append(pred[val_valid_tmax, 0])
                targets_tmax.append(y[val_valid_tmax, 0])
                era5_tmax_list.append(torch.zeros(val_valid_tmax.sum()))

            if val_valid_tmin.any():
                preds_tmin.append(pred[val_valid_tmin, 1])
                targets_tmin.append(y[val_valid_tmin, 1])
                era5_tmin_list.append(torch.zeros(val_valid_tmin.sum()))

    def mae(preds, targets):
        if not preds:
            return float("nan")
        return (torch.cat(preds) - torch.cat(targets)).abs().mean().item()

    return {
        "val_mae_tmax": mae(preds_tmax, targets_tmax),
        "val_mae_tmin": mae(preds_tmin, targets_tmin),
        "baseline_mae_tmax": mae(era5_tmax_list, targets_tmax),
        "baseline_mae_tmin": mae(era5_tmin_list, targets_tmin),
    }


def run_fold(
    fold_name,
    fold_train_df,
    trainval_df,
    get_mask_fn,
    edge_index,
    edge_attr,
    station_order,
    args,
    device,
):
    """
    Train and validate one fold with early stopping. Returns best val MAE metrics.
    fold_train_df: subset of df to fit the scaler.
    trainval_df: full trainval df to iterate over.
    get_mask_fn: function returning train/val masks per batch.
    """
    progress_message(f"\n{'='*60}")
    progress_message(f"  Fold {fold_name}")
    progress_message(f"{'='*60}")

    # Fit scaler on this fold's training data only
    scaler_ds_tmp = ERA5LandDataset(fold_train_df, scaler=None)
    scaler = fit_scaler(scaler_ds_tmp)

    # Build full-timespan datasets for training loop
    full_dataset = ERA5LandDataset(trainval_df, scaler=scaler, station_order=station_order)
    full_dataset.edge_index = edge_index
    full_dataset.edge_attr = edge_attr

    model = build_model(cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = OffsetLoss(
        lambda_tmax=cfg.loss.lambda_tmax,
        lambda_tmin=cfg.loss.lambda_tmin,
    )

    edge_index_dev = edge_index.to(device)
    edge_attr_dev = edge_attr.to(device)

    best_score = float("inf")
    best_metrics = {}
    best_state = None
    
    patience_counter = 0
    max_patience = cfg.train.patience

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc=f"Fold {fold_name}",
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
        disable=not PROGRESS_ENABLED,
    )

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(
            range(len(full_dataset)),
            desc=f"    Epoch {epoch:4d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not PROGRESS_ENABLED,
        )
        for idx in batch_bar:
            batch = full_dataset[idx]
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            train_mask, val_mask = get_mask_fn(batch)
            train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)

            vm = valid_mask.clone()
            vm[~train_mask_t] = False

            if vm.sum() == 0:
                continue

            optimizer.zero_grad()
            pred = model(x, edge_index_dev, edge_attr_dev)
            loss, l_tmax, l_tmin = criterion(pred, y, vm)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        metrics = evaluate_fold(model, full_dataset, get_mask_fn, device)
        if PROGRESS_ENABLED:
            epoch_bar.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                val_tmax=f"{metrics['val_mae_tmax']:.4f}",
                val_tmin=f"{metrics['val_mae_tmin']:.4f}",
                base_tmax=f"{metrics['baseline_mae_tmax']:.4f}",
            )
        else:
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val MAE Tmax: {metrics['val_mae_tmax']:.4f} | "
                f"Val MAE Tmin: {metrics['val_mae_tmin']:.4f} | "
                f"Baseline Tmax: {metrics['baseline_mae_tmax']:.4f}",
                flush=True,
            )

        current_score = selection_score(metrics)
        if current_score < best_score:
            best_score = current_score
            best_metrics = metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            progress_message(
                f"  Early stopping triggered after {epoch} epochs (Patience={max_patience})."
            )
            break

    if PROGRESS_ENABLED and best_metrics:
        progress_message(
            f"  Best Fold {fold_name} | "
            f"Val MAE Tmax: {best_metrics['val_mae_tmax']:.4f} | "
            f"Val MAE Tmin: {best_metrics['val_mae_tmin']:.4f} | "
            f"Baseline Tmax: {best_metrics['baseline_mae_tmax']:.4f}"
        )

    return best_metrics, best_state, scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path)
    df = standardize_input_columns(df)
    df["time"] = pd.to_datetime(df["time"])

    trainval_df, test_df = temporal_split(df, test_year=cfg.split.test_year)
    trainval_df, selected_years = restrict_train_years(trainval_df, n_years=cfg.train.train_years)
    print(f"TrainVal rows: {len(trainval_df)} | Test rows: {len(test_df)}")
    print(f"Training years used: {selected_years}")

    unique_stations = (
        trainval_df[["station", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )
    edge_index, edge_attr, station_order = build_static_graph(unique_stations, k=cfg.graph.k)
    edge_attr, edge_scaler = normalize_edge_attr(edge_attr)
    print(f"Graph: {len(unique_stations)} nodes, {edge_index.shape[1]} edges (k={cfg.graph.k})")

    cv_mode = cfg.train.cv_mode.lower()
    print(f"Cross-Validation Mode: {cv_mode.upper()}")
    
    # Setup Folds
    folds = []
    
    if cv_mode == "random":
        print(f"Random Block splits (K={cfg.split.n_blocks}):")
        station_to_block = build_random_station_folds(unique_stations, n_blocks=cfg.split.n_blocks)
        summarize_folds(unique_stations, station_to_block)
        
        for b in range(cfg.split.n_blocks):
            train_station_ids = [s for s, block in station_to_block.items() if block != b]
            train_df_fold = trainval_df[trainval_df["station"].isin(train_station_ids)]
            
            def make_mask_fn(val_b):
                return lambda batch: get_fold_masks(batch["station_ids"], station_to_block, val_b)
                
            folds.append({
                "name": str(b),
                "train_df_fold": train_df_fold,
                "get_mask_fn": make_mask_fn(b)
            })

    elif cv_mode == "slobo":
        print(f"SLOBO spatial blocks (K={cfg.split.n_blocks}, random_state=42):")
        station_to_block = build_slobo_folds(unique_stations, n_blocks=cfg.split.n_blocks, random_state=42)
        summarize_folds(unique_stations, station_to_block)
        
        for b in range(cfg.split.n_blocks):
            train_station_ids = [s for s, block in station_to_block.items() if block != b]
            train_df_fold = trainval_df[trainval_df["station"].isin(train_station_ids)]
            
            def make_mask_fn(val_b):
                return lambda batch: get_fold_masks(batch["station_ids"], station_to_block, val_b)
                
            folds.append({
                "name": str(b),
                "train_df_fold": train_df_fold,
                "get_mask_fn": make_mask_fn(b)
            })

    elif cv_mode == "st_lobo":
        print(f"ST-LOBO spatial blocks (K={cfg.split.n_blocks}, random_state=42):")
        station_to_block = build_slobo_folds(unique_stations, n_blocks=cfg.split.n_blocks, random_state=42)
        summarize_folds(unique_stations, station_to_block)
        
        windows = build_temporal_windows(trainval_df, n_windows=cfg.split.n_windows)
        print(f"ST-LOBO temporal windows (T={cfg.split.n_windows}):")
        for i, (sy, ey) in enumerate(windows):
            print(f"  Window {i}: {sy}-{ey}")
            
        for s in range(cfg.split.n_blocks):
            for t in range(cfg.split.n_windows):
                val_stations = set(sid for sid, block in station_to_block.items() if block == s)
                start_year, end_year = windows[t]
                
                # Training rows: NOT in spatial block 's' OR NOT in temporal window 't'
                is_val_node = trainval_df["station"].isin(val_stations)
                is_val_time = (trainval_df["time"].dt.year >= start_year) & (trainval_df["time"].dt.year <= end_year)
                train_df_fold = trainval_df[~(is_val_node & is_val_time)]
                
                def make_mask_fn(val_s, val_t):
                    def _mask_fn(batch):
                        n = len(batch["station_ids"])
                        # batch["date"] is a string; repeat it N times (one per node)
                        # so get_st_fold_masks can call .year.values on a DatetimeIndex
                        dates_idx = pd.DatetimeIndex([batch["date"]] * n)
                        return get_st_fold_masks(
                            batch["station_ids"],
                            dates_idx,
                            station_to_block, windows, val_s, val_t,
                        )
                    return _mask_fn
                    
                folds.append({
                    "name": f"s={s}, t={t}",
                    "train_df_fold": train_df_fold,
                    "get_mask_fn": make_mask_fn(s, t)
                })

    # Filter folds if args.fold >= 0
    if args.fold >= 0 and args.fold < len(folds):
        folds = [folds[args.fold]]

    # Run Folds
    all_fold_metrics = []
    best_checkpoint = None
    best_scaler = None

    fold_iterator = tqdm(
        folds,
        desc="Folds",
        unit="fold",
        leave=True,
        dynamic_ncols=True,
        disable=not PROGRESS_ENABLED,
    )

    for fold_data in fold_iterator:
        metrics, state_dict, scaler = run_fold(
            fold_data["name"], fold_data["train_df_fold"], trainval_df,
            fold_data["get_mask_fn"], edge_index, edge_attr, station_order, args, device
        )
        # Verify min valid samples threshold
        if np.isnan(metrics["val_mae_tmax"]):
            print("  Skipping fold summary (NaN validation - insufficient val samples).")
            continue
            
        metrics["fold_name"] = fold_data["name"]
        metrics["state_dict"] = state_dict
        metrics["scaler"] = scaler
        all_fold_metrics.append(metrics)
        
        if PROGRESS_ENABLED:
            fold_iterator.set_postfix(
                fold=fold_data["name"],
                val_tmax=f"{metrics['val_mae_tmax']:.4f}",
            )

    if all_fold_metrics:
        best_fold = min(all_fold_metrics, key=selection_score)
        best_checkpoint = best_fold["state_dict"]
        best_scaler = best_fold["scaler"]

    if cv_mode == "st_lobo" and len(test_df) > 0 and best_checkpoint is not None and best_scaler is not None:
        print("\n" + "="*60)
        print("  Final Test Evaluation (best validation-selected ST-LOBO checkpoint)")
        print("="*60)

        test_edge_index, test_edge_attr, _ = build_static_graph(unique_stations, k=cfg.graph.k)
        test_edge_attr, _ = normalize_edge_attr(test_edge_attr, edge_scaler)

        def all_val_mask(batch):
            n = len(batch["station_ids"])
            return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

        model = build_model(cfg, dropout_override=0.0).to(device)
        model.load_state_dict(best_checkpoint)

        test_dataset = ERA5LandDataset(test_df, scaler=best_scaler, station_order=station_order)
        test_dataset.edge_index = test_edge_index
        test_dataset.edge_attr = test_edge_attr

        test_metrics = evaluate_fold(model, test_dataset, all_val_mask, device)
        print(f"  Test MAE Tmax: {test_metrics['val_mae_tmax']:.4f}")
        print(f"  Test MAE Tmin: {test_metrics['val_mae_tmin']:.4f}")
        print(f"  Baseline Tmax: {test_metrics['baseline_mae_tmax']:.4f}")
        print(f"  Baseline Tmin: {test_metrics['baseline_mae_tmin']:.4f}")

    print("\n" + "="*60)
    print("  Summary (median across folds)")
    print("="*60)
    for key in ["val_mae_tmax", "val_mae_tmin", "baseline_mae_tmax", "baseline_mae_tmin"]:
        vals = [m[key] for m in all_fold_metrics if not np.isnan(m.get(key, float("nan")))]
        if vals:
            print(f"  {key:<30}: {np.median(vals):.4f}  (std={np.std(vals):.4f})")

    if best_checkpoint is not None:
        ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "model_state_dict": best_checkpoint,
                "edge_scaler": {
                    "mean": edge_scaler["mean"].cpu(),
                    "std": edge_scaler["std"].cpu(),
                },
                "model_config": {
                    "model_type": cfg.model.model_type,
                    "num_channels": cfg.model.num_channels,
                    "aggregation": cfg.model.aggregation,
                    "active_channels": cfg.model.active_channels,
                    "in_features": cfg.model.in_features,
                    "hidden_dim": cfg.model.hidden_dim,
                    "heads": cfg.model.heads,
                    "num_gnn_layers": cfg.model.num_gnn_layers,
                    "edge_dim": cfg.model.edge_dim,
                    "out_dim": cfg.model.out_dim,
                    "dropout": cfg.model.dropout,
                },
                "graph_config": {
                    "k": cfg.graph.k,
                },
            },
            ckpt_path,
        )
        scaler_path = os.path.join(args.checkpoint_dir, "scaler.pkl")
        save_scaler(best_scaler, scaler_path)
        print(f"\nSaved checkpoint → {ckpt_path}")
        print(f"Saved scaler    → {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5 Offset MPT Training")
    parser.add_argument("--data_path", type=str, default=cfg.train.data_path)
    parser.add_argument("--checkpoint_dir", type=str, default=cfg.train.checkpoint_dir)
    parser.add_argument("--epochs", type=int, default=cfg.train.epochs)
    parser.add_argument("--lr", type=float, default=cfg.train.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.train.weight_decay)
    parser.add_argument("--train_years", type=int, default=cfg.train.train_years,
                        help="Use only the most recent N pre-test years for train/validation. "
                             "Set <= 0 to use all available years before test_year.")
    parser.add_argument("--fold", type=int, default=cfg.train.fold,
                        help="Fold index to train (0-indexed). -1 = run all folds.")
    parser.add_argument("--cv_mode", type=str, default=cfg.train.cv_mode,
                        help="Cross-validation mode: random, slobo, st_lobo")
    parser.add_argument("--model_type", type=str, default=cfg.model.model_type,
                        help="Model variant: baseline (default) or multi_channel")
    parser.add_argument("--num_channels", type=int, default=cfg.model.num_channels,
                        help="Parallel attention channels (multi_channel only)")
    parser.add_argument("--aggregation", type=str, default=cfg.model.aggregation,
                        help="Channel aggregation: mean (default) or concat")
    parser.add_argument("--active_channels", type=str, default=cfg.model.active_channels,
                        help="Active channel names for multi_channel model. "
                             "Comma-separated: temperature,pressure,terrain "
                             "or 'all' (default). Ablation: 'temperature,pressure'")
    args = parser.parse_args()

    # Override config with argparse values
    cfg.train.cv_mode = args.cv_mode
    cfg.train.train_years = args.train_years
    cfg.model.model_type = args.model_type
    cfg.model.num_channels = args.num_channels
    cfg.model.aggregation = args.aggregation
    cfg.model.active_channels = args.active_channels

    train(args)
