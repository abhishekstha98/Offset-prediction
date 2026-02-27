"""
train.py — Training script for the ERA5 Offset MPT.

Pipeline:
  1. Load full comparison_all_years.csv.
  2. Apply temporal split (hold out test_year).
  3. Build static k-NN graph from all unique stations.
  4. Fit node-feature scaler on training data.
  5. Run SLOBO cross-validation folds (or a single specified fold).
  6. For each fold: train and evaluate with masked MAE loss.
  7. Report median MAE across folds (robust to single bad stations).
  8. Save best-fold checkpoint + scaler.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.config import cfg
from src.data.dataset import ERA5LandDataset, fit_scaler, save_scaler
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.split import temporal_split, build_slobo_folds, get_fold_masks, summarize_folds
from src.models.mpt import OffsetMPT
from src.utils.loss import OffsetLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model, dataset, edge_index, edge_attr, station_to_block, val_block, device):
    """
    Evaluate model on the validation block of a given dataset split.
    Returns: dict with val_mae_tmax, val_mae_tmin, baseline_mae_tmax, baseline_mae_tmin.
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
            station_ids = batch["station_ids"]

            _, val_mask = get_fold_masks(station_ids, station_to_block, val_block)

            if val_mask.sum() == 0:
                continue

            pred = model(x, edge_index.to(device), edge_attr.to(device))  # (N, 2)
            pred = pred.cpu()

            # Validation nodes only
            val_valid_tmax = val_mask & valid_mask[:, 0].numpy()
            val_valid_tmin = val_mask & valid_mask[:, 1].numpy()

            if val_valid_tmax.any():
                preds_tmax.append(pred[val_valid_tmax, 0])
                targets_tmax.append(y[val_valid_tmax, 0])
                # Baseline: ERA5 is "0 offset" prediction, compared against target offset
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
        "baseline_mae_tmax": mae(era5_tmax_list, targets_tmax),  # ERA5 raw error in offset-space
        "baseline_mae_tmin": mae(era5_tmin_list, targets_tmin),
    }


def run_fold(fold_idx, trainval_df, station_to_block, edge_index, edge_attr, args, device):
    """Train and validate one SLOBO fold. Returns best val MAE metrics."""
    print(f"\n{'='*60}")
    print(f"  SLOBO Fold {fold_idx}")
    print(f"{'='*60}")

    # Split stations into train/val
    unique_stations = trainval_df[["station", "lat", "lon", "height"]].drop_duplicates("station")
    val_block = fold_idx

    n_val = sum(1 for v in station_to_block.values() if v == val_block)
    n_train = len(station_to_block) - n_val
    print(f"  Train stations: {n_train} | Val stations: {n_val}")

    # Fit scaler on training nodes' data only
    train_station_ids = [s for s, b in station_to_block.items() if b != val_block]
    train_df_fold = trainval_df[trainval_df["station"].isin(train_station_ids)]
    scaler_ds_tmp = ERA5LandDataset(train_df_fold, scaler=None)
    scaler = fit_scaler(scaler_ds_tmp)

    # Build full-timespan datasets (all stations in graph, scaler applied)
    full_dataset = ERA5LandDataset(trainval_df, scaler=scaler)

    # Model, optimizer, loss
    model = OffsetMPT(
        in_features=cfg.model.in_features,
        hidden_dim=cfg.model.hidden_dim,
        heads=cfg.model.heads,
        num_gnn_layers=cfg.model.num_gnn_layers,
        edge_dim=cfg.model.edge_dim,
        out_dim=cfg.model.out_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = OffsetLoss(
        lambda_tmax=cfg.loss.lambda_tmax,
        lambda_tmin=cfg.loss.lambda_tmin,
    )

    edge_index_dev = edge_index.to(device)
    edge_attr_dev = edge_attr.to(device)

    best_val_tmax = float("inf")
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for idx in range(len(full_dataset)):
            batch = full_dataset[idx]
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            station_ids = batch["station_ids"]

            train_mask, _ = get_fold_masks(station_ids, station_to_block, val_block)
            train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)

            # Mask valid_mask further to only training nodes
            vm = valid_mask.clone()
            vm[~train_mask_t] = False

            if vm.sum() == 0:
                continue

            optimizer.zero_grad()
            pred = model(x, edge_index_dev, edge_attr_dev)  # (N, 2)
            loss, l_tmax, l_tmin = criterion(pred, y, vm)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            metrics = evaluate(model, full_dataset, edge_index, edge_attr,
                               station_to_block, val_block, device)
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val MAE Tmax: {metrics['val_mae_tmax']:.4f} | "
                f"Val MAE Tmin: {metrics['val_mae_tmin']:.4f} | "
                f"Baseline Tmax: {metrics['baseline_mae_tmax']:.4f}"
            )
            if metrics["val_mae_tmax"] < best_val_tmax:
                best_val_tmax = metrics["val_mae_tmax"]
                best_metrics = metrics
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_metrics, best_state, scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load data
    print("Loading comparison_all_years.csv ...")
    df = pd.read_csv(args.data_path)

    # 2. Temporal split
    trainval_df, test_df = temporal_split(df, test_year=cfg.split.test_year)
    print(f"TrainVal rows: {len(trainval_df)} | Test rows: {len(test_df)}")

    # 3. Build static graph (from ALL unique stations in trainval)
    unique_stations = (
        trainval_df[["station", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )
    edge_index, edge_attr, station_order = build_static_graph(
        unique_stations, k=cfg.graph.k
    )
    edge_attr, edge_scaler = normalize_edge_attr(edge_attr)
    print(f"Graph: {len(unique_stations)} nodes, {edge_index.shape[1]} edges (k={cfg.graph.k})")

    # 4. SLOBO fold assignments
    station_to_block = build_slobo_folds(unique_stations, n_blocks=cfg.split.n_blocks)
    print(f"\nSLOBO blocks (K={cfg.split.n_blocks}):")
    summarize_folds(unique_stations, station_to_block)

    # 5. Run folds
    folds_to_run = list(range(cfg.split.n_blocks)) if args.fold < 0 else [args.fold]
    all_fold_metrics = []
    best_overall_tmax = float("inf")
    best_checkpoint = None
    best_scaler = None

    for fold_idx in folds_to_run:
        metrics, state_dict, scaler = run_fold(
            fold_idx, trainval_df, station_to_block,
            edge_index, edge_attr, args, device
        )
        all_fold_metrics.append(metrics)
        if metrics.get("val_mae_tmax", float("inf")) < best_overall_tmax:
            best_overall_tmax = metrics["val_mae_tmax"]
            best_checkpoint = state_dict
            best_scaler = scaler

    # 6. Report aggregate metrics using MEDIAN (robust to single bad stations)
    print("\n" + "="*60)
    print("  SLOBO Summary (median across folds)")
    print("="*60)
    for key in ["val_mae_tmax", "val_mae_tmin", "baseline_mae_tmax", "baseline_mae_tmin"]:
        vals = [m[key] for m in all_fold_metrics if not np.isnan(m.get(key, float("nan")))]
        if vals:
            print(f"  {key:<30}: {np.median(vals):.4f}  (std={np.std(vals):.4f})")

    # 7. Save best checkpoint and scaler
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    torch.save({"model_state_dict": best_checkpoint, "edge_scaler": edge_scaler}, ckpt_path)
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
    parser.add_argument("--fold", type=int, default=cfg.train.fold,
                        help="SLOBO fold index to train (0-indexed). -1 = run all folds.")
    args = parser.parse_args()
    train(args)
