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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import pickle
import re
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from src.config import cfg
from src.data.dataset import (
    ERA5LandDataset,
    fit_scaler,
    save_scaler,
    standardize_input_columns,
)
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.hourly_to_daily import load_training_frame
from src.data.split import (
    temporal_split, restrict_train_years, build_slobo_folds, get_fold_masks, summarize_folds,
    build_random_station_folds, build_temporal_windows, get_st_fold_masks
)
from src.models.factory import build_model
from src.utils.loss import BackboneMultiTaskLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROGRESS_ENABLED = sys.stdout.isatty()
SUMMARY_KEYS = [
    "val_mae_tmax",
    "val_mae_tmin",
    "baseline_mae_tmax",
    "baseline_mae_tmin",
]


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


def iso_now():
    """Return a compact ISO-8601 timestamp in local time."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip().lower()).strip("-") or "default"


def _to_cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_copy(v) for v in value)
    return value


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _atomic_pickle_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def _atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _write_status_json(state, path):
    public_state = {
        "run_key": state.get("run_key"),
        "status": state.get("status"),
        "started_at": state.get("started_at"),
        "updated_at": state.get("updated_at"),
        "completed_at": state.get("completed_at"),
        "current_fold": state.get("active_fold_name"),
        "current_epoch": state.get("active_epoch"),
        "n_total_folds": state.get("n_total_folds"),
        "completed_folds": [fold["fold_name"] for fold in state.get("completed_folds", [])],
        "n_completed_folds": len(state.get("completed_folds", [])),
        "last_error": state.get("last_error"),
        "final_summary": state.get("final_summary"),
        "final_test_metrics": state.get("final_test_metrics"),
        "metadata_path": state.get("metadata_path"),
        "active_snapshot_path": state.get("active_snapshot_path"),
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(public_state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _persist_run_state(state):
    state["updated_at"] = iso_now()
    _atomic_pickle_save(state, state["metadata_path"])
    _write_status_json(state, state["status_path"])


def _run_signature(args):
    return {
        "data_path": os.path.abspath(args.data_path),
        "cv_mode": args.cv_mode,
        "model_type": args.model_type,
        "num_channels": args.num_channels,
        "aggregation": args.aggregation,
        "active_channels": args.active_channels,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_years": args.train_years,
        "fold": args.fold,
        "sequence_length": args.sequence_length,
        "temporal_layers": args.temporal_layers,
        "max_seq_len": args.max_seq_len,
        "temporal_pooling": args.temporal_pooling,
        "enable_fog_head": args.enable_fog_head,
        "fog_out_dim": args.fog_out_dim,
        "min_hours_per_day": args.min_hours_per_day,
        "patience": cfg.train.patience,
        "graph_k": cfg.graph.k,
        "n_blocks": cfg.split.n_blocks,
        "n_windows": cfg.split.n_windows,
        "test_year": cfg.split.test_year,
        "hidden_dim": cfg.model.hidden_dim,
        "heads": cfg.model.heads,
        "num_gnn_layers": cfg.model.num_gnn_layers,
        "dropout": cfg.model.dropout,
    }


def _run_key(args):
    parts = [
        args.cv_mode,
        args.model_type,
        f"channels-{_safe_slug(args.active_channels)}",
        f"years{args.train_years}",
        f"epochs{args.epochs}",
        f"seq{args.sequence_length}",
        f"tl{args.temporal_layers}",
        f"pool-{_safe_slug(args.temporal_pooling)}",
        f"agg-{_safe_slug(args.aggregation)}",
        f"fold{args.fold}",
    ]
    if args.enable_fog_head:
        parts.append(f"fog{args.fog_out_dim}")
    return "_".join(parts)


def _metadata_path(args):
    return os.path.join(args.resume_dir, f"train_{_run_key(args)}.pkl")


def _active_snapshot_path(args):
    return os.path.join(args.resume_dir, f"train_{_run_key(args)}.active.pt")


def _status_path(args):
    return os.path.join(args.resume_dir, f"train_{_run_key(args)}.status.json")


def _initial_run_state(args):
    metadata_path = _metadata_path(args)
    active_snapshot_path = _active_snapshot_path(args)
    status_path = _status_path(args)
    now = iso_now()
    return {
        "run_key": _run_key(args),
        "config": _run_signature(args),
        "status": "initialized",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "completed_folds": [],
        "active_fold_name": None,
        "active_epoch": 0,
        "n_total_folds": None,
        "last_error": None,
        "final_summary": None,
        "final_test_metrics": None,
        "metadata_path": metadata_path,
        "active_snapshot_path": active_snapshot_path,
        "status_path": status_path,
    }


def _load_run_state(args):
    os.makedirs(args.resume_dir, exist_ok=True)
    metadata_path = _metadata_path(args)
    active_snapshot_path = _active_snapshot_path(args)
    status_path = _status_path(args)

    if not args.resume:
        if os.path.exists(active_snapshot_path):
            os.remove(active_snapshot_path)
        state = _initial_run_state(args)
        progress_message(f"RUN_STATUS: STARTED | fresh run | state={metadata_path}")
        return state

    if not os.path.exists(metadata_path):
        state = _initial_run_state(args)
        progress_message(f"RUN_STATUS: STARTED | fresh run | state={metadata_path}")
        return state

    with open(metadata_path, "rb") as f:
        state = pickle.load(f)

    expected = _run_signature(args)
    actual = state.get("config", {})
    mismatches = []
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            mismatches.append(f"{key}: state={actual.get(key)!r}, current={expected_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches[:8])
        raise ValueError(
            f"Resume state {metadata_path} does not match this run. {mismatch_text}. "
            "Use --no_resume or a different --resume_dir to start fresh."
        )

    state["metadata_path"] = metadata_path
    state["active_snapshot_path"] = active_snapshot_path
    state["status_path"] = status_path
    progress_message(f"RUN_STATUS: RESUMING | state={metadata_path}")
    return state


def _load_active_snapshot(state):
    path = state.get("active_snapshot_path")
    if path and os.path.exists(path):
        # Resume snapshots are locally generated training state, not public
        # model weights. PyTorch 2.6 changed torch.load(..., weights_only=True)
        # by default, which breaks loading these richer dicts.
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


def _record_active_fold_progress(state, snapshot):
    _atomic_torch_save(snapshot, state["active_snapshot_path"])
    state["status"] = "running"
    state["completed_at"] = None
    state["last_error"] = None
    state["active_fold_name"] = snapshot["fold_name"]
    state["active_epoch"] = snapshot["epoch"]
    _persist_run_state(state)


def _clear_active_fold_progress(state):
    snapshot_path = state.get("active_snapshot_path")
    if snapshot_path and os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    state["active_fold_name"] = None
    state["active_epoch"] = 0


def _upsert_completed_fold(state, fold_record):
    completed = [fold for fold in state["completed_folds"] if fold["fold_name"] != fold_record["fold_name"]]
    completed.append(fold_record)
    state["completed_folds"] = completed


def _compute_summary(all_fold_metrics):
    summary = {}
    for key in SUMMARY_KEYS:
        vals = [m[key] for m in all_fold_metrics if not np.isnan(m.get(key, float("nan")))]
        if vals:
            summary[key] = {
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
            }
    return summary


def _print_summary(summary):
    print("\n" + "="*60)
    print("  Summary (median across folds)")
    print("="*60)
    for key in SUMMARY_KEYS:
        if key in summary:
            print(
                f"  {key:<30}: "
                f"{summary[key]['median']:.4f}  (std={summary[key]['std']:.4f})"
            )


def _save_best_artifacts(best_fold, edge_scaler, args):
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": best_fold["state_dict"],
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
                "sequence_length": cfg.model.sequence_length,
                "temporal_layers": cfg.model.temporal_layers,
                "max_seq_len": cfg.model.max_seq_len,
                "temporal_pooling": cfg.model.temporal_pooling,
                "edge_dim": cfg.model.edge_dim,
                "out_dim": cfg.model.out_dim,
                "enable_fog_head": cfg.model.enable_fog_head,
                "fog_out_dim": cfg.model.fog_out_dim,
                "dropout": cfg.model.dropout,
                "feature_columns": best_fold["scaler"].get("feature_columns"),
            },
            "graph_config": {
                "k": cfg.graph.k,
            },
        },
        ckpt_path,
    )
    scaler_path = os.path.join(args.checkpoint_dir, "scaler.pkl")
    save_scaler(best_fold["scaler"], scaler_path)
    print(f"\nSaved checkpoint → {ckpt_path}")
    print(f"Saved scaler    → {scaler_path}")


def _forward_predictions(model, x, edge_index, edge_attr):
    if hasattr(model, "forward_multitask"):
        return model.forward_multitask(x, edge_index, edge_attr)
    pred = model(x, edge_index, edge_attr)
    return {"offset": pred, "fog_logits": None, "hidden": None}

def evaluate_fold(model, dataset, get_mask_fn, device):
    """
    Evaluate model on the validation subset defined by get_mask_fn.
    get_mask_fn(batch) -> (train_mask, val_mask)
    """
    model.eval()
    preds_tmax, targets_tmax = [], []
    preds_tmin, targets_tmin = [], []
    era5_tmax_list, era5_tmin_list = [], []
    fog_logits_list, fog_targets_list = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            x = batch["x"].to(device)
            y = batch["y"]
            valid_mask = batch["valid_mask"]

            train_mask, val_mask = get_mask_fn(batch)
            if val_mask.sum() == 0:
                continue

            outputs = _forward_predictions(
                model,
                x,
                dataset.edge_index.to(device),
                dataset.edge_attr.to(device),
            )
            pred = outputs["offset"].cpu()

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

            fog_logits = outputs.get("fog_logits")
            fog_valid_mask = batch.get("fog_valid_mask")
            fog_target = batch.get("fog_target")
            if (
                fog_logits is not None
                and fog_valid_mask is not None
                and fog_target is not None
            ):
                fog_mask_np = val_mask & fog_valid_mask.numpy()
                if fog_mask_np.any():
                    fog_logits_list.append(fog_logits.cpu().squeeze(-1)[fog_mask_np])
                    fog_targets_list.append(fog_target[fog_mask_np])

    def mae(preds, targets):
        if not preds:
            return float("nan")
        return (torch.cat(preds) - torch.cat(targets)).abs().mean().item()

    def fog_bce(preds, targets):
        if not preds:
            return float("nan")
        logits = torch.cat(preds)
        target = torch.cat(targets).float()
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target).item()

    return {
        "val_mae_tmax": mae(preds_tmax, targets_tmax),
        "val_mae_tmin": mae(preds_tmin, targets_tmin),
        "baseline_mae_tmax": mae(era5_tmax_list, targets_tmax),
        "baseline_mae_tmin": mae(era5_tmin_list, targets_tmin),
        "val_fog_bce": fog_bce(fog_logits_list, fog_targets_list),
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
    resume_snapshot=None,
    on_epoch_end=None,
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

    if resume_snapshot is not None:
        scaler = resume_snapshot["scaler"]
    else:
        scaler_ds_tmp = ERA5LandDataset(
            fold_train_df,
            scaler=None,
            sequence_length=cfg.model.sequence_length,
        )
        scaler = fit_scaler(scaler_ds_tmp)

    full_dataset = ERA5LandDataset(
        trainval_df,
        scaler=scaler,
        station_order=station_order,
        sequence_length=cfg.model.sequence_length,
    )
    full_dataset.edge_index = edge_index
    full_dataset.edge_attr = edge_attr

    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = BackboneMultiTaskLoss(
        lambda_tmax=cfg.loss.lambda_tmax,
        lambda_tmin=cfg.loss.lambda_tmin,
        lambda_fog=cfg.loss.lambda_fog,
    )

    edge_index_dev = edge_index.to(device)
    edge_attr_dev = edge_attr.to(device)

    if resume_snapshot is not None:
        model.load_state_dict(resume_snapshot["model_state_dict"])
        optimizer.load_state_dict(resume_snapshot["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        best_score = resume_snapshot["best_score"]
        best_metrics = resume_snapshot["best_metrics"]
        best_state = resume_snapshot["best_state"]
        patience_counter = resume_snapshot["patience_counter"]
        start_epoch = int(resume_snapshot["epoch"]) + 1
        progress_message(
            f"  Resuming fold {fold_name} from epoch {start_epoch} "
            f"(last completed epoch: {resume_snapshot['epoch']})."
        )
    else:
        best_score = float("inf")
        best_metrics = {}
        best_state = None
        patience_counter = 0
        start_epoch = 1

    max_patience = cfg.train.patience

    epoch_bar = tqdm(
        range(start_epoch, args.epochs + 1),
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
            fog_target = batch["fog_target"].to(device)
            fog_valid_mask = batch["fog_valid_mask"].to(device)

            train_mask, val_mask = get_mask_fn(batch)
            train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)

            vm = valid_mask.clone()
            vm[~train_mask_t] = False

            if vm.sum() == 0:
                continue

            optimizer.zero_grad()
            outputs = _forward_predictions(model, x, edge_index_dev, edge_attr_dev)
            fog_vm = fog_valid_mask & train_mask_t
            losses = criterion(
                outputs["offset"],
                y,
                vm,
                fog_logits=outputs["fog_logits"],
                fog_target=fog_target,
                fog_valid_mask=fog_vm,
            )
            losses["total"].backward()
            optimizer.step()

            epoch_loss += losses["total"].item()
            n_batches += 1
            batch_bar.set_postfix(loss=f"{losses['total'].item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        metrics = evaluate_fold(model, full_dataset, get_mask_fn, device)
        if PROGRESS_ENABLED:
            epoch_bar.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                val_tmax=f"{metrics['val_mae_tmax']:.4f}",
                val_tmin=f"{metrics['val_mae_tmin']:.4f}",
                base_tmax=f"{metrics['baseline_mae_tmax']:.4f}",
            )
            if not np.isnan(metrics.get("val_fog_bce", float("nan"))):
                epoch_bar.set_postfix(
                    train_loss=f"{avg_loss:.4f}",
                    val_tmax=f"{metrics['val_mae_tmax']:.4f}",
                    val_tmin=f"{metrics['val_mae_tmin']:.4f}",
                    val_fog=f"{metrics['val_fog_bce']:.4f}",
                )
        else:
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val MAE Tmax: {metrics['val_mae_tmax']:.4f} | "
                f"Val MAE Tmin: {metrics['val_mae_tmin']:.4f} | "
                f"Baseline Tmax: {metrics['baseline_mae_tmax']:.4f}"
                + (
                    f" | Val Fog BCE: {metrics['val_fog_bce']:.4f}"
                    if not np.isnan(metrics.get("val_fog_bce", float("nan")))
                    else ""
                ),
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
            if on_epoch_end is not None:
                on_epoch_end(
                    {
                        "fold_name": fold_name,
                        "epoch": epoch,
                        "scaler": scaler,
                        "model_state_dict": _to_cpu_copy(model.state_dict()),
                        "optimizer_state_dict": _to_cpu_copy(optimizer.state_dict()),
                        "best_score": best_score,
                        "best_metrics": dict(best_metrics),
                        "best_state": _to_cpu_copy(best_state),
                        "patience_counter": patience_counter,
                    }
                )
            break

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "fold_name": fold_name,
                    "epoch": epoch,
                    "scaler": scaler,
                    "model_state_dict": _to_cpu_copy(model.state_dict()),
                    "optimizer_state_dict": _to_cpu_copy(optimizer.state_dict()),
                    "best_score": best_score,
                    "best_metrics": dict(best_metrics),
                    "best_state": _to_cpu_copy(best_state),
                    "patience_counter": patience_counter,
                }
            )

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

def _evaluate_st_lobo_test(
    best_fold,
    unique_stations,
    test_df,
    station_order,
    edge_scaler,
    device,
):
    test_edge_index, test_edge_attr, _ = build_static_graph(unique_stations, k=cfg.graph.k)
    test_edge_attr, _ = normalize_edge_attr(test_edge_attr, edge_scaler)

    def all_val_mask(batch):
        n = len(batch["station_ids"])
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

    model = build_model(cfg, dropout_override=0.0).to(device)
    model.load_state_dict(best_fold["state_dict"])

    test_dataset = ERA5LandDataset(
        test_df,
        scaler=best_fold["scaler"],
        station_order=station_order,
        sequence_length=cfg.model.sequence_length,
    )
    test_dataset.edge_index = test_edge_index
    test_dataset.edge_attr = test_edge_attr
    return evaluate_fold(model, test_dataset, all_val_mask, device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_state = _load_run_state(args)

    print(f"Loading {args.data_path} ...")
    df, converted_from_hourly = load_training_frame(
        args.data_path,
        min_hours_per_day=args.min_hours_per_day,
    )
    if converted_from_hourly:
        print(
            "Detected hourly station/ERA5 schema; converted it to the daily "
            "training format in memory."
        )
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

    if run_state.get("n_total_folds") not in (None, len(folds)):
        raise ValueError(
            f"Resume state expects {run_state['n_total_folds']} folds, but this run has {len(folds)}. "
            "Use --no_resume or a different --resume_dir to start fresh."
        )
    run_state["n_total_folds"] = len(folds)
    run_state["last_error"] = None

    completed_lookup = {
        fold_record["fold_name"]: fold_record for fold_record in run_state.get("completed_folds", [])
    }
    active_snapshot = _load_active_snapshot(run_state)

    if run_state.get("status") == "completed" and len(completed_lookup) == len(folds):
        progress_message("RUN_STATUS: COMPLETED | existing resume state already finished.")
        all_fold_metrics = [completed_lookup[fold_data["name"]] for fold_data in folds]
        summary = run_state.get("final_summary") or _compute_summary(all_fold_metrics)
        final_test_metrics = run_state.get("final_test_metrics")
        if all_fold_metrics:
            best_fold = min(all_fold_metrics, key=selection_score)
            _save_best_artifacts(best_fold, edge_scaler, args)
            if cv_mode == "st_lobo" and final_test_metrics is not None:
                print("\n" + "="*60)
                print("  Final Test Evaluation (best validation-selected ST-LOBO checkpoint)")
                print("="*60)
                print(f"  Test MAE Tmax: {final_test_metrics['val_mae_tmax']:.4f}")
                print(f"  Test MAE Tmin: {final_test_metrics['val_mae_tmin']:.4f}")
                print(f"  Baseline Tmax: {final_test_metrics['baseline_mae_tmax']:.4f}")
                print(f"  Baseline Tmin: {final_test_metrics['baseline_mae_tmin']:.4f}")
        _print_summary(summary)
        return

    run_state["status"] = "running"
    _persist_run_state(run_state)

    all_fold_metrics = []
    for fold_data in folds:
        if fold_data["name"] in completed_lookup:
            all_fold_metrics.append(completed_lookup[fold_data["name"]])

    fold_iterator = tqdm(
        folds,
        desc="Folds",
        unit="fold",
        leave=True,
        dynamic_ncols=True,
        disable=not PROGRESS_ENABLED,
    )

    try:
        for fold_data in fold_iterator:
            fold_name = fold_data["name"]
            if fold_name in completed_lookup:
                progress_message(f"  Skipping completed fold {fold_name}.")
                continue

            run_state["active_fold_name"] = fold_name
            if active_snapshot is None or active_snapshot.get("fold_name") != fold_name:
                run_state["active_epoch"] = 0
                _persist_run_state(run_state)

            resume_snapshot = None
            if active_snapshot is not None and active_snapshot.get("fold_name") == fold_name:
                resume_snapshot = active_snapshot

            def persist_epoch(snapshot):
                _record_active_fold_progress(run_state, snapshot)

            metrics, state_dict, scaler = run_fold(
                fold_name,
                fold_data["train_df_fold"],
                trainval_df,
                fold_data["get_mask_fn"],
                edge_index,
                edge_attr,
                station_order,
                args,
                device,
                resume_snapshot=resume_snapshot,
                on_epoch_end=persist_epoch,
            )
            active_snapshot = None

            if np.isnan(metrics["val_mae_tmax"]):
                print("  Skipping fold summary (NaN validation - insufficient val samples).")
                _clear_active_fold_progress(run_state)
                _persist_run_state(run_state)
                continue

            fold_record = dict(metrics)
            fold_record["fold_name"] = fold_name
            fold_record["state_dict"] = state_dict
            fold_record["scaler"] = scaler
            completed_lookup[fold_name] = fold_record
            all_fold_metrics = [
                completed_lookup[name]
                for name in [fold["name"] for fold in folds]
                if name in completed_lookup
            ]
            _upsert_completed_fold(run_state, fold_record)
            _clear_active_fold_progress(run_state)
            _persist_run_state(run_state)

            if PROGRESS_ENABLED:
                fold_iterator.set_postfix(
                    fold=fold_name,
                    val_tmax=f"{metrics['val_mae_tmax']:.4f}",
                )
    except KeyboardInterrupt:
        run_state["status"] = "interrupted"
        run_state["last_error"] = "KeyboardInterrupt"
        _persist_run_state(run_state)
        progress_message("RUN_STATUS: INTERRUPTED")
        raise
    except Exception:
        run_state["status"] = "failed"
        run_state["last_error"] = traceback.format_exc()
        _persist_run_state(run_state)
        progress_message("RUN_STATUS: FAILED")
        raise

    best_fold = min(all_fold_metrics, key=selection_score) if all_fold_metrics else None
    final_test_metrics = None

    if cv_mode == "st_lobo" and len(test_df) > 0 and best_fold is not None:
        print("\n" + "="*60)
        print("  Final Test Evaluation (best validation-selected ST-LOBO checkpoint)")
        print("="*60)
        final_test_metrics = _evaluate_st_lobo_test(
            best_fold,
            unique_stations,
            test_df,
            station_order,
            edge_scaler,
            device,
        )
        print(f"  Test MAE Tmax: {final_test_metrics['val_mae_tmax']:.4f}")
        print(f"  Test MAE Tmin: {final_test_metrics['val_mae_tmin']:.4f}")
        print(f"  Baseline Tmax: {final_test_metrics['baseline_mae_tmax']:.4f}")
        print(f"  Baseline Tmin: {final_test_metrics['baseline_mae_tmin']:.4f}")

    summary = _compute_summary(all_fold_metrics)
    _print_summary(summary)

    if best_fold is not None:
        _save_best_artifacts(best_fold, edge_scaler, args)

    run_state["status"] = "completed"
    run_state["completed_at"] = iso_now()
    run_state["final_summary"] = summary
    run_state["final_test_metrics"] = final_test_metrics
    run_state["last_error"] = None
    _clear_active_fold_progress(run_state)
    _persist_run_state(run_state)
    progress_message("RUN_STATUS: COMPLETED")


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
                             "Comma-separated: temperature,humidity_stability,wind,terrain "
                             "or 'all' (default).")
    parser.add_argument("--sequence_length", type=int, default=cfg.model.sequence_length,
                        help="Number of time steps per graph sample. 1 keeps daily snapshots; "
                             ">1 enables temporal self-attention in OffsetMPT.")
    parser.add_argument("--temporal_layers", type=int, default=cfg.model.temporal_layers,
                        help="Number of temporal TransformerEncoder layers in OffsetMPT.")
    parser.add_argument("--max_seq_len", type=int, default=cfg.model.max_seq_len,
                        help="Maximum supported temporal context for learned temporal embeddings.")
    parser.add_argument("--temporal_pooling", type=str, default=cfg.model.temporal_pooling,
                        help="Temporal pooling mode for OffsetMPT: last or attention.")
    parser.add_argument("--enable_fog_head", action="store_true",
                        help="Attach an auxiliary fog/visibility head to the shared backbone.")
    parser.add_argument("--fog_out_dim", type=int, default=cfg.model.fog_out_dim,
                        help="Fog head output dimension. Use 1 for binary fog logits.")
    parser.add_argument(
        "--min_hours_per_day",
        type=int,
        default=20,
        help="When --data_path points to raw hourly era5_merged.csv, require at least "
             "this many hourly rows per station-day before daily aggregation.",
    )
    parser.add_argument("--resume_dir", type=str, default="outputs/train_state",
                        help="Directory for resumable training state files.")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Ignore existing train-state files and start fresh.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    # Override config with argparse values
    cfg.train.cv_mode = args.cv_mode
    cfg.train.train_years = args.train_years
    cfg.model.model_type = args.model_type
    cfg.model.num_channels = args.num_channels
    cfg.model.aggregation = args.aggregation
    cfg.model.active_channels = args.active_channels
    cfg.model.sequence_length = args.sequence_length
    cfg.model.temporal_layers = args.temporal_layers
    cfg.model.max_seq_len = args.max_seq_len
    cfg.model.temporal_pooling = args.temporal_pooling
    cfg.model.enable_fog_head = args.enable_fog_head
    cfg.model.fog_out_dim = args.fog_out_dim

    train(args)
