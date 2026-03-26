"""
station_withholding_test.py — Evaluate MPT spatial generalization degradation.

Protocol:
1. Load trainval dataset (2015-2023).
2. Randomly hold out `m` stations.
3. Train model on remaining `23-m` stations.
4. Evaluate MAE on seen vs unseen stations.
5. Repeat for m in [1, 3, 5, 10, 15] to plot a performance degradation curve.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from src.config import cfg
from src.data.dataset import ERA5LandDataset, fit_scaler, standardize_input_columns
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.split import temporal_split, restrict_train_years
from src.models.mpt import OffsetMPT
from src.models.factory import build_model
from src.utils.loss import OffsetLoss
from src.train import evaluate_fold


import time

PROGRESS_ENABLED = sys.stdout.isatty()


def progress_message(message):
    """Write log messages without breaking active tqdm bars."""
    if PROGRESS_ENABLED:
        tqdm.write(message)
    else:
        print(message, flush=True)

def run_withholding_experiment(
    m,
    trainval_df,
    unique_stations,
    edge_index,
    edge_attr,
    station_order,
    device,
    args,
):
    """Run withholding experiment iteratively."""
    progress_message(f"\n" + "="*50)
    progress_message(f"  Experiment: Withholding m={m} stations")
    progress_message("="*50)

    all_sids = unique_stations["station"].tolist()
    
    # 1. Determine number of iterations
    if m == 0:
        n_iters = 1
    elif m == 1:
        n_iters = len(all_sids) # Leave-one-out for all 23
    else:
        n_iters = min(10, len(all_sids)) # Monte Carlo N=10 max, bounded by total stations just in case
        
    progress_message(f"  Running {n_iters} iterations for m={m}...")

    # Data structures to accumulate results across iterations
    iter_best_seen = []
    iter_best_unseen = []
    iter_baseline = []

    # For plotting average learning curves, we will accumulate loss/mae per epoch.
    # Because early stopping makes epochs variable, we pad with the best last known value.
    epoch_curves = {
        "train_loss": [[] for _ in range(args.epochs)],
        "seen_mae": [[] for _ in range(args.epochs)],
        "unseen_mae": [[] for _ in range(args.epochs)]
    }
    epoch_real_counts = [0 for _ in range(args.epochs)]

    start_time_all = time.time()
    
    iteration_bar = tqdm(
        range(n_iters),
        desc=f"m={m}",
        unit="iter",
        leave=False,
        dynamic_ncols=True,
        disable=not PROGRESS_ENABLED,
    )

    for it in iteration_bar:
        if not PROGRESS_ENABLED:
            print(f"\n  --- Iteration {it+1}/{n_iters} ---", flush=True)
        iter_start = time.time()
        
        # 2. Randomly split stations into SEEN (train) and UNSEEN (holdout)
        # Dynamic reproducible seed per m and iteration
        random.seed(args.seed + m * 1000 + it) 
        
        if m == 1:
            # Deterministic leave-one-out
            unseen_sids = {all_sids[it]}
        else:
            unseen_sids = set(random.sample(all_sids, m))
            
        seen_sids = set(all_sids) - unseen_sids
        progress_message(f"  Unseen stations ({m}): {sorted(unseen_sids)}")

        # 3. Fit scaler ONLY on seen (training) stations
        train_df = trainval_df[trainval_df["station"].isin(seen_sids)]
        scaler_ds = ERA5LandDataset(train_df, scaler=None)
        scaler = fit_scaler(scaler_ds)

        # 4. Create full dataset for inference/loss
        full_dataset = ERA5LandDataset(trainval_df, scaler=scaler, station_order=station_order)
        full_dataset.edge_index = edge_index
        full_dataset.edge_attr = edge_attr

        # 5. Define Masks
        def get_mask(batch, sids_set):
            mask = np.array([sid in sids_set for sid in batch["station_ids"]])
            return mask, mask 

        # 6. Model setup
        model = build_model(cfg).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = OffsetLoss(lambda_tmax=cfg.loss.lambda_tmax, lambda_tmin=cfg.loss.lambda_tmin)
        
        edge_index_dev = edge_index.to(device)
        edge_attr_dev = edge_attr.to(device)

        best_unseen_tmax = float("inf")
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_counter = 0
        use_seen_for_stopping = (m == 0)
        # Track the last valid metric for padding after early stopping
        last_train_loss = 0.0
        last_seen_mae = 0.0
        last_unseen_mae = 0.0

        epoch_bar = tqdm(
            range(1, args.epochs + 1),
            desc=f"m={m} iter={it+1}",
            unit="epoch",
            leave=False,
            dynamic_ncols=True,
            disable=not PROGRESS_ENABLED,
        )

        for epoch in epoch_bar:
            if epoch == 2 and it == 0:
                # Time estimation after 1st epoch of 1st iteration
                time_per_epoch = time.time() - iter_start
                est_total = time_per_epoch * cfg.train.patience * 2 * n_iters  # very rough estimate
                progress_message(
                    f"  [Time Est] ~{time_per_epoch:.1f}s/epoch. ETA for m={m}: ~{est_total/60:.1f} mins."
                )

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

                is_seen = torch.tensor([sid in seen_sids for sid in batch["station_ids"]],
                                        dtype=torch.bool, device=device)
                vm = valid_mask.clone()
                vm[~is_seen] = False

                if vm.sum() == 0:
                    continue

                optimizer.zero_grad()
                pred = model(x, edge_index_dev, edge_attr_dev)
                loss, _, _ = criterion(pred, y, vm)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / max(n_batches, 1)

            # Evaluate:
            seen_metrics_epoch = evaluate_fold(model, full_dataset, lambda b: get_mask(b, seen_sids), device)
            seen_val = seen_metrics_epoch["val_mae_tmax"]
            
            if use_seen_for_stopping:
                monitor_val = seen_val
                unseen_val = float('nan')
                unseen_val_str = "N/A"
            else:
                unseen_metrics_epoch = evaluate_fold(model, full_dataset, lambda b: get_mask(b, unseen_sids), device)
                unseen_val = unseen_metrics_epoch["val_mae_tmax"]
                monitor_val = unseen_val
                unseen_val_str = f"{monitor_val:.4f}"

            if PROGRESS_ENABLED:
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "seen": f"{seen_val:.4f}",
                    "patience": f"{patience_counter}/{cfg.train.patience}",
                }
                if m > 0:
                    postfix["unseen"] = unseen_val_str
                epoch_bar.set_postfix(**postfix)
            elif epoch % 10 == 0:
                print(
                    f"    Epoch {epoch:4d}/{args.epochs}"
                    f" | Loss: {avg_loss:.4f}"
                    f" | Monitor MAE: {monitor_val:.4f}"
                    f" | Patience: {patience_counter}/{cfg.train.patience}",
                    flush=True,
                )

            # Record curves
            last_train_loss = avg_loss
            last_seen_mae = seen_val
            last_unseen_mae = unseen_val if m > 0 else seen_val
            
            epoch_curves["train_loss"][epoch-1].append(last_train_loss)
            epoch_curves["seen_mae"][epoch-1].append(last_seen_mae)
            epoch_curves["unseen_mae"][epoch-1].append(last_unseen_mae)
            epoch_real_counts[epoch-1] += 1

            if not np.isnan(monitor_val) and monitor_val < best_unseen_tmax:
                best_unseen_tmax = monitor_val
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.train.patience:
                progress_message(f"    Early stopping triggered after {epoch} epochs.")
                # Pad the rest of the epochs with the final sustained value
                for future_epoch in range(epoch, args.epochs):
                    epoch_curves["train_loss"][future_epoch].append(last_train_loss)
                    epoch_curves["seen_mae"][future_epoch].append(last_seen_mae)
                    epoch_curves["unseen_mae"][future_epoch].append(last_unseen_mae)
                break

        # 7. Final Evaluation on Best Checkpoint for this iteration
        model.load_state_dict(best_state)
        # Save unique checkpoint
        ckpt_path = os.path.join(cfg.train.checkpoint_dir, f"best_model_withholding_m{m}_iter{it}.pt")
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        torch.save(best_state, ckpt_path)

        seen_metrics = evaluate_fold(model, full_dataset, lambda b: get_mask(b, seen_sids), device)
        unseen_metrics = evaluate_fold(model, full_dataset, lambda b: get_mask(b, unseen_sids), device)
        
        iter_best_seen.append(seen_metrics['val_mae_tmax'])
        if m > 0:
            iter_best_unseen.append(unseen_metrics['val_mae_tmax'])
        iter_baseline.append(unseen_metrics.get("baseline_mae_tmax", seen_metrics["baseline_mae_tmax"]))

        if PROGRESS_ENABLED:
            iteration_bar.set_postfix(
                seen=f"{seen_metrics['val_mae_tmax']:.4f}",
                unseen=f"{unseen_metrics['val_mae_tmax']:.4f}" if m > 0 else "N/A",
            )

        summary = f"    Iter {it+1} Results -> Seen MAE: {seen_metrics['val_mae_tmax']:.4f}"
        if m > 0:
            summary += f" | Unseen MAE: {unseen_metrics['val_mae_tmax']:.4f}"
        progress_message(summary)

    # Calculate Aggregated Stats
    mean_seen = np.mean(iter_best_seen)
    mean_baseline = np.mean(iter_baseline)
    if m > 0:
        mean_unseen = np.mean(iter_best_unseen)
        std_unseen = np.std(iter_best_unseen)
    else:
        mean_unseen = float('nan')
        std_unseen = 0.0
        
    progress_message("\n  ========================================")
    progress_message("  AGGREGATED RESULTS (Across all iterations)")
    progress_message(f"  SEEN   mean MAE Tmax: {mean_seen:.4f}")
    if m > 0:
        progress_message(f"  UNSEEN mean MAE Tmax: {mean_unseen:.4f} ± {std_unseen:.4f}")
        progress_message(f"  Degradation: {mean_unseen - mean_seen:.4f} °C")
    progress_message("  ========================================")
    
    # Save the averaged learning curves for plotting
    avg_curves = {
        "train_loss": [np.mean(vals) if vals else float('nan') for vals in epoch_curves["train_loss"]],
        "seen_mae": [np.mean(vals) if vals else float('nan') for vals in epoch_curves["seen_mae"]],
        "unseen_mae": [np.mean(vals) if vals else float('nan') for vals in epoch_curves["unseen_mae"]],
    }
    
    # Stop plotting at the last epoch that was actually trained by at least one run.
    valid_len = max((i + 1 for i, count in enumerate(epoch_real_counts) if count > 0), default=0)
    avg_curves = {k: v[:valid_len] for k, v in avg_curves.items()}

    # Print out curve data so plot_results.py can parse it
    # We prefix with a marker so plot_results.py can easily find it via regex
    progress_message("\n  [AGGREGATED_CURVES_START]")
    for e in range(valid_len):
        if m == 0:
            progress_message(
                f"    Epoch {e+1}:  [avg_train_loss={avg_curves['train_loss'][e]:.4f}, "
                f"avg_seen_mae={avg_curves['seen_mae'][e]:.4f}]"
            )
        else:
            progress_message(
                f"    Epoch {e+1}:  [avg_train_loss={avg_curves['train_loss'][e]:.4f}, "
                f"avg_seen_mae={avg_curves['seen_mae'][e]:.4f}, "
                f"avg_unseen_mae={avg_curves['unseen_mae'][e]:.4f}]"
            )
    progress_message("  [AGGREGATED_CURVES_END]\n")

    return {
        "m": m,
        "seen_mae_tmax": mean_seen,
        "unseen_mae_tmax": mean_unseen,
        "unseen_mae_std": std_unseen,
        "baseline_tmax": mean_baseline
    }



def main():
    parser = argparse.ArgumentParser("Station Withholding Degradation Run")
    parser.add_argument("--data_path", type=str, default=cfg.train.data_path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=cfg.train.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.train.weight_decay)
    parser.add_argument("--train_years", type=int, default=cfg.train.train_years,
                        help="Use only the most recent N pre-test years for train/validation. "
                             "Set <= 0 to use all available years before test_year.")
    parser.add_argument("--model_type", type=str, default=cfg.model.model_type,
                        help="Model variant: baseline or multi_channel")
    parser.add_argument("--num_channels", type=int, default=cfg.model.num_channels,
                        help="Parallel attention channels (multi_channel only)")
    parser.add_argument("--aggregation", type=str, default=cfg.model.aggregation,
                        help="Channel aggregation: mean or concat")
    parser.add_argument("--active_channels", type=str, default=cfg.model.active_channels,
                        help="Active channel names for multi_channel model. "
                             "Comma-separated subset or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m_values", type=int, nargs="+", default=[1, 3, 5, 10, 15],
                        help="Number of stations to withhold")
    args = parser.parse_args()
    cfg.train.train_years = args.train_years
    cfg.model.model_type = args.model_type
    cfg.model.num_channels = args.num_channels
    cfg.model.aggregation = args.aggregation
    cfg.model.active_channels = args.active_channels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path)
    df = standardize_input_columns(df)
    df["time"] = pd.to_datetime(df["time"])
    trainval_df, _ = temporal_split(df, test_year=cfg.split.test_year)
    trainval_df, selected_years = restrict_train_years(trainval_df, n_years=cfg.train.train_years)
    print(f"Training years used: {selected_years}")
    
    unique_stations = trainval_df[["station", "lat", "lon", "height"]].drop_duplicates("station").reset_index(drop=True)
    edge_index, edge_attr, station_order = build_static_graph(unique_stations, k=cfg.graph.k)
    edge_attr, _ = normalize_edge_attr(edge_attr)

    results = []
    # Always include baseline (m=0, train on all, evaluate on all)
    m_list = [0] + args.m_values
    
    m_bar = tqdm(
        m_list,
        desc="Withholding runs",
        unit="config",
        leave=True,
        dynamic_ncols=True,
        disable=not PROGRESS_ENABLED,
    )

    for m in m_bar:
        if m >= len(unique_stations):
            progress_message(
                f"Skipping m={m}, cannot withhold more than {len(unique_stations)} stations."
            )
            continue
        res = run_withholding_experiment(
            m, trainval_df, unique_stations, edge_index, edge_attr, station_order, device, args
        )
        results.append(res)
        if PROGRESS_ENABLED:
            m_bar.set_postfix(m=m, seen=f"{res['seen_mae_tmax']:.4f}")
        
    progress_message("\n" + "="*50)
    progress_message("  FINAL DEGRADATION CURVE")
    progress_message("="*50)
    progress_message("  m\tSeen MAE\tUnseen MAE\tBaseline")
    for r in results:
        m = r["m"]
        s = r["seen_mae_tmax"]
        u = r["unseen_mae_tmax"]
        b = r["baseline_tmax"]
        progress_message(f"  {m}\t{s:.4f}\t\t{u:.4f}\t\t{b:.4f}")

if __name__ == "__main__":
    main()
