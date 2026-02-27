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

from src.config import cfg
from src.data.dataset import ERA5LandDataset, fit_scaler
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.split import temporal_split
from src.models.mpt import OffsetMPT
from src.utils.loss import OffsetLoss
from src.train import evaluate_fold


def run_withholding_experiment(m, trainval_df, unique_stations, edge_index, edge_attr, device, args):
    """Run one withholding experiment holding out `m` random stations."""
    print(f"\n" + "="*50)
    print(f"  Experiment: Withholding m={m} stations")
    print("="*50)

    # 1. Randomly split stations into SEEN (train) and UNSEEN (holdout)
    all_sids = unique_stations["station"].tolist()
    random.seed(args.seed + m) # different seed per runs
    
    unseen_sids = set(random.sample(all_sids, m))
    seen_sids = set(all_sids) - unseen_sids
    
    print(f"  Unseen stations ({m}): {unseen_sids}")

    # 2. Fit scaler ONLY on seen (training) stations
    train_df = trainval_df[trainval_df["station"].isin(seen_sids)]
    scaler_ds = ERA5LandDataset(train_df, scaler=None)
    scaler = fit_scaler(scaler_ds)

    # 3. Create full dataset for inference/loss
    full_dataset = ERA5LandDataset(trainval_df, scaler=scaler)
    full_dataset.edge_index = edge_index
    full_dataset.edge_attr = edge_attr

    # 4. Define Masks
    def get_mask(batch, sids_set):
        # returns (train_mask, val_mask) based on the supplied set
        mask = np.array([sid in sids_set for sid in batch["station_ids"]])
        return mask, mask # when evaluating seen, we evaluate on the training set

    # 5. Model setup
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
    criterion = OffsetLoss(lambda_tmax=cfg.loss.lambda_tmax, lambda_tmin=cfg.loss.lambda_tmin)
    
    edge_index_dev = edge_index.to(device)
    edge_attr_dev = edge_attr.to(device)

    # 6. Train with early stopping on UNSEEN (validation)
    best_unseen_tmax = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in range(len(full_dataset)):
            batch = full_dataset[idx]
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            
            # Training only uses SEEN nodes
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

        # Evaluate every epoch
        unseen_metrics = evaluate_fold(model, full_dataset, lambda b: get_mask(b, unseen_sids), device)
        
        if unseen_metrics["val_mae_tmax"] < best_unseen_tmax:
            best_unseen_tmax = unseen_metrics["val_mae_tmax"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= cfg.train.patience:
            break

    # 7. Final Evaluation on Best Checkpoint
    model.load_state_dict(best_state)
    seen_metrics = evaluate_fold(model, full_dataset, lambda b: get_mask(b, seen_sids), device)
    unseen_metrics = evaluate_fold(model, full_dataset, lambda b: get_mask(b, unseen_sids), device)
    
    print("\n  --- Final Results ---")
    print(f"  SEEN   (Train) MAE Tmax: {seen_metrics['val_mae_tmax']:.4f}")
    if m > 0:
        print(f"  UNSEEN (Val)   MAE Tmax: {unseen_metrics['val_mae_tmax']:.4f}")
        print(f"  Degradation: {unseen_metrics['val_mae_tmax'] - seen_metrics['val_mae_tmax']:.4f} °C")
    
    return {
        "m": m,
        "seen_mae_tmax": seen_metrics["val_mae_tmax"],
        "unseen_mae_tmax": unseen_metrics.get("val_mae_tmax", float("nan")),
        "baseline_tmax": unseen_metrics.get("baseline_mae_tmax", seen_metrics["baseline_mae_tmax"])
    }


def main():
    parser = argparse.ArgumentParser("Station Withholding Degradation Run")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=cfg.train.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.train.weight_decay)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m_values", type=int, nargs="+", default=[1, 3, 5, 10, 15],
                        help="Number of stations to withhold")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(cfg.train.data_path)
    df["time"] = pd.to_datetime(df["time"])
    trainval_df, _ = temporal_split(df, test_year=cfg.split.test_year)
    
    unique_stations = trainval_df[["station", "lat", "lon", "height"]].drop_duplicates("station").reset_index(drop=True)
    edge_index, edge_attr, _ = build_static_graph(unique_stations, k=cfg.graph.k)
    edge_attr, _ = normalize_edge_attr(edge_attr)

    results = []
    # Always include baseline (m=0, train on all, evaluate on all)
    m_list = [0] + args.m_values
    
    for m in m_list:
        if m >= len(unique_stations):
            print(f"Skipping m={m}, cannot withhold more than {len(unique_stations)} stations.")
            continue
        res = run_withholding_experiment(m, trainval_df, unique_stations, edge_index, edge_attr, device, args)
        results.append(res)
        
    print("\n" + "="*50)
    print("  FINAL DEGRADATION CURVE")
    print("="*50)
    print("  m\tSeen MAE\tUnseen MAE\tBaseline")
    for r in results:
        m = r["m"]
        s = r["seen_mae_tmax"]
        u = r["unseen_mae_tmax"]
        b = r["baseline_tmax"]
        print(f"  {m}\t{s:.4f}\t\t{u:.4f}\t\t{b:.4f}")

if __name__ == "__main__":
    main()
