"""
inference.py — Run trained ERA5 Offset MPT on a date range and output corrected temperatures.

Loads:
  - Centralized config (src/config.py)
  - Model checkpoint (best_model.pt)
  - Node feature scaler (scaler.pkl)
  - Edge scaler (saved inside checkpoint)

Outputs per station per day:
  - era5_tmax / era5_tmin   (raw ERA5 input)
  - pred_offset_tmax / pred_offset_tmin  (model predictions)
  - corrected_tmax / corrected_tmin      (ERA5 + predicted offset)
  - station_tmax / station_tmin          (ground truth, if available)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import pandas as pd
import torch

from src.config import cfg
from src.data.dataset import ERA5LandDataset, load_scaler
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.split import temporal_split
from src.models.mpt import OffsetMPT


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    print("Loading comparison_all_years.csv ...")
    df = pd.read_csv(args.data_path)

    # Filter to the requested date range
    df["time"] = pd.to_datetime(df["time"])
    if args.start_date:
        df = df[df["time"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        df = df[df["time"] <= pd.Timestamp(args.end_date)]

    if df.empty:
        print("No data found for the specified date range.")
        return

    # 2. Load node scaler + edge scaler from checkpoint
    scaler = load_scaler(args.scaler_path)
    checkpoint = torch.load(args.model_path, map_location=device)
    edge_scaler = checkpoint.get("edge_scaler", None)

    # 3. Build static graph from station metadata in this date range
    unique_stations = (
        df[["station", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )
    edge_index, edge_attr, station_order = build_static_graph(
        unique_stations, k=cfg.graph.k
    )
    if edge_scaler is not None:
        edge_attr, _ = normalize_edge_attr(edge_attr, edge_scaler)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # 4. Dataset with normalization applied
    dataset = ERA5LandDataset(df, scaler=scaler)

    # 5. Load model
    model = OffsetMPT(
        in_features=cfg.model.in_features,
        hidden_dim=cfg.model.hidden_dim,
        heads=cfg.model.heads,
        num_gnn_layers=cfg.model.num_gnn_layers,
        edge_dim=cfg.model.edge_dim,
        out_dim=cfg.model.out_dim,
        dropout=0.0,   # no dropout at inference
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # 6. Run inference day by day
    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            x = batch["x"].to(device)
            y = batch["y"]            # ground truth offsets (may contain 0 for NaN)
            vm = batch["valid_mask"]  # (N, 2) bool
            station_ids = batch["station_ids"]
            date = batch["date"]

            # Raw ERA5 values (before normalization) — re-fetch from original df
            day_df = df[df["time"] == pd.Timestamp(date)].reset_index(drop=True)

            pred = model(x, edge_index, edge_attr).cpu()  # (N, 2)

            for i, sid in enumerate(station_ids):
                row = {
                    "date": date,
                    "station": sid,
                    "era5_tmax": day_df.loc[i, "mx2t"] if i < len(day_df) else np.nan,
                    "era5_tmin": day_df.loc[i, "mn2t"] if i < len(day_df) else np.nan,
                    "pred_offset_tmax": pred[i, 0].item(),
                    "pred_offset_tmin": pred[i, 1].item(),
                    "station_tmax": day_df.loc[i, "TX"] if i < len(day_df) else np.nan,
                    "station_tmin": day_df.loc[i, "TN"] if i < len(day_df) else np.nan,
                }
                row["corrected_tmax"] = row["era5_tmax"] + row["pred_offset_tmax"]
                row["corrected_tmin"] = row["era5_tmin"] + row["pred_offset_tmin"]
                records.append(row)

    results = pd.DataFrame(records)

    # 7. Compute and print summary metrics
    valid_tmax = results["station_tmax"].notna() & results["era5_tmax"].notna()
    valid_tmin = results["station_tmin"].notna() & results["era5_tmin"].notna()

    if valid_tmax.any():
        baseline_tmax = (results.loc[valid_tmax, "era5_tmax"] - results.loc[valid_tmax, "station_tmax"]).abs().mean()
        corrected_tmax = (results.loc[valid_tmax, "corrected_tmax"] - results.loc[valid_tmax, "station_tmax"]).abs().mean()
        print(f"\nTmax  | Baseline MAE (ERA5): {baseline_tmax:.4f}°C | Corrected MAE: {corrected_tmax:.4f}°C")

    if valid_tmin.any():
        baseline_tmin = (results.loc[valid_tmin, "era5_tmin"] - results.loc[valid_tmin, "station_tmin"]).abs().mean()
        corrected_tmin = (results.loc[valid_tmin, "corrected_tmin"] - results.loc[valid_tmin, "station_tmin"]).abs().mean()
        print(f"Tmin  | Baseline MAE (ERA5): {baseline_tmin:.4f}°C | Corrected MAE: {corrected_tmin:.4f}°C")

    # 8. Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results.to_csv(args.output_path, index=False)
    print(f"\nResults saved → {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5 Offset MPT Inference")
    parser.add_argument("--data_path", type=str, default=cfg.train.data_path)
    parser.add_argument("--model_path", type=str, default=os.path.join(cfg.train.checkpoint_dir, "best_model.pt"))
    parser.add_argument("--scaler_path", type=str, default=cfg.train.scaler_path)
    parser.add_argument("--output_path", type=str, default="d:/Offset Prediction Research/outputs/inference_results.csv")
    parser.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD start date filter")
    parser.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD end date filter")
    args = parser.parse_args()
    inference(args)
