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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import cfg
from src.data.dataset import ERA5LandDataset, load_scaler, standardize_input_columns
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.models.factory import build_model


def _infer_model_config_from_state_dict(state_dict: dict) -> dict:
    """Fallback for older checkpoints that do not save model metadata."""
    first_layer = state_dict["node_encoder.net.0.weight"]
    inferred = {
        "in_features": int(first_layer.shape[1]),
        "hidden_dim": int(first_layer.shape[0]),
        "out_dim": int(state_dict["output_head.net.2.weight"].shape[0]),
        "num_gnn_layers": len(
            {key.split(".")[1] for key in state_dict if key.startswith("conv_layers.")}
        ),
    }

    channel_keys = sorted(
        key for key in state_dict if key.startswith("conv_layers.0.channels.") and key.endswith(".proj_q.weight")
    )
    if channel_keys:
        if inferred["in_features"] == 6:
            dim_to_name = {2: "temperature", 3: "pressure", 1: "terrain"}
        else:
            dim_to_name = {5: "temperature", 8: "humidity_stability", 3: "wind", 1: "terrain"}
        channel_names = []
        for key in channel_keys:
            channel_dim = int(state_dict[key].shape[1])
            channel_names.append(dim_to_name.get(channel_dim, f"dim_{channel_dim}"))
        inferred.update(
            {
                "model_type": "multi_channel",
                "num_channels": len(channel_names),
                "aggregation": "concat",
                "active_channels": ",".join(channel_names),
                "enable_fog_head": False,
                "fog_out_dim": 1,
            }
        )
    else:
        inferred.update(
            {
                "model_type": "baseline",
                "num_channels": cfg.model.num_channels,
                "aggregation": cfg.model.aggregation,
                "active_channels": "all",
                "enable_fog_head": False,
                "fog_out_dim": 1,
            }
        )

    return inferred


def _apply_checkpoint_config(checkpoint: dict, args):
    model_cfg = checkpoint.get("model_config")
    if model_cfg is None:
        model_cfg = _infer_model_config_from_state_dict(checkpoint["model_state_dict"])

    cfg.model.model_type = args.model_type or model_cfg.get("model_type", cfg.model.model_type)
    cfg.model.num_channels = args.num_channels or model_cfg.get("num_channels", cfg.model.num_channels)
    cfg.model.aggregation = args.aggregation or model_cfg.get("aggregation", cfg.model.aggregation)
    cfg.model.active_channels = args.active_channels or model_cfg.get("active_channels", cfg.model.active_channels)
    cfg.model.in_features = model_cfg.get("in_features", cfg.model.in_features)
    cfg.model.hidden_dim = model_cfg.get("hidden_dim", cfg.model.hidden_dim)
    cfg.model.heads = model_cfg.get("heads", cfg.model.heads)
    cfg.model.num_gnn_layers = model_cfg.get("num_gnn_layers", cfg.model.num_gnn_layers)
    cfg.model.sequence_length = args.sequence_length or model_cfg.get("sequence_length", cfg.model.sequence_length)
    cfg.model.temporal_layers = args.temporal_layers or model_cfg.get("temporal_layers", cfg.model.temporal_layers)
    cfg.model.max_seq_len = args.max_seq_len or model_cfg.get("max_seq_len", cfg.model.max_seq_len)
    cfg.model.temporal_pooling = args.temporal_pooling or model_cfg.get("temporal_pooling", cfg.model.temporal_pooling)
    cfg.model.edge_dim = model_cfg.get("edge_dim", cfg.model.edge_dim)
    cfg.model.out_dim = model_cfg.get("out_dim", cfg.model.out_dim)
    cfg.model.enable_fog_head = model_cfg.get("enable_fog_head", cfg.model.enable_fog_head)
    cfg.model.fog_out_dim = model_cfg.get("fog_out_dim", cfg.model.fog_out_dim)

    graph_cfg = checkpoint.get("graph_config", {})
    cfg.graph.k = graph_cfg.get("k", cfg.graph.k)
    return model_cfg


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    print(f"Loading {args.data_path} ...")
    df = pd.read_csv(args.data_path)
    df = standardize_input_columns(df)

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
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_cfg = _apply_checkpoint_config(checkpoint, args)
    if "feature_columns" in model_cfg and "feature_columns" not in scaler:
        scaler["feature_columns"] = model_cfg["feature_columns"]
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
    dataset = ERA5LandDataset(
        df,
        scaler=scaler,
        station_order=station_order,
        sequence_length=cfg.model.sequence_length,
    )

    # 5. Load model
    model = build_model(cfg, dropout_override=0.0).to(device)
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

            if hasattr(model, "forward_multitask"):
                outputs = model.forward_multitask(x, edge_index, edge_attr)
                pred = outputs["offset"].cpu()
                fog_logits = outputs["fog_logits"]
                fog_logits = fog_logits.cpu() if fog_logits is not None else None
            else:
                pred = model(x, edge_index, edge_attr).cpu()  # (N, 2)
                fog_logits = None

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
                if fog_logits is not None:
                    row["fog_logit"] = fog_logits[i, 0].item()
                    row["fog_probability"] = float(torch.sigmoid(fog_logits[i, 0]).item())
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
    parser.add_argument("--output_path", type=str, default=str(Path("outputs") / "inference_results.csv"))
    parser.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD start date filter")
    parser.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD end date filter")
    parser.add_argument("--model_type", type=str, default=None, help="Optional model override for older checkpoints.")
    parser.add_argument("--num_channels", type=int, default=None, help="Optional channel-count override for older checkpoints.")
    parser.add_argument("--aggregation", type=str, default=None, help="Optional aggregation override for older checkpoints.")
    parser.add_argument("--active_channels", type=str, default=None, help="Optional channel-name override for older checkpoints.")
    parser.add_argument("--sequence_length", type=int, default=None, help="Optional sequence length override for older checkpoints.")
    parser.add_argument("--temporal_layers", type=int, default=None, help="Optional temporal layer override for older checkpoints.")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Optional max sequence length override for older checkpoints.")
    parser.add_argument("--temporal_pooling", type=str, default=None, help="Optional temporal pooling override for older checkpoints.")
    parser.add_argument("--enable_fog_head", action="store_true",
                        help="Attach the fog head when loading a compatible checkpoint.")
    parser.add_argument("--fog_out_dim", type=int, default=None,
                        help="Fog head output dimension override for compatible checkpoints.")
    args = parser.parse_args()
    if args.sequence_length is not None:
        cfg.model.sequence_length = args.sequence_length
    if args.temporal_layers is not None:
        cfg.model.temporal_layers = args.temporal_layers
    if args.max_seq_len is not None:
        cfg.model.max_seq_len = args.max_seq_len
    if args.temporal_pooling is not None:
        cfg.model.temporal_pooling = args.temporal_pooling
    if args.enable_fog_head:
        cfg.model.enable_fog_head = True
    if args.fog_out_dim is not None:
        cfg.model.fog_out_dim = args.fog_out_dim
    inference(args)
