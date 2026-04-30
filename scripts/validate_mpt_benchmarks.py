#!/usr/bin/env python3
"""
Validate the station-level MPT against reproducible baselines.

Baselines:
  1. Raw ERA5 daily Tmax/Tmin from merged.csv.
  2. Optional station-level GraphCast CSV predictions.
  3. Optional station-level Aurora CSV predictions.

GraphCast/Aurora are placeholders until exported gridded outputs are available.
Use scripts/evaluate_gridded_baseline.py to bilinearly interpolate gridded model
outputs to station coordinates, then pass the resulting CSVs here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import cfg
from src.data.dataset import ERA5LandDataset, load_scaler, standardize_input_columns
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.models.factory import build_model


def _mae(pred: pd.Series, target: pd.Series) -> float:
    mask = pred.notna() & target.notna()
    if not mask.any():
        return float("nan")
    return float((pred[mask] - target[mask]).abs().mean())


def _filter_dates(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    if start_date:
        df = df[df["time"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["time"] <= pd.Timestamp(end_date)]
    return df


def _load_station_predictions(path: str | None, source: str) -> tuple[pd.DataFrame | None, dict]:
    if not path:
        return None, {
            "status": "placeholder_pending_bilinear_interpolation",
            "expected_csv_columns": ["station", "date", "pred_tmax", "pred_tmin"],
            "source_model": source,
        }

    pred = pd.read_csv(path)
    required = ["station", "date", "pred_tmax", "pred_tmin"]
    missing = [col for col in required if col not in pred.columns]
    if missing:
        raise ValueError(f"{source} prediction CSV missing columns {missing}; found {pred.columns.tolist()}")
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    return pred[required], {"status": "loaded", "source_model": source, "path": path}


def _evaluate_station_prediction_csv(
    pred: pd.DataFrame,
    target_df: pd.DataFrame,
    source: str,
) -> tuple[pd.DataFrame, dict]:
    target = target_df[["station", "date", "TX", "TN"]].copy()
    matched = pred.merge(target, on=["station", "date"], how="inner")
    if matched.empty:
        raise ValueError(f"No matched rows for {source} predictions.")
    matched[f"{source}_abs_err_tmax"] = (matched["pred_tmax"] - matched["TX"]).abs()
    matched[f"{source}_abs_err_tmin"] = (matched["pred_tmin"] - matched["TN"]).abs()
    summary = {
        "status": "evaluated",
        "n_matched_rows": int(len(matched)),
        "n_stations": int(matched["station"].nunique()),
        "mae_tmax": float(matched[f"{source}_abs_err_tmax"].mean()),
        "mae_tmin": float(matched[f"{source}_abs_err_tmin"].mean()),
    }
    return matched, summary


def _apply_checkpoint_config(checkpoint: dict) -> dict:
    model_config = checkpoint.get("model_config", {})
    cfg.model.model_type = model_config.get("model_type", cfg.model.model_type)
    cfg.model.num_channels = model_config.get("num_channels", cfg.model.num_channels)
    cfg.model.aggregation = model_config.get("aggregation", cfg.model.aggregation)
    cfg.model.active_channels = model_config.get("active_channels", cfg.model.active_channels)
    cfg.model.in_features = model_config.get("in_features", cfg.model.in_features)
    cfg.model.hidden_dim = model_config.get("hidden_dim", cfg.model.hidden_dim)
    cfg.model.heads = model_config.get("heads", cfg.model.heads)
    cfg.model.num_gnn_layers = model_config.get("num_gnn_layers", cfg.model.num_gnn_layers)
    cfg.model.sequence_length = model_config.get("sequence_length", cfg.model.sequence_length)
    cfg.model.temporal_layers = model_config.get("temporal_layers", cfg.model.temporal_layers)
    cfg.model.edge_dim = model_config.get("edge_dim", cfg.model.edge_dim)
    cfg.model.out_dim = model_config.get("out_dim", cfg.model.out_dim)
    cfg.graph.k = checkpoint.get("graph_config", {}).get("k", cfg.graph.k)
    return model_config


def _evaluate_mpt(df: pd.DataFrame, model_path: str, scaler_path: str) -> tuple[pd.DataFrame, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = load_scaler(scaler_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    model_config = _apply_checkpoint_config(checkpoint)
    if "feature_columns" in model_config and "feature_columns" not in scaler:
        scaler["feature_columns"] = model_config["feature_columns"]

    unique_stations = (
        df[["station", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )
    edge_index, edge_attr, station_order = build_static_graph(unique_stations, k=cfg.graph.k)
    edge_scaler = checkpoint.get("edge_scaler")
    if edge_scaler is not None:
        edge_attr, _ = normalize_edge_attr(edge_attr, edge_scaler)

    dataset = ERA5LandDataset(
        df,
        scaler=scaler,
        station_order=station_order,
        sequence_length=cfg.model.sequence_length,
    )

    model = build_model(cfg, dropout_override=0.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            pred_offset = model(batch["x"].to(device), edge_index, edge_attr).cpu().numpy()
            date = pd.Timestamp(batch["date"])
            day_df = df[df["time"] == date].copy()
            day_df["_station_order"] = day_df["station"].map(
                {sid: i for i, sid in enumerate(station_order)}
            )
            day_df = day_df.sort_values("_station_order").reset_index(drop=True)

            for row_idx, sid in enumerate(batch["station_ids"]):
                raw = day_df.iloc[row_idx]
                records.append(
                    {
                        "station": sid,
                        "date": date.date(),
                        "mpt_pred_tmax": raw["mx2t"] + pred_offset[row_idx, 0],
                        "mpt_pred_tmin": raw["mn2t"] + pred_offset[row_idx, 1],
                        "TX": raw["TX"],
                        "TN": raw["TN"],
                    }
                )

    pred_df = pd.DataFrame(records)
    summary = {
        "status": "evaluated",
        "model_path": model_path,
        "scaler_path": scaler_path,
        "sequence_length": cfg.model.sequence_length,
        "mae_tmax": _mae(pred_df["mpt_pred_tmax"], pred_df["TX"]),
        "mae_tmin": _mae(pred_df["mpt_pred_tmin"], pred_df["TN"]),
    }
    return pred_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MPT vs ERA5, GraphCast, and Aurora baselines.")
    parser.add_argument("--target-csv", default="merged.csv")
    parser.add_argument("--model-path", default=None, help="Optional trained MPT checkpoint.")
    parser.add_argument("--scaler-path", default=cfg.train.scaler_path)
    parser.add_argument("--graphcast-csv", default=None, help="Optional station-level GraphCast prediction CSV.")
    parser.add_argument("--aurora-csv", default=None, help="Optional station-level Aurora prediction CSV.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmark_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_csv(args.target_csv)
    target_df = standardize_input_columns(target_df)
    target_df = _filter_dates(target_df, args.start_date, args.end_date)
    if target_df.empty:
        raise ValueError("No target rows remain after date filtering.")
    target_df["date"] = target_df["time"].dt.date

    summary = {
        "target_csv": args.target_csv,
        "n_rows": int(len(target_df)),
        "n_stations": int(target_df["station"].nunique()),
        "date_min": str(target_df["time"].min().date()),
        "date_max": str(target_df["time"].max().date()),
        "raw_era5": {
            "status": "evaluated",
            "mae_tmax": _mae(target_df["mx2t"], target_df["TX"]),
            "mae_tmin": _mae(target_df["mn2t"], target_df["TN"]),
        },
    }

    if args.model_path:
        mpt_predictions, summary["mpt"] = _evaluate_mpt(target_df, args.model_path, args.scaler_path)
        mpt_predictions.to_csv(output_dir / "mpt_station_predictions.csv", index=False)
    else:
        summary["mpt"] = {"status": "skipped_no_model_path"}

    for source, path in (("graphcast", args.graphcast_csv), ("aurora", args.aurora_csv)):
        pred, source_summary = _load_station_predictions(path, source)
        if pred is not None:
            matched, source_summary = _evaluate_station_prediction_csv(pred, target_df, source)
            matched.to_csv(output_dir / f"{source}_matched_predictions.csv", index=False)
        summary[source] = source_summary

    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
