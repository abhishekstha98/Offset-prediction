#!/usr/bin/env python3
"""
Evaluate an external gridded baseline (for example GraphCast or Aurora)
against the repo's station observations after spatial interpolation.

Supported modes:
  1. subdaily_t2m:
     The external file contains a sub-daily 2m temperature field.
     The script interpolates to station locations and aggregates to daily
     Tmax / Tmin by day.

  2. daily_tmax_tmin:
     The external file already contains daily Tmax / Tmin fields.

  3. csv_daily_tmax_tmin:
     The external file is already a station-level CSV with daily Tmax / Tmin
     predictions and station/date identifiers.

Expected output:
  - matched daily station predictions CSV
  - per-station MAE CSV
  - summary JSON

This script does not run GraphCast or Aurora itself. It evaluates exported
gridded outputs once they exist on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import standardize_input_columns


COMMON_TIME_NAMES = ("time", "valid_time", "datetime")
COMMON_LAT_NAMES = ("latitude", "lat")
COMMON_LON_NAMES = ("longitude", "lon")


def _import_xarray():
    try:
        import xarray as xr  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "xarray is required for scripts/evaluate_gridded_baseline.py.\n"
            "Install it in the environment you will use for GraphCast/Aurora evaluation, for example:\n"
            "  pip install xarray netcdf4 scipy\n"
            "If you are reading GRIB files, also install cfgrib."
        ) from exc
    return xr


def detect_name(dataset, explicit: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if explicit:
        if explicit in dataset.coords or explicit in dataset.dims or explicit in dataset.variables:
            return explicit
        raise KeyError(f"Explicit {kind} name '{explicit}' not found in dataset.")

    for name in candidates:
        if name in dataset.coords or name in dataset.dims or name in dataset.variables:
            return name
    raise KeyError(f"Could not detect {kind} name. Tried: {candidates}")


def open_dataset(path: Path, engine: str | None):
    xr = _import_xarray()
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    if engine:
        return xr.open_dataset(path, engine=engine)
    return xr.open_dataset(path)


def maybe_convert_station_lon(lons: np.ndarray, dataset_lon) -> np.ndarray:
    lon_values = np.asarray(dataset_lon.values)
    if lon_values.min() >= 0 and np.nanmax(lons) <= 180:
        return np.where(lons < 0, lons + 360.0, lons)
    return lons


def build_station_manifest(target_df: pd.DataFrame) -> pd.DataFrame:
    return (
        target_df[["station", "stationname", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .sort_values("station")
        .reset_index(drop=True)
    )


def ensure_2d_time_station(da, time_name: str):
    da = da.squeeze(drop=True)
    if time_name not in da.dims:
        raise ValueError(f"Expected time dimension '{time_name}' in interpolated array, got {da.dims}")
    if "station" not in da.dims:
        raise ValueError(f"Expected 'station' dimension after interpolation, got {da.dims}")
    extra_dims = [d for d in da.dims if d not in {time_name, "station"}]
    if extra_dims:
        raise ValueError(
            "Prediction array still has extra dimensions after squeeze/interp: "
            f"{extra_dims}. Export a simpler dataset first."
        )
    return da.transpose(time_name, "station")


def interpolate_field(
    da,
    station_df: pd.DataFrame,
    lat_name: str,
    lon_name: str,
    method: str,
) :
    xr = _import_xarray()
    station_lats = xr.DataArray(station_df["lat"].to_numpy(), dims="station")
    station_lons = xr.DataArray(
        maybe_convert_station_lon(station_df["lon"].to_numpy(), da[lon_name]),
        dims="station",
    )
    return da.interp({lat_name: station_lats, lon_name: station_lons}, method=method)


def attach_dates(frame: pd.DataFrame, time_col: str, timezone: str | None) -> pd.DataFrame:
    ts = pd.to_datetime(frame[time_col], utc=True)
    if timezone:
        frame["date"] = ts.dt.tz_convert(timezone).dt.date
    else:
        frame["date"] = ts.dt.date
    return frame


def evaluate_predictions(pred_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_eval = target_df[
        ["station", "date", "TX", "TN", "mx2t", "mn2t", "stationname", "lat", "lon", "height"]
    ].copy()

    matched = pred_df.merge(target_eval, on=["station", "date"], how="inner")
    if matched.empty:
        raise ValueError("No overlapping station/date rows between predictions and target CSV.")

    matched["abs_err_tmax"] = (matched["pred_tmax"] - matched["TX"]).abs()
    matched["abs_err_tmin"] = (matched["pred_tmin"] - matched["TN"]).abs()
    matched["abs_err_era5_tmax"] = (matched["mx2t"] - matched["TX"]).abs()
    matched["abs_err_era5_tmin"] = (matched["mn2t"] - matched["TN"]).abs()

    station_summary = (
        matched.groupby(["station", "stationname"], dropna=False)
        .agg(
            n_days=("date", "count"),
            mae_tmax=("abs_err_tmax", "mean"),
            mae_tmin=("abs_err_tmin", "mean"),
            era5_mae_tmax=("abs_err_era5_tmax", "mean"),
            era5_mae_tmin=("abs_err_era5_tmin", "mean"),
        )
        .reset_index()
        .sort_values("station")
    )

    summary = {
        "n_matched_rows": int(len(matched)),
        "n_stations": int(matched["station"].nunique()),
        "date_min": str(pd.to_datetime(matched["date"]).min().date()),
        "date_max": str(pd.to_datetime(matched["date"]).max().date()),
        "mae_tmax": float(matched["abs_err_tmax"].mean()),
        "mae_tmin": float(matched["abs_err_tmin"].mean()),
        "era5_mae_tmax_on_same_subset": float(matched["abs_err_era5_tmax"].mean()),
        "era5_mae_tmin_on_same_subset": float(matched["abs_err_era5_tmin"].mean()),
    }
    return matched, station_summary, summary


def load_station_csv_predictions(
    prediction_path: Path,
    station_col: str,
    date_col: str,
    pred_tmax_col: str,
    pred_tmin_col: str,
) -> pd.DataFrame:
    pred_df = pd.read_csv(prediction_path)
    required = [station_col, date_col, pred_tmax_col, pred_tmin_col]
    missing = [col for col in required if col not in pred_df.columns]
    if missing:
        raise ValueError(
            f"CSV prediction file is missing required columns: {missing}. "
            f"Available columns: {pred_df.columns.tolist()}"
        )

    pred_df = pred_df.rename(
        columns={
            station_col: "station",
            date_col: "date",
            pred_tmax_col: "pred_tmax",
            pred_tmin_col: "pred_tmin",
        }
    )
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.date
    return pred_df[["station", "date", "pred_tmax", "pred_tmin"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphCast/Aurora-style gridded outputs at station level.")
    parser.add_argument(
        "--prediction-path",
        required=True,
        help="Path to NetCDF/Zarr with gridded predictions, or a CSV for csv_daily_tmax_tmin mode.",
    )
    parser.add_argument("--source-model", required=True, help="Short label, e.g. graphcast or aurora.")
    parser.add_argument("--target-csv", default="merged.csv", help="Path to the repo's target CSV.")
    parser.add_argument(
        "--mode",
        choices=("subdaily_t2m", "daily_tmax_tmin", "csv_daily_tmax_tmin"),
        required=True,
        help="How to interpret the prediction file.",
    )
    parser.add_argument("--temp-var", default=None, help="Variable name for sub-daily 2m temperature.")
    parser.add_argument("--tmax-var", default=None, help="Variable name for daily Tmax.")
    parser.add_argument("--tmin-var", default=None, help="Variable name for daily Tmin.")
    parser.add_argument("--time-name", default=None, help="Time coordinate name if not auto-detected.")
    parser.add_argument("--lat-name", default=None, help="Latitude coordinate name if not auto-detected.")
    parser.add_argument("--lon-name", default=None, help="Longitude coordinate name if not auto-detected.")
    parser.add_argument("--station-col", default="station", help="Station column name for csv_daily_tmax_tmin mode.")
    parser.add_argument("--date-col", default="date", help="Date column name for csv_daily_tmax_tmin mode.")
    parser.add_argument("--pred-tmax-col", default="pred_tmax", help="Predicted Tmax column for csv_daily_tmax_tmin mode.")
    parser.add_argument("--pred-tmin-col", default="pred_tmin", help="Predicted Tmin column for csv_daily_tmax_tmin mode.")
    parser.add_argument("--engine", default=None, help="Optional xarray engine, e.g. netcdf4 or cfgrib.")
    parser.add_argument("--interp", default="linear", choices=("linear", "nearest"), help="Interpolation method.")
    parser.add_argument("--timezone", default="Europe/Amsterdam", help="Timezone used for daily aggregation.")
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD start date.")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD end date.")
    parser.add_argument(
        "--kelvin",
        action="store_true",
        help="Interpret prediction temperatures as Kelvin and convert to Celsius.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write evaluation outputs. Defaults to outputs/external_baselines/<source-model>/",
    )
    args = parser.parse_args()

    prediction_path = Path(args.prediction_path)
    output_dir = Path(args.output_dir or f"outputs/external_baselines/{args.source_model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_csv(args.target_csv)
    target_df = standardize_input_columns(target_df)
    target_df["time"] = pd.to_datetime(target_df["time"])
    target_df["date"] = target_df["time"].dt.date

    if args.start_date:
        target_df = target_df[target_df["time"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        target_df = target_df[target_df["time"] <= pd.Timestamp(args.end_date)]

    station_df = build_station_manifest(target_df)

    if args.mode == "csv_daily_tmax_tmin":
        pred_df = load_station_csv_predictions(
            prediction_path=prediction_path,
            station_col=args.station_col,
            date_col=args.date_col,
            pred_tmax_col=args.pred_tmax_col,
            pred_tmin_col=args.pred_tmin_col,
        )
        if args.start_date:
            pred_df = pred_df[pd.to_datetime(pred_df["date"]) >= pd.Timestamp(args.start_date)]
        if args.end_date:
            pred_df = pred_df[pd.to_datetime(pred_df["date"]) <= pd.Timestamp(args.end_date)]
    else:
        ds = open_dataset(prediction_path, args.engine)
        time_name = detect_name(ds, args.time_name, COMMON_TIME_NAMES, "time")
        lat_name = detect_name(ds, args.lat_name, COMMON_LAT_NAMES, "latitude")
        lon_name = detect_name(ds, args.lon_name, COMMON_LON_NAMES, "longitude")

        if args.mode == "subdaily_t2m":
            if not args.temp_var:
                raise ValueError("--temp-var is required in subdaily_t2m mode.")
            temp_da = interpolate_field(ds[args.temp_var], station_df, lat_name, lon_name, args.interp)
            temp_da = ensure_2d_time_station(temp_da, time_name)
            temp_df = temp_da.to_dataframe(name="temperature").reset_index()
            temp_df = attach_dates(temp_df, time_name, args.timezone)
            if args.kelvin:
                temp_df["temperature"] = temp_df["temperature"] - 273.15
            pred_df = (
                temp_df.groupby(["station", "date"], dropna=False)["temperature"]
                .agg(pred_tmax="max", pred_tmin="min")
                .reset_index()
            )
        else:
            if not args.tmax_var or not args.tmin_var:
                raise ValueError("--tmax-var and --tmin-var are required in daily_tmax_tmin mode.")
            tmax_da = interpolate_field(ds[args.tmax_var], station_df, lat_name, lon_name, args.interp)
            tmin_da = interpolate_field(ds[args.tmin_var], station_df, lat_name, lon_name, args.interp)
            tmax_da = ensure_2d_time_station(tmax_da, time_name)
            tmin_da = ensure_2d_time_station(tmin_da, time_name)
            tmax_df = tmax_da.to_dataframe(name="pred_tmax").reset_index()
            tmin_df = tmin_da.to_dataframe(name="pred_tmin").reset_index()
            if args.kelvin:
                tmax_df["pred_tmax"] = tmax_df["pred_tmax"] - 273.15
                tmin_df["pred_tmin"] = tmin_df["pred_tmin"] - 273.15
            tmax_df = attach_dates(tmax_df, time_name, args.timezone)
            tmin_df = attach_dates(tmin_df, time_name, args.timezone)
            pred_df = tmax_df[["station", "date", "pred_tmax"]].merge(
                tmin_df[["station", "date", "pred_tmin"]],
                on=["station", "date"],
                how="inner",
            )

    matched, station_summary, summary = evaluate_predictions(pred_df, target_df)
    summary.update(
        {
            "source_model": args.source_model,
            "prediction_path": str(prediction_path),
            "mode": args.mode,
            "interp": args.interp,
            "timezone": args.timezone,
        }
    )

    matched_path = output_dir / "matched_station_predictions.csv"
    station_summary_path = output_dir / "station_mae.csv"
    summary_path = output_dir / "summary.json"

    matched.to_csv(matched_path, index=False)
    station_summary.to_csv(station_summary_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Matched rows written to {matched_path}")
    print(f"Per-station MAE written to {station_summary_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
