#!/usr/bin/env python3
"""
Prepare a provisional fog-study dataset from the station/ERA5 comparison CSV.

This script only maps fields whose meanings are already sufficiently clear.
Columns with unresolved semantics, especially visibility and present-weather
codes, are preserved as raw review fields instead of being forced into final
label columns prematurely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_TO_CANONICAL = {
    "T_station": "t2m_obs",
    "TD_station": "td2m_obs",
    "UG_station": "rh_obs",
    "P_station": "surface_pressure_obs",
    "FH_station": "wind_speed_obs",
    "DD_station": "wind_dir_obs",
    "era5_sp": "surface_pressure",
    "era5_tp": "tp",
    "era5_tcc": "tcc",
    "era5_blh": "blh",
}


PROVISIONAL_COLUMNS = [
    "station",
    "time",
    "stationname",
    "lat",
    "lon",
    "height",
    "t2m_obs",
    "era5_t2m",
    "td2m_obs",
    "era5_d2m",
    "rh_obs",
    "era5_UG",
    "surface_pressure_obs",
    "surface_pressure",
    "wind_speed_obs",
    "era5_ws10",
    "wind_dir_obs",
    "era5_u10",
    "era5_v10",
    "RH_station",
    "tp",
    "N_station",
    "tcc",
    "blh",
    "era5_ssrd",
    "era5_strd",
    "VV_station_raw",
    "W1_station_raw",
    "WW_station_raw",
    "visibility_m",
    "present_weather_code",
    "fog_label",
    "low_visibility_label",
    "visibility_class",
]


def build_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df[["station", "stationname", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .copy()
    )
    coverage = (
        df.groupby("station", as_index=False)["time"]
        .agg(start_time_utc="min", end_time_utc="max")
    )
    meta = meta.merge(coverage, on="station", how="left")
    meta = meta.rename(columns={"stationname": "name"})
    meta["country"] = "Netherlands"
    meta["station_type"] = pd.NA
    return meta[
        [
            "station",
            "name",
            "lat",
            "lon",
            "height",
            "country",
            "station_type",
            "start_time_utc",
            "end_time_utc",
        ]
    ].sort_values("station")


def build_station_manifest(df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby("station", as_index=False)["time"]
        .agg(start_date="min", end_date="max", n_rows="count")
    )
    manifest = (
        df[["station", "stationname", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .merge(coverage, on="station", how="left")
        .sort_values("station")
    )
    return manifest


def build_provisional_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=False)
    out = out.rename(columns=RAW_TO_CANONICAL)

    # Preserve raw visibility and weather-code candidates without assuming units
    # or codebook semantics until the data preparer confirms them.
    out["VV_station_raw"] = out["VV_station"]
    out["W1_station_raw"] = out["W1_station"]
    out["WW_station_raw"] = out["WW_station"]

    out["visibility_m"] = pd.NA
    out["present_weather_code"] = pd.NA
    out["fog_label"] = pd.NA
    out["low_visibility_label"] = pd.NA
    out["visibility_class"] = pd.NA

    return out[PROVISIONAL_COLUMNS].sort_values(["station", "time"]).reset_index(drop=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Prepare provisional fog-study dataset outputs.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default=str(repo_root / "datasets/provisional_fog_1997"))
    parser.add_argument(
        "--manifest-path",
        default=str(repo_root / "baseline_assets/netherlands_station_manifest_1997.csv"),
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    provisional = build_provisional_dataset(df)
    station_meta = build_station_metadata(df.assign(time=pd.to_datetime(df["time"])))
    manifest = build_station_manifest(df.assign(time=pd.to_datetime(df["time"])))

    provisional_path = output_dir / "fog_training_merged_hourly_provisional_1997.csv"
    station_meta_path = output_dir / "station_metadata_1997.csv"
    review_path = output_dir / "label_review_candidates_1997.csv"

    provisional.to_csv(provisional_path, index=False)
    station_meta.to_csv(station_meta_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    provisional[
        ["station", "time", "VV_station_raw", "W1_station_raw", "WW_station_raw"]
    ].to_csv(review_path, index=False)

    print(f"Wrote provisional dataset: {provisional_path}")
    print(f"Wrote station metadata:   {station_meta_path}")
    print(f"Wrote station manifest:   {manifest_path}")
    print(f"Wrote label review CSV:   {review_path}")


if __name__ == "__main__":
    main()
