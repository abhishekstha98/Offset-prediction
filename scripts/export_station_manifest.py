#!/usr/bin/env python3
"""
Export the Netherlands station manifest used by this repo.

This is intended for external-baseline work such as GraphCast / Aurora
interpolation and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    station_meta = (
        df[["station", "stationname", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .copy()
    )

    coverage = (
        df.groupby("station")["time"]
        .agg(start_date="min", end_date="max", n_rows="count")
        .reset_index()
    )

    manifest = (
        station_meta.merge(coverage, on="station", how="left")
        .sort_values("station")
        .reset_index(drop=True)
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export station manifest for external baseline evaluation.")
    parser.add_argument("--data-path", default="merged.csv", help="Path to merged station/ERA5 CSV.")
    parser.add_argument(
        "--output-path",
        default="baseline_assets/netherlands_station_manifest.csv",
        help="Where to write the station manifest CSV.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.output_path)

    df = pd.read_csv(data_path, usecols=["station", "stationname", "lat", "lon", "height", "time"])
    df["time"] = pd.to_datetime(df["time"])

    manifest = build_manifest(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print(f"Wrote {len(manifest)} stations to {out_path}")
    print(
        manifest[
            ["station", "stationname", "lat", "lon", "height", "start_date", "end_date", "n_rows"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
