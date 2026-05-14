#!/usr/bin/env python3
"""
Build a trainable daily offset dataset from the hourly `era5_merged.csv` export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.hourly_to_daily import build_trainable_daily_dataset, load_hourly_frame


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Convert hourly era5_merged.csv into a trainable daily dataset."
    )
    parser.add_argument(
        "--input-csv",
        default=str(repo_root / "era5_merged.csv"),
        help="Hourly merged station/ERA5 CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(repo_root / "datasets/fog_ready/era5_trainable_daily.csv"),
        help="Output daily training CSV compatible with src/train.py.",
    )
    parser.add_argument(
        "--min-hours-per-day",
        type=int,
        default=20,
        help="Drop station-days with fewer than this many hourly rows before aggregation.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading hourly data: {input_csv}")
    hourly = load_hourly_frame(input_csv)
    print(f"Loaded {len(hourly):,} rows across {hourly['station'].nunique()} stations.")

    daily = build_trainable_daily_dataset(hourly, min_hours_per_day=args.min_hours_per_day)
    daily.to_csv(output_csv, index=False)

    print(f"Wrote daily training dataset: {output_csv}")
    print(f"Rows: {len(daily):,}")
    print(f"Time span: {daily['time'].min()} -> {daily['time'].max()}")
    print(
        "Fog labels present on "
        f"{int(daily['fog_label'].notna().sum()):,} rows; "
        f"positive fog days: {int((daily['fog_label'] == 1).sum()):,}"
    )
    print(
        "Training command:"
        f" python src/train.py --data_path {output_csv} --cv_mode slobo"
        " --model_type multi_channel"
        " --active_channels temperature,humidity_stability,wind"
        " --enable_fog_head"
    )


if __name__ == "__main__":
    main()
