#!/usr/bin/env python3
"""
Prepare fog-study experiment datasets from the hourly station/ERA5 merged CSV.

Outputs:
  - canonical hourly fog dataset (CSV)
  - daily evaluation reference table (CSV)
  - station metadata (CSV)
  - station manifest for GraphCast/Aurora interpolation (CSV)

This script uses KNMI hourly dataset semantics:
  - VV: coded horizontal visibility
  - W1: hourly fog indicator (0/1)
  - WW: WMO 4680 present weather code
  - P: sea-level pressure, not station surface pressure
  - RH: hourly precipitation amount

The final visibility value written as `visibility_m` is an approximate midpoint
decoded from the KNMI visibility code bins. The raw code is preserved as
`visibility_code`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "station",
    "time",
    "stationname",
    "lat",
    "lon",
    "height",
    "t2m_obs",
    "td2m_obs",
    "rh_obs",
    "msl_pressure_obs",
    "wind_speed_obs",
    "wind_dir_obs",
    "precip_obs",
    "cloud_cover_obs",
    "visibility_code",
    "visibility_lower_m",
    "visibility_upper_m",
    "visibility_m",
    "present_weather_code",
    "fog_indicator",
    "fog_from_weather_code",
    "fog_label",
    "low_visibility_label",
    "visibility_class",
    "era5_t2m",
    "era5_d2m",
    "era5_UG",
    "surface_pressure",
    "era5_ws10",
    "era5_u10",
    "era5_v10",
    "tp",
    "tcc",
    "blh",
    "era5_ssrd",
    "era5_strd",
]


def decode_visibility_bounds(code: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Decode KNMI VV visibility classes into lower/upper metre bounds.

    Reference:
      0 = <100m
      1..49 = N*100m .. (N+1)*100m
      50 = 5..6km
      56..79 = 6..30km in 1km steps
      80..88 = 30..75km in 5km steps
      89 = >70km
    """
    code_num = pd.to_numeric(code, errors="coerce")
    lower = pd.Series(np.nan, index=code.index, dtype="float64")
    upper = pd.Series(np.nan, index=code.index, dtype="float64")

    mask0 = code_num == 0
    lower.loc[mask0] = 0.0
    upper.loc[mask0] = 100.0

    mask1_49 = code_num.between(1, 49, inclusive="both")
    lower.loc[mask1_49] = code_num.loc[mask1_49] * 100.0
    upper.loc[mask1_49] = (code_num.loc[mask1_49] + 1.0) * 100.0

    mask50 = code_num == 50
    lower.loc[mask50] = 5000.0
    upper.loc[mask50] = 6000.0

    mask56_79 = code_num.between(56, 79, inclusive="both")
    lower.loc[mask56_79] = (code_num.loc[mask56_79] - 50.0) * 1000.0
    upper.loc[mask56_79] = (code_num.loc[mask56_79] - 49.0) * 1000.0

    mask80_88 = code_num.between(80, 88, inclusive="both")
    lower.loc[mask80_88] = 30000.0 + (code_num.loc[mask80_88] - 80.0) * 5000.0
    upper.loc[mask80_88] = 35000.0 + (code_num.loc[mask80_88] - 80.0) * 5000.0

    mask89 = code_num == 89
    lower.loc[mask89] = 70000.0
    upper.loc[mask89] = np.nan

    return lower, upper


def derive_visibility_midpoint(lower: pd.Series, upper: pd.Series) -> pd.Series:
    midpoint = (lower + upper) / 2.0
    midpoint.loc[upper.isna()] = lower.loc[upper.isna()]
    return midpoint


def derive_fog_labels(
    visibility_code: pd.Series,
    fog_indicator: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    visibility_num = pd.to_numeric(visibility_code, errors="coerce")
    fog_indicator_num = pd.to_numeric(fog_indicator, errors="coerce")

    fog_label = pd.Series(pd.NA, index=visibility_code.index, dtype="Float64")
    low_visibility_label = pd.Series(pd.NA, index=visibility_code.index, dtype="Float64")
    visibility_class = pd.Series(pd.NA, index=visibility_code.index, dtype="Float64")

    # Fog label:
    #   1 if explicit fog indicator says fog occurred
    #   0 if explicit fog indicator says no fog occurred
    #   otherwise fallback to visibility-code threshold < 1000m (codes 0..9)
    fog_label.loc[fog_indicator_num == 1] = 1.0
    fog_label.loc[fog_indicator_num == 0] = 0.0
    fallback_mask = fog_indicator_num.isna() & visibility_num.notna()
    fog_label.loc[fallback_mask & (visibility_num <= 9)] = 1.0
    fog_label.loc[fallback_mask & (visibility_num > 9)] = 0.0

    # Low visibility: visibility below 5km corresponds to codes 0..49.
    vis_known = visibility_num.notna()
    low_visibility_label.loc[vis_known & (visibility_num <= 49)] = 1.0
    low_visibility_label.loc[vis_known & (visibility_num >= 50)] = 0.0

    # Visibility class:
    #   3 = <200m
    #   2 = 200m..1000m
    #   1 = 1000m..5000m
    #   0 = >5000m
    visibility_class.loc[vis_known & (visibility_num <= 1)] = 3.0
    visibility_class.loc[vis_known & visibility_num.between(2, 9, inclusive="both")] = 2.0
    visibility_class.loc[vis_known & visibility_num.between(10, 49, inclusive="both")] = 1.0
    visibility_class.loc[vis_known & (visibility_num >= 50)] = 0.0

    return fog_label, low_visibility_label, visibility_class


def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    out["station"] = chunk["station"]
    out["time"] = pd.to_datetime(chunk["time"], utc=True)
    out["stationname"] = chunk["stationname"]
    out["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
    out["height"] = pd.to_numeric(chunk["height"], errors="coerce")

    out["t2m_obs"] = pd.to_numeric(chunk["T_station"], errors="coerce")
    out["td2m_obs"] = pd.to_numeric(chunk["TD_station"], errors="coerce")
    out["rh_obs"] = pd.to_numeric(chunk["UG_station"], errors="coerce")
    out["msl_pressure_obs"] = pd.to_numeric(chunk["P_station"], errors="coerce")
    out["wind_speed_obs"] = pd.to_numeric(chunk["FH_station"], errors="coerce")
    out["wind_dir_obs"] = pd.to_numeric(chunk["DD_station"], errors="coerce")
    out["precip_obs"] = pd.to_numeric(chunk["RH_station"], errors="coerce")
    out["cloud_cover_obs"] = pd.to_numeric(chunk["N_station"], errors="coerce")

    visibility_code = pd.to_numeric(chunk["VV_station"], errors="coerce")
    out["visibility_code"] = visibility_code
    lower, upper = decode_visibility_bounds(visibility_code)
    out["visibility_lower_m"] = lower
    out["visibility_upper_m"] = upper
    out["visibility_m"] = derive_visibility_midpoint(lower, upper)

    out["present_weather_code"] = pd.to_numeric(chunk["WW_station"], errors="coerce")
    out["fog_indicator"] = pd.to_numeric(chunk["W1_station"], errors="coerce")
    out["fog_from_weather_code"] = (
        out["present_weather_code"].between(40, 49, inclusive="both")
    ).astype("Int64")

    fog_label, low_visibility_label, visibility_class = derive_fog_labels(
        out["visibility_code"],
        out["fog_indicator"],
    )
    out["fog_label"] = fog_label
    out["low_visibility_label"] = low_visibility_label
    out["visibility_class"] = visibility_class

    out["era5_t2m"] = pd.to_numeric(chunk["era5_t2m"], errors="coerce")
    out["era5_d2m"] = pd.to_numeric(chunk["era5_d2m"], errors="coerce")
    out["era5_UG"] = pd.to_numeric(chunk["era5_UG"], errors="coerce")
    out["surface_pressure"] = pd.to_numeric(chunk["era5_sp"], errors="coerce")
    out["era5_ws10"] = pd.to_numeric(chunk["era5_ws10"], errors="coerce")
    out["era5_u10"] = pd.to_numeric(chunk["era5_u10"], errors="coerce")
    out["era5_v10"] = pd.to_numeric(chunk["era5_v10"], errors="coerce")
    out["tp"] = pd.to_numeric(chunk["era5_tp"], errors="coerce")
    out["tcc"] = pd.to_numeric(chunk["era5_tcc"], errors="coerce")
    out["blh"] = pd.to_numeric(chunk["era5_blh"], errors="coerce")
    out["era5_ssrd"] = pd.to_numeric(chunk["era5_ssrd"], errors="coerce")
    out["era5_strd"] = pd.to_numeric(chunk["era5_strd"], errors="coerce")

    return out[CANONICAL_COLUMNS]


def aggregate_daily(chunk: pd.DataFrame) -> pd.DataFrame:
    day = chunk.copy()
    day["date"] = pd.to_datetime(day["time"], utc=True).dt.date
    grouped = (
        day.groupby(["station", "date"], as_index=False)
        .agg(
            stationname=("stationname", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            height=("height", "first"),
            obs_tmax=("t2m_obs", "max"),
            obs_tmin=("t2m_obs", "min"),
            era5_tmax=("era5_t2m", "max"),
            era5_tmin=("era5_t2m", "min"),
            fog_hours=("fog_label", "sum"),
            low_visibility_hours=("low_visibility_label", "sum"),
            min_visibility_m=("visibility_m", "min"),
            n_hours=("station", "count"),
        )
    )
    return grouped


def build_station_metadata(summary: pd.DataFrame) -> pd.DataFrame:
    meta = summary.rename(columns={"stationname": "name"})
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


def build_station_manifest(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        [
            "station",
            "stationname",
            "lat",
            "lon",
            "height",
            "start_time_utc",
            "end_time_utc",
            "n_rows",
        ]
    ].sort_values("station")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Prepare fog experiment datasets from era5_merged.csv.")
    parser.add_argument("--input-csv", default=str(repo_root / "era5_merged.csv"))
    parser.add_argument("--output-dir", default=str(repo_root / "datasets/fog_ready"))
    parser.add_argument("--manifest-path", default=str(repo_root / "baseline_assets/netherlands_station_manifest.csv"))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    hourly_out = output_dir / "fog_training_merged_hourly.csv"
    daily_out = output_dir / "fog_daily_eval_reference.csv"
    meta_out = output_dir / "station_metadata.csv"

    if hourly_out.exists():
        hourly_out.unlink()

    coverage_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=args.chunk_size), start=1):
        canonical = transform_chunk(chunk)
        canonical.to_csv(hourly_out, mode="a", header=(i == 1), index=False)

        coverage_parts.append(
            canonical.groupby("station", as_index=False)
            .agg(
                stationname=("stationname", "first"),
                lat=("lat", "first"),
                lon=("lon", "first"),
                height=("height", "first"),
                start_time_utc=("time", "min"),
                end_time_utc=("time", "max"),
                n_rows=("station", "count"),
            )
        )
        daily_parts.append(aggregate_daily(canonical))
        print(f"Processed chunk {i}", flush=True)

    coverage = (
        pd.concat(coverage_parts, ignore_index=True)
        .groupby("station", as_index=False)
        .agg(
            stationname=("stationname", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            height=("height", "first"),
            start_time_utc=("start_time_utc", "min"),
            end_time_utc=("end_time_utc", "max"),
            n_rows=("n_rows", "sum"),
        )
    )

    daily = (
        pd.concat(daily_parts, ignore_index=True)
        .groupby(["station", "date"], as_index=False)
        .agg(
            stationname=("stationname", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            height=("height", "first"),
            obs_tmax=("obs_tmax", "max"),
            obs_tmin=("obs_tmin", "min"),
            era5_tmax=("era5_tmax", "max"),
            era5_tmin=("era5_tmin", "min"),
            fog_hours=("fog_hours", "sum"),
            low_visibility_hours=("low_visibility_hours", "sum"),
            min_visibility_m=("min_visibility_m", "min"),
            n_hours=("n_hours", "sum"),
        )
        .sort_values(["station", "date"])
    )

    station_meta = build_station_metadata(coverage)
    manifest = build_station_manifest(coverage)

    daily.to_csv(daily_out, index=False)
    station_meta.to_csv(meta_out, index=False)
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote hourly dataset:   {hourly_out}")
    print(f"Wrote daily reference:  {daily_out}")
    print(f"Wrote station metadata: {meta_out}")
    print(f"Wrote station manifest: {manifest_path}")


if __name__ == "__main__":
    main()
