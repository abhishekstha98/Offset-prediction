"""
Utilities for converting the hourly `era5_merged.csv` schema into the daily
training schema expected by `src.train`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_HOURLY_REQUIRED_COLUMNS = [
    "station",
    "time",
    "stationname",
    "lat",
    "lon",
    "height",
    "T_station",
    "era5_t2m",
    "TD_station",
    "era5_d2m",
    "UG_station",
    "era5_UG",
    "era5_ws10",
    "era5_u10",
    "era5_v10",
    "VV_station",
    "W1_station",
    "WW_station",
]

TRAINING_OUTPUT_COLUMNS = [
    "station",
    "time",
    "stationname",
    "lat",
    "lon",
    "height",
    "TX",
    "era5_mx2t",
    "TN",
    "era5_mn2t",
    "era5_t2m",
    "era5_d2m",
    "UG_station",
    "era5_UG",
    "era5_u10",
    "era5_v10",
    "era5_ws10",
    "fog_hours",
    "low_visibility_hours",
    "min_visibility_m",
    "fog_label",
    "low_visibility_label",
    "visibility_class",
    "n_station_hours",
    "n_era5_hours",
    "station_daily_source_date",
    "era5_daily_source_date",
]


def detect_hourly_training_source(columns: list[str] | pd.Index | set[str]) -> bool:
    """Return True when the CSV looks like raw hourly station/ERA5 data."""
    cols = set(columns)
    has_targets = "TX" in cols and "TN" in cols and ("era5_mx2t" in cols or "mx2t" in cols)
    if has_targets:
        return False
    return set(RAW_HOURLY_REQUIRED_COLUMNS).issubset(cols)


def decode_visibility_bounds(code: pd.Series) -> tuple[pd.Series, pd.Series]:
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


def derive_hourly_fog_labels(
    visibility_code: pd.Series,
    fog_indicator: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    visibility_num = pd.to_numeric(visibility_code, errors="coerce")
    fog_indicator_num = pd.to_numeric(fog_indicator, errors="coerce")

    fog_label = pd.Series(np.nan, index=visibility_code.index, dtype="float64")
    low_visibility_label = pd.Series(np.nan, index=visibility_code.index, dtype="float64")
    visibility_class = pd.Series(np.nan, index=visibility_code.index, dtype="float64")

    fog_label.loc[fog_indicator_num == 1] = 1.0
    fog_label.loc[fog_indicator_num == 0] = 0.0
    fallback_mask = fog_indicator_num.isna() & visibility_num.notna()
    fog_label.loc[fallback_mask & (visibility_num <= 9)] = 1.0
    fog_label.loc[fallback_mask & (visibility_num > 9)] = 0.0

    vis_known = visibility_num.notna()
    low_visibility_label.loc[vis_known & (visibility_num <= 49)] = 1.0
    low_visibility_label.loc[vis_known & (visibility_num >= 50)] = 0.0

    visibility_class.loc[vis_known & (visibility_num <= 1)] = 3.0
    visibility_class.loc[vis_known & visibility_num.between(2, 9, inclusive="both")] = 2.0
    visibility_class.loc[vis_known & visibility_num.between(10, 49, inclusive="both")] = 1.0
    visibility_class.loc[vis_known & (visibility_num >= 50)] = 0.0

    return fog_label, low_visibility_label, visibility_class


def load_hourly_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=RAW_HOURLY_REQUIRED_COLUMNS,
        dtype={
            "station": "string",
            "stationname": "string",
            "lat": "float32",
            "lon": "float32",
            "height": "float32",
            "T_station": "float32",
            "era5_t2m": "float32",
            "TD_station": "float32",
            "era5_d2m": "float32",
            "UG_station": "float32",
            "era5_UG": "float32",
            "era5_ws10": "float32",
            "era5_u10": "float32",
            "era5_v10": "float32",
            "VV_station": "float32",
            "W1_station": "float32",
            "WW_station": "float32",
        },
        parse_dates=["time"],
    )


def prepare_hourly_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["raw_date"] = out["time"].dt.floor("D")

    lower, upper = decode_visibility_bounds(out["VV_station"])
    out["visibility_m"] = derive_visibility_midpoint(lower, upper)

    fog_label, low_visibility_label, visibility_class = derive_hourly_fog_labels(
        out["VV_station"],
        out["W1_station"],
    )
    out["fog_label_hourly"] = fog_label
    out["low_visibility_label_hourly"] = low_visibility_label
    out["visibility_class_hourly"] = visibility_class

    out["fog_label_valid"] = out["fog_label_hourly"].notna().astype("int16")
    out["fog_label_positive"] = out["fog_label_hourly"].fillna(0.0).astype("float32")
    out["low_visibility_valid"] = out["low_visibility_label_hourly"].notna().astype("int16")
    out["low_visibility_positive"] = out["low_visibility_label_hourly"].fillna(0.0).astype(
        "float32"
    )
    out["row_count"] = 1
    return out


def build_station_daily(df: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:
    grouped = df.groupby(["station", "raw_date"], observed=True, sort=True)
    daily = grouped.agg(
        stationname=("stationname", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        height=("height", "first"),
        TX=("T_station", "max"),
        TN=("T_station", "min"),
        UG_station=("UG_station", "mean"),
        fog_hours=("fog_label_positive", "sum"),
        fog_label_valid_hours=("fog_label_valid", "sum"),
        low_visibility_hours=("low_visibility_positive", "sum"),
        low_visibility_valid_hours=("low_visibility_valid", "sum"),
        min_visibility_m=("visibility_m", "min"),
        visibility_class=("visibility_class_hourly", "max"),
        n_station_hours=("row_count", "sum"),
    ).reset_index()

    daily = daily[daily["n_station_hours"] >= min_hours_per_day].copy()
    daily["fog_label"] = np.where(
        daily["fog_label_valid_hours"] > 0,
        (daily["fog_hours"] > 0).astype("float32"),
        np.nan,
    )
    daily["low_visibility_label"] = np.where(
        daily["low_visibility_valid_hours"] > 0,
        (daily["low_visibility_hours"] > 0).astype("float32"),
        np.nan,
    )
    daily["station_daily_source_date"] = daily["raw_date"]
    daily["time"] = daily["raw_date"] + pd.Timedelta(days=1)
    return daily.drop(columns=["raw_date", "fog_label_valid_hours", "low_visibility_valid_hours"])


def build_era5_daily(df: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:
    grouped = df.groupby(["station", "raw_date"], observed=True, sort=True)
    daily = grouped.agg(
        era5_mx2t=("era5_t2m", "max"),
        era5_mn2t=("era5_t2m", "min"),
        era5_t2m=("era5_t2m", "mean"),
        era5_d2m=("era5_d2m", "mean"),
        era5_UG=("era5_UG", "mean"),
        era5_u10=("era5_u10", "mean"),
        era5_v10=("era5_v10", "mean"),
        n_era5_hours=("row_count", "sum"),
    ).reset_index()

    daily = daily[daily["n_era5_hours"] >= min_hours_per_day].copy()
    daily["era5_ws10"] = np.sqrt(daily["era5_u10"] ** 2 + daily["era5_v10"] ** 2)
    daily["era5_daily_source_date"] = daily["raw_date"]
    daily["time"] = daily["raw_date"]
    return daily.drop(columns=["raw_date"])


def build_trainable_daily_dataset(hourly: pd.DataFrame, min_hours_per_day: int = 20) -> pd.DataFrame:
    prepared = prepare_hourly_derivatives(hourly)
    station_daily = build_station_daily(prepared, min_hours_per_day=min_hours_per_day)
    era5_daily = build_era5_daily(prepared, min_hours_per_day=min_hours_per_day)
    merged = era5_daily.merge(
        station_daily,
        on=["station", "time"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values(["station", "time"]).reset_index(drop=True)

    for col in TRAINING_OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[TRAINING_OUTPUT_COLUMNS]


def load_training_frame(path: str | Path, min_hours_per_day: int = 20) -> tuple[pd.DataFrame, bool]:
    """
    Load either a ready-made daily training CSV or a raw hourly CSV.

    Returns:
        (dataframe, was_hourly_converted)
    """
    header = pd.read_csv(path, nrows=0)
    if detect_hourly_training_source(header.columns):
        hourly = load_hourly_frame(path)
        return build_trainable_daily_dataset(hourly, min_hours_per_day=min_hours_per_day), True
    return pd.read_csv(path), False
