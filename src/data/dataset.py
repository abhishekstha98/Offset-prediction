"""
dataset.py — ERA5LandDataset for offset prediction.

Loads comparison_all_years.csv which contains aligned ERA5 and station observations.
Each call to __getitem__ returns one day's node features, target offsets, valid masks,
and station metadata needed for graph construction and SLOBO masking.

Legacy node features (6): [mx2t, mn2t, UG_era5, height, sin_doy, cos_doy]
Fog-upgrade node features (17):
    [mx2t, mn2t, era5_t2m, era5_d2m, UG_era5, dewpoint_spread_2m,
     rh_2m, era5_u10, era5_v10, wind_speed_10m, theta_v_2m,
     theta_v_delta_1d, t2m_delta_1d, dewpoint_spread_delta_1d,
     height, sin_doy, cos_doy]
Targets (2):       [offset_tmax, offset_tmin]  (TX - mx2t, TN - mn2t)
valid_mask (2):    True where both target and ERA5 input are non-NaN.

NOTE on UG_station: Deliberately excluded from node features.
UG_station is a local ground-truth observation. Including it would constitute
data leakage — at real inference time only ERA5/reanalysis fields are available.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
from typing import Sequence


_COLUMN_ALIASES = {
    "era5_mx2t": "mx2t",
    "era5_mn2t": "mn2t",
    "era5_UG": "UG_era5",
    "t2m": "era5_t2m",
    "d2m": "era5_d2m",
    "u10": "era5_u10",
    "v10": "era5_v10",
    "ws10": "era5_ws10",
}

LEGACY_FEATURE_COLUMNS = [
    "mx2t",
    "mn2t",
    "UG_era5",
    "height",
    "sin_doy",
    "cos_doy",
]

FOG_FEATURE_COLUMNS = [
    "mx2t",
    "mn2t",
    "era5_t2m",
    "era5_d2m",
    "UG_era5",
    "dewpoint_spread_2m",
    "rh_2m",
    "era5_u10",
    "era5_v10",
    "wind_speed_10m",
    "theta_v_2m",
    "theta_v_delta_1d",
    "t2m_delta_1d",
    "dewpoint_spread_delta_1d",
    "height",
    "sin_doy",
    "cos_doy",
]

FOG_LABEL_CANDIDATES = [
    "fog_label",
    "low_visibility_label",
    "visibility_class",
]


def standardize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw CSV schemas to the canonical training column names.
    """
    df = df.copy()
    rename_map = {
        src: dst
        for src, dst in _COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _temperature_to_kelvin(temp: pd.Series) -> pd.Series:
    """Treat values above 150 as Kelvin, otherwise Celsius."""
    values = pd.to_numeric(temp, errors="coerce")
    if values.dropna().median() > 150:
        return values
    return values + 273.15


def _saturation_vapor_pressure_hpa(temp_c: pd.Series) -> pd.Series:
    """Magnus saturation vapor pressure over water, with temperature in Celsius."""
    temp_c = pd.to_numeric(temp_c, errors="coerce")
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def _estimate_pressure_pa(height_m: pd.Series) -> pd.Series:
    """Standard-atmosphere surface pressure estimate from station elevation."""
    height_m = pd.to_numeric(height_m, errors="coerce").fillna(0.0)
    return 101325.0 * np.power(1.0 - 2.25577e-5 * height_m, 5.2559)


def _resolve_pressure_pa(df: pd.DataFrame) -> pd.Series:
    """Use pressure from the CSV if present; otherwise estimate from height."""
    for col in ("surface_pressure", "era5_sp", "sp"):
        if col in df.columns:
            pressure = pd.to_numeric(df[col], errors="coerce")
            if pressure.dropna().median() < 2000:
                pressure = pressure * 100.0  # hPa -> Pa
            return pressure
    return _estimate_pressure_pa(df["height"])


def _infer_lag_steps(df: pd.DataFrame, time_col: str, station_col: str) -> int:
    """
    Infer how many rows correspond to a one-day lag per station.

    Daily data returns 1. Hourly data returns 24. This keeps the feature name
    theta_v_delta_1d stable across the current daily dataset and future hourly
    ERA5-Land exports.
    """
    deltas = (
        df.sort_values([station_col, time_col])
        .groupby(station_col, sort=False)[time_col]
        .diff()
        .dropna()
    )
    if deltas.empty:
        return 1
    median_hours = deltas.median() / pd.Timedelta(hours=1)
    if not np.isfinite(median_hours) or median_hours <= 0:
        return 1
    return max(1, int(round(24.0 / median_hours)))


def add_fog_features(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    station_col: str = "station",
    lag_steps: int | None = None,
    fill_lag_value: float = 0.0,
) -> pd.DataFrame:
    """
    Add fog-relevant meteorological features to a station-time dataframe.

    The transformation is row-wise and PyTorch-compatible: the returned columns
    are numeric float features that can be stacked into tensors directly.
    It works for the current daily `merged.csv` and for future hourly exports.
    """
    df = standardize_input_columns(df).copy()
    df[time_col] = pd.to_datetime(df[time_col])

    required = ["era5_t2m", "era5_d2m", "era5_u10", "era5_v10", "height"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Cannot compute fog features; missing required columns: "
            f"{missing}. Available columns: {df.columns.tolist()}"
        )

    # Existing CSV values are Celsius. The Kelvin helper keeps future ERA5
    # exports safe if they arrive in native Kelvin units.
    t2m_raw = pd.to_numeric(df["era5_t2m"], errors="coerce")
    d2m_raw = pd.to_numeric(df["era5_d2m"], errors="coerce")
    t2m_k = _temperature_to_kelvin(t2m_raw)
    d2m_k = _temperature_to_kelvin(d2m_raw)
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15

    df["dewpoint_spread_2m"] = t2m_c - d2m_c

    es_t = _saturation_vapor_pressure_hpa(t2m_c)
    es_td = _saturation_vapor_pressure_hpa(d2m_c)
    df["rh_2m"] = (100.0 * es_td / es_t).clip(lower=0.0, upper=100.0)

    if "era5_ws10" in df.columns:
        df["wind_speed_10m"] = pd.to_numeric(df["era5_ws10"], errors="coerce")
    else:
        u10 = pd.to_numeric(df["era5_u10"], errors="coerce")
        v10 = pd.to_numeric(df["era5_v10"], errors="coerce")
        df["wind_speed_10m"] = np.sqrt(u10**2 + v10**2)

    pressure_pa = _resolve_pressure_pa(df)
    vapor_pressure_pa = _saturation_vapor_pressure_hpa(d2m_c) * 100.0
    specific_humidity = 0.622 * vapor_pressure_pa / (pressure_pa - 0.378 * vapor_pressure_pa)
    theta = t2m_k * np.power(100000.0 / pressure_pa, 0.286)
    df["theta_v_2m"] = theta * (1.0 + 0.61 * specific_humidity)

    df = df.sort_values([station_col, time_col]).reset_index(drop=True)
    if lag_steps is None:
        lag_steps = _infer_lag_steps(df, time_col, station_col)

    grouped = df.groupby(station_col, sort=False)
    df["theta_v_delta_1d"] = grouped["theta_v_2m"].diff(lag_steps).fillna(fill_lag_value)
    df["t2m_delta_1d"] = grouped["era5_t2m"].diff(lag_steps).fillna(fill_lag_value)
    df["dewpoint_spread_delta_1d"] = (
        grouped["dewpoint_spread_2m"].diff(lag_steps).fillna(fill_lag_value)
    )

    return df


def resolve_feature_columns(
    scaler: dict | None = None,
    feature_columns: Sequence[str] | None = None,
) -> list[str]:
    """
    Resolve the feature list for new and legacy checkpoints.

    New scalers/checkpoints store `feature_columns`. Legacy 6-feature scalers
    did not, so they are inferred from the length of their mean vector.
    """
    if feature_columns is not None:
        return list(feature_columns)
    if scaler is not None and "feature_columns" in scaler:
        return list(scaler["feature_columns"])
    if scaler is not None and "mean" in scaler:
        n_features = int(np.asarray(scaler["mean"]).shape[0])
        if n_features == len(LEGACY_FEATURE_COLUMNS):
            return list(LEGACY_FEATURE_COLUMNS)
        if n_features == len(FOG_FEATURE_COLUMNS):
            return list(FOG_FEATURE_COLUMNS)
        raise ValueError(f"Cannot infer feature columns for scaler with {n_features} features.")
    return list(FOG_FEATURE_COLUMNS)


class ERA5LandDataset(Dataset):
    """
    One item = one day's spatial graph data.

    Attributes:
        unique_dates  (np.ndarray): Sorted array of all dates in the split.
        unique_stations (pd.DataFrame): One row per station with lat, lon, height,
                                        station ID — used for graph building & SLOBO.
        scaler (dict|None):  If provided, used to standardize node features;
                             otherwise raw values are returned (fit mode).
    """

    # Column names in comparison_all_years.csv
    STATION_COL = "station"
    TIME_COL = "time"
    LAT_COL = "lat"
    LON_COL = "lon"
    HEIGHT_COL = "height"
    ERA5_TMAX_COL = "mx2t"
    ERA5_TMIN_COL = "mn2t"
    ERA5_HUM_COL = "UG_era5"
    STA_TMAX_COL = "TX"
    STA_TMIN_COL = "TN"

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: dict | None = None,
        station_order: Sequence[str] | None = None,
        feature_columns: Sequence[str] | None = None,
        sequence_length: int = 1,
    ):
        """
        Args:
            df:      Pre-filtered DataFrame (e.g., train-years only, or test year).
                     Must contain all required columns defined above.
            scaler:  dict with keys 'mean' and 'std',
                     pre-fitted on training data. If None, returns raw features
                     so you can fit the scaler externally.
            feature_columns: Optional explicit feature list. Defaults to the
                     17-feature fog-upgrade list unless a legacy scaler is used.
            sequence_length: Number of time steps to return. 1 preserves the
                     historical daily graph shape (N, F). Values >1 return
                     (T, N, F) for spatiotemporal MPT training.
        """
        df = standardize_input_columns(df).copy()
        df[self.TIME_COL] = pd.to_datetime(df[self.TIME_COL])
        resolved_feature_columns = resolve_feature_columns(scaler, feature_columns)

        # Day-of-year encoding is available for both legacy and fog-upgrade features.
        doy = df[self.TIME_COL].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)

        missing_requested_features = [
            col for col in resolved_feature_columns if col not in df.columns
        ]
        if missing_requested_features:
            df = add_fog_features(df)
            df[self.TIME_COL] = pd.to_datetime(df[self.TIME_COL])
            doy = df[self.TIME_COL].dt.dayofyear
            df["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
            df["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)

        # Compute target offsets
        df["offset_tmax"] = df[self.STA_TMAX_COL] - df[self.ERA5_TMAX_COL]
        df["offset_tmin"] = df[self.STA_TMIN_COL] - df[self.ERA5_TMIN_COL]

        self.df = df
        self.scaler = scaler
        self.feature_columns = resolved_feature_columns
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        self.fog_label_col = next(
            (col for col in FOG_LABEL_CANDIDATES if col in df.columns),
            None,
        )

        self.unique_dates = np.sort(df[self.TIME_COL].unique())
        if len(self.unique_dates) < self.sequence_length:
            raise ValueError(
                f"Dataset has {len(self.unique_dates)} unique times, fewer than "
                f"sequence_length={self.sequence_length}."
            )
        self.sample_dates = self.unique_dates[self.sequence_length - 1 :]
        self.station_order = list(station_order) if station_order is not None else None
        self._station_order_lookup = (
            {sid: idx for idx, sid in enumerate(self.station_order)}
            if self.station_order is not None
            else None
        )
        self.unique_stations = (
            df[[self.STATION_COL, self.LAT_COL, self.LON_COL, self.HEIGHT_COL]]
            .drop_duplicates(subset=[self.STATION_COL])
            .reset_index(drop=True)
        )
        if self._station_order_lookup is not None:
            self.unique_stations["_station_order"] = self.unique_stations[self.STATION_COL].map(
                self._station_order_lookup
            )
            self.unique_stations = (
                self.unique_stations.sort_values("_station_order")
                .drop(columns=["_station_order"])
                .reset_index(drop=True)
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_dates)

    def _frame_at_time(self, date) -> pd.DataFrame:
        day_df = self.df[self.df[self.TIME_COL] == date].copy()
        if self._station_order_lookup is not None:
            day_df["_station_order"] = day_df[self.STATION_COL].map(self._station_order_lookup)
            if day_df["_station_order"].isna().any():
                missing = day_df.loc[day_df["_station_order"].isna(), self.STATION_COL].unique().tolist()
                raise ValueError(
                    f"Encountered station IDs not present in station_order for date {date}: {missing}"
                )
            day_df = (
                day_df.sort_values("_station_order")
                .drop(columns=["_station_order"])
                .reset_index(drop=True)
            )
        return day_df

    def _features_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns in dataset frame: {missing}")
        x_raw = frame[self.feature_columns].values.astype(np.float32)

        # Apply normalization if scaler is available
        if self.scaler is not None:
            x_raw = (x_raw - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)
        return np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, idx: int) -> dict:
        date = self.sample_dates[idx]
        day_df = self._frame_at_time(date)

        if self.sequence_length == 1:
            x_raw = self._features_from_frame(day_df)
        else:
            date_idx = int(np.where(self.unique_dates == date)[0][0])
            window_dates = self.unique_dates[date_idx - self.sequence_length + 1 : date_idx + 1]
            x_raw = np.stack(
                [self._features_from_frame(self._frame_at_time(t)) for t in window_dates],
                axis=0,
            )

        # Targets
        offset_tmax = day_df["offset_tmax"].values.astype(np.float32)
        offset_tmin = day_df["offset_tmin"].values.astype(np.float32)
        y = np.stack([offset_tmax, offset_tmin], axis=1)  # (N, 2)

        # Valid mask: True where target AND ERA5 inputs are non-NaN.
        # Nodes with missing targets still appear in the graph (spatial context),
        # but are masked out of the loss computation.
        valid_tmax = ~np.isnan(offset_tmax)
        valid_tmin = ~np.isnan(offset_tmin)
        valid_mask = np.stack([valid_tmax, valid_tmin], axis=1)  # (N, 2) bool

        # Replace NaN targets with 0.0 so tensors remain finite
        y = np.nan_to_num(y, nan=0.0)

        if self.fog_label_col is not None:
            fog_target = day_df[self.fog_label_col].values.astype(np.float32)
            fog_valid_mask = ~np.isnan(fog_target)
            fog_target = np.nan_to_num(fog_target, nan=0.0)
        else:
            fog_target = np.zeros(len(day_df), dtype=np.float32)
            fog_valid_mask = np.zeros(len(day_df), dtype=bool)

        # Station spatial coordinates (for graph builder) and IDs (for SLOBO mask)
        pos = day_df[[self.LAT_COL, self.LON_COL]].values.astype(np.float32)
        heights = day_df[self.HEIGHT_COL].values.astype(np.float32)
        station_ids = day_df[self.STATION_COL].values  # str array

        return {
            "x": torch.tensor(x_raw, dtype=torch.float),           # (N, F) or (T, N, F)
            "y": torch.tensor(y, dtype=torch.float),                # (N, 2)
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),  # (N, 2)
            "fog_target": torch.tensor(fog_target, dtype=torch.float),  # (N,)
            "fog_valid_mask": torch.tensor(fog_valid_mask, dtype=torch.bool),  # (N,)
            "pos": torch.tensor(pos, dtype=torch.float),            # (N, 2) lat/lon
            "heights": torch.tensor(heights, dtype=torch.float),    # (N,)
            "station_ids": station_ids,                              # (N,) str
            "date": str(date),
        }


# ------------------------------------------------------------------
# Scaler utilities
# ------------------------------------------------------------------

def fit_scaler(dataset: ERA5LandDataset) -> dict:
    """
    Fit a StandardScaler over node features across ALL items in the dataset.
    Only call this on the training split.

    Returns:
        dict with 'mean', 'std', and 'feature_columns'.
    """
    feat_cols = dataset.feature_columns
    all_feats = dataset.df[feat_cols].values.astype(np.float32)
    # Only use rows where features are non-NaN
    mask = ~np.any(np.isnan(all_feats), axis=1)
    all_feats = all_feats[mask]
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    return {"mean": mean, "std": std, "feature_columns": list(feat_cols)}


def save_scaler(scaler: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
