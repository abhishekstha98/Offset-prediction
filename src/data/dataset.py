"""
dataset.py — ERA5LandDataset for offset prediction.

Loads comparison_all_years.csv which contains aligned ERA5 and station observations.
Each call to __getitem__ returns one day's node features, target offsets, valid masks,
and station metadata needed for graph construction and SLOBO masking.

Node features (6): [mx2t, mn2t, UG_era5, height, sin_doy, cos_doy]
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

    def __init__(self, df: pd.DataFrame, scaler: dict | None = None):
        """
        Args:
            df:      Pre-filtered DataFrame (e.g., train-years only, or test year).
                     Must contain all required columns defined above.
            scaler:  dict with keys 'mean' and 'std' (both shape (6,) arrays),
                     pre-fitted on training data. If None, returns raw features
                     so you can fit the scaler externally.
        """
        df = df.copy()
        df[self.TIME_COL] = pd.to_datetime(df[self.TIME_COL])

        # Compute target offsets
        df["offset_tmax"] = df[self.STA_TMAX_COL] - df[self.ERA5_TMAX_COL]
        df["offset_tmin"] = df[self.STA_TMIN_COL] - df[self.ERA5_TMIN_COL]

        # Day-of-year encoding
        doy = df[self.TIME_COL].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)

        self.df = df
        self.scaler = scaler

        self.unique_dates = np.sort(df[self.TIME_COL].unique())
        self.unique_stations = (
            df[[self.STATION_COL, self.LAT_COL, self.LON_COL, self.HEIGHT_COL]]
            .drop_duplicates(subset=[self.STATION_COL])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.unique_dates)

    def __getitem__(self, idx: int) -> dict:
        date = self.unique_dates[idx]
        day_df = self.df[self.df[self.TIME_COL] == date].copy()

        # Node features: [mx2t, mn2t, UG_era5, height, sin_doy, cos_doy]
        feat_cols = [
            self.ERA5_TMAX_COL,
            self.ERA5_TMIN_COL,
            self.ERA5_HUM_COL,
            self.HEIGHT_COL,
            "sin_doy",
            "cos_doy",
        ]
        x_raw = day_df[feat_cols].values.astype(np.float32)

        # Apply normalization if scaler is available
        if self.scaler is not None:
            x_raw = (x_raw - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)

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

        # Station spatial coordinates (for graph builder) and IDs (for SLOBO mask)
        pos = day_df[[self.LAT_COL, self.LON_COL]].values.astype(np.float32)
        heights = day_df[self.HEIGHT_COL].values.astype(np.float32)
        station_ids = day_df[self.STATION_COL].values  # str array

        return {
            "x": torch.tensor(x_raw, dtype=torch.float),           # (N, 6)
            "y": torch.tensor(y, dtype=torch.float),                # (N, 2)
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),  # (N, 2)
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
        dict with 'mean' and 'std' arrays of shape (6,).
    """
    feat_cols = [
        ERA5LandDataset.ERA5_TMAX_COL,
        ERA5LandDataset.ERA5_TMIN_COL,
        ERA5LandDataset.ERA5_HUM_COL,
        ERA5LandDataset.HEIGHT_COL,
        "sin_doy",
        "cos_doy",
    ]
    all_feats = dataset.df[feat_cols].values.astype(np.float32)
    # Only use rows where features are non-NaN
    mask = ~np.any(np.isnan(all_feats), axis=1)
    all_feats = all_feats[mask]
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    return {"mean": mean, "std": std}


def save_scaler(scaler: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
