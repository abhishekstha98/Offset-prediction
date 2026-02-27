"""
split.py — Spatial Leave-One-Block-Out (SLOBO) cross-validation and temporal splitting.

Strategy:
  - Temporal: The most recent year (cfg.split.test_year) is held out as the
    final test set. SLOBO folds are constructed only from the remaining years.
  - Spatial: K-Means clusters stations into n_blocks geographical blocks.
    Each fold withholds one block for validation, training on the rest.
    Critically, withheld-block stations are HIDDEN from loss computation but
    still participate in message passing (they provide spatial context).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def temporal_split(df: pd.DataFrame, test_year: int, time_col: str = "time"):
    """
    Split a multi-year DataFrame into trainval and test by year.

    Args:
        df:        Full merged DataFrame.
        test_year: Year to hold out entirely as the final test set.
        time_col:  Name of the date/time column.

    Returns:
        trainval_df:  All rows from years < test_year.
        test_df:      All rows from test_year.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["_year"] = df[time_col].dt.year
    trainval_df = df[df["_year"] < test_year].drop(columns=["_year"]).reset_index(drop=True)
    test_df = df[df["_year"] == test_year].drop(columns=["_year"]).reset_index(drop=True)
    return trainval_df, test_df


def build_slobo_folds(
    unique_stations: pd.DataFrame,
    n_blocks: int = 4,
    lat_col: str = "lat",
    lon_col: str = "lon",
    station_col: str = "station",
    random_state: int = 42,
):
    """
    Cluster stations into spatial blocks using K-Means and return fold assignments.

    Args:
        unique_stations: DataFrame with one row per station, containing lat/lon.
        n_blocks:        Number of spatial clusters = number of SLOBO folds.
                         For 23 stations, recommended K=4 (≈5-6 stations/fold).
                         Do NOT exceed K=5 for this dataset size.
        lat_col, lon_col, station_col: Column names.
        random_state:    For reproducibility.

    Returns:
        station_to_block: dict mapping station_id → block_id (0-indexed).
    """
    coords = unique_stations[[lat_col, lon_col]].values
    kmeans = KMeans(n_clusters=n_blocks, random_state=random_state, n_init=10)
    block_labels = kmeans.fit_predict(coords)

    station_to_block = dict(
        zip(unique_stations[station_col].values, block_labels.tolist())
    )
    return station_to_block


def get_fold_masks(
    station_ids_in_df: np.ndarray,
    station_to_block: dict,
    val_block: int,
):
    """
    Given the station IDs (in the order they appear in a daily df), return
    boolean masks for training nodes and validation nodes.

    Validation nodes: belong to val_block → excluded from LOSS, but included
        in the graph for message passing.
    Training nodes:   all other blocks → contribute to the loss.

    Args:
        station_ids_in_df: Array of station IDs aligned with node ordering.
        station_to_block:  Mapping returned by build_slobo_folds().
        val_block:         Which block index to treat as validation this fold.

    Returns:
        train_mask: bool tensor-like array, True for training nodes.
        val_mask:   bool tensor-like array, True for validation nodes.
    """
    blocks = np.array([station_to_block.get(sid, -1) for sid in station_ids_in_df])
    val_mask = blocks == val_block
    train_mask = ~val_mask
    return train_mask, val_mask


def summarize_folds(unique_stations: pd.DataFrame, station_to_block: dict, station_col: str = "station"):
    """
    Print a human-readable summary of how stations are distributed across blocks.
    """
    unique_stations = unique_stations.copy()
    unique_stations["block"] = unique_stations[station_col].map(station_to_block)
    summary = unique_stations.groupby("block")[station_col].apply(list)
    for block_id, stations in summary.items():
        print(f"  Block {block_id}: {len(stations)} stations → {stations}")
