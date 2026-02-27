"""
graph_builder.py — Static k-NN spatial graph with 4-component edge features.

Graph structure:
  - Nodes: weather stations (one per unique station in the dataset).
  - Edges: directed k-NN connections.  For each node i, we connect its k
    nearest neighbours (by great-circle distance) as sources.
  - Edge features (4): [distance_km, Δlat, Δlon, Δheight]
    Δlat/Δlon/Δheight are computed as (source - target), i.e. directional.
    This means the features for A→B and B→A are sign-flipped on dims 1-3,
    which is the correct convention for a directed message-passing network.

Static graph rationale:
  Stations have fixed coordinates. Missing daily observations don't change
  the network topology — nodes with NaN ground truth are handled via the
  valid_mask in the loss, not by removing them from the graph.  Keeping the
  graph static avoids recomputing it every day and preserves spatial context
  even for partially observed days.
"""

import torch
import numpy as np
from scipy.spatial import cKDTree


def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def build_static_graph(
    unique_stations,
    k: int = 3,
    lat_col: str = "lat",
    lon_col: str = "lon",
    height_col: str = "height",
    station_col: str = "station",
):
    """
    Build a static directed k-NN graph from unique station coordinates.

    Args:
        unique_stations: pd.DataFrame with one row per station.
                         Must have lat, lon, height, station columns.
        k:               Number of nearest neighbours per node.
                         Recommended k=3 for 23-station Netherlands dataset.
                         Revisit when moving to Nepal.

    Returns:
        edge_index:  LongTensor of shape (2, E).  edge_index[0] = source, [1] = target.
        edge_attr:   FloatTensor of shape (E, 4).
                     Columns: [distance_km, Δlat, Δlon, Δheight]
                     where Δ = source_value - target_value.
        station_order: list of station IDs in node-index order,
                       so node i corresponds to station_order[i].
    """
    lats = unique_stations[lat_col].values
    lons = unique_stations[lon_col].values
    heights = unique_stations[height_col].values
    station_ids = unique_stations[station_col].values.tolist()
    N = len(station_ids)

    # Use lat/lon directly for k-NN search (units don't matter here,
    # haversine is used for the actual distance feature)
    coords = np.stack([lats, lons], axis=1)
    tree = cKDTree(coords)

    # k+1 because query includes the point itself
    dists, indices = tree.query(coords, k=min(k + 1, N))

    source_nodes = []
    target_nodes = []
    dist_km_list = []
    delta_lat_list = []
    delta_lon_list = []
    delta_height_list = []

    for i in range(N):
        for j_rank, j in enumerate(indices[i]):
            if j == i:
                continue  # skip self-loops

            # Compute great-circle distance
            d_km = haversine_km(lats[i], lons[i], lats[j], lons[j])

            # Edge direction: j (source) → i (target)
            # Δ = source - target
            source_nodes.append(j)
            target_nodes.append(i)
            dist_km_list.append(d_km)
            delta_lat_list.append(float(lats[j] - lats[i]))
            delta_lon_list.append(float(lons[j] - lons[i]))
            delta_height_list.append(float(heights[j] - heights[i]))

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(
        np.column_stack([dist_km_list, delta_lat_list, delta_lon_list, delta_height_list]),
        dtype=torch.float,
    )

    return edge_index, edge_attr, station_ids


def normalize_edge_attr(edge_attr: torch.Tensor, edge_scaler: dict | None = None):
    """
    Optionally standardize continuous edge features.
    distance_km is always positive; Δlat, Δlon, Δheight can be negative.

    Args:
        edge_attr:     (E, 4) tensor.
        edge_scaler:   dict with 'mean' and 'std' (shape (4,)), or None.
                       If None, computes the statistics and returns them.

    Returns:
        normed_attr:   (E, 4) standardized tensor.
        scaler:        dict with 'mean', 'std'.
    """
    if edge_scaler is None:
        mean = edge_attr.mean(dim=0)
        std = edge_attr.std(dim=0) + 1e-8
        edge_scaler = {"mean": mean, "std": std}

    normed = (edge_attr - edge_scaler["mean"]) / edge_scaler["std"]
    return normed, edge_scaler
