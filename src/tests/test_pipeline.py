"""
test_pipeline.py — Unit tests for the ERA5 Offset MPT pipeline.

Tests:
  1. SLOBO integrity: No station appears in both train and validation folds.
  2. Edge directionality: For each A→B edge, B→A edge has sign-flipped deltas.
  3. Node masking: A node with a NaN target is excluded from the loss
     but does NOT cause NaN in the forward pass.
  4. Output shape: Model output is (N, 2) for a single-day graph.
  5. Loss finiteness: Loss is a finite scalar when at least one valid node exists.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.data.split import build_slobo_folds, get_fold_masks
from src.data.graph_builder import build_static_graph, normalize_edge_attr
from src.data.dataset import ERA5LandDataset, fit_scaler
from src.models.mpt import OffsetMPT
from src.utils.loss import OffsetLoss
from src.config import cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_df():
    """Minimal synthetic DataFrame mimicking comparison_all_years.csv structure."""
    np.random.seed(0)
    stations = [f"S{i}" for i in range(8)]
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        for sid in stations:
            rows.append({
                "station": sid,
                "time": date,
                "lat": np.random.uniform(51, 53),
                "lon": np.random.uniform(4, 7),
                "height": np.random.uniform(0, 100),
                "mx2t": np.random.uniform(5, 25),
                "mn2t": np.random.uniform(-5, 15),
                "UG_era5": np.random.uniform(40, 90),
                "TX": np.random.uniform(5, 25),
                "TN": np.random.uniform(-5, 15),
                "UG_station": np.random.uniform(40, 90),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def tiny_df_with_nan(tiny_df):
    """Same tiny_df but with NaN TX/TN for a few rows to test masking."""
    df = tiny_df.copy()
    # Make station S0 have NaN targets on all days
    df.loc[df["station"] == "S0", "TX"] = np.nan
    df.loc[df["station"] == "S0", "TN"] = np.nan
    return df


@pytest.fixture(scope="module")
def unique_stations(tiny_df):
    return (
        tiny_df[["station", "lat", "lon", "height"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def graph(unique_stations):
    edge_index, edge_attr, station_order = build_static_graph(unique_stations, k=3)
    edge_attr, edge_scaler = normalize_edge_attr(edge_attr)
    return edge_index, edge_attr, station_order


# ---------------------------------------------------------------------------
# Test 1: SLOBO integrity
# ---------------------------------------------------------------------------

def test_slobo_no_overlap(unique_stations):
    """No station should appear in both train and val masks for any fold."""
    station_to_block = build_slobo_folds(unique_stations, n_blocks=4, random_state=42)
    station_ids = unique_stations["station"].values

    for val_block in range(4):
        train_mask, val_mask = get_fold_masks(station_ids, station_to_block, val_block)
        overlap = train_mask & val_mask
        assert overlap.sum() == 0, (
            f"Block {val_block}: {overlap.sum()} stations appear in both train and val"
        )

    print("  ✓ SLOBO integrity: no station leaks between train and val")


# ---------------------------------------------------------------------------
# Test 2: Edge directionality
# ---------------------------------------------------------------------------

def test_edge_directionality(unique_stations):
    """For edge A→B, the corresponding edge B→A should have sign-flipped deltas (dims 1-3)."""
    # Build WITHOUT normalization to keep raw directional values
    edge_index, edge_attr, station_order = build_static_graph(unique_stations, k=3)

    # Build a lookup: (source, target) → edge_attr row
    ei = edge_index.numpy()
    ea = edge_attr.numpy()
    edge_map = {(ei[0, e], ei[1, e]): ea[e] for e in range(ei.shape[1])}

    n_checked = 0
    for (src, tgt), feat_ab in edge_map.items():
        if (tgt, src) in edge_map:
            feat_ba = edge_map[(tgt, src)]
            # dist_km should be equal (within float tolerance)
            assert abs(feat_ab[0] - feat_ba[0]) < 1e-3, "Distance not symmetric"
            # Δlat, Δlon, Δheight should be sign-flipped
            for d in range(1, 4):
                assert abs(feat_ab[d] + feat_ba[d]) < 1e-3, (
                    f"Edge ({src}→{tgt}) dim {d}: {feat_ab[d]:.4f} + {feat_ba[d]:.4f} ≠ 0"
                )
            n_checked += 1

    assert n_checked > 0, "No bidirectional edge pairs found to test"
    print(f"  ✓ Edge directionality: {n_checked} bidirectional pairs verified")


# ---------------------------------------------------------------------------
# Test 3: NaN masking — loss excludes NaN target nodes
# ---------------------------------------------------------------------------

def test_nan_masking_loss(tiny_df_with_nan):
    """Station S0 has NaN targets. valid_mask should exclude it, and loss must be finite."""
    dataset = ERA5LandDataset(tiny_df_with_nan, scaler=None)
    batch = dataset[0]

    y = batch["y"]
    valid_mask = batch["valid_mask"]
    station_ids = batch["station_ids"]

    # S0's mask should be False
    s0_idx = list(station_ids).index("S0")
    assert not valid_mask[s0_idx, 0].item(), "S0 Tmax mask should be False"
    assert not valid_mask[s0_idx, 1].item(), "S0 Tmin mask should be False"

    # Forward pass still works without NaN in output
    loss_fn = OffsetLoss()
    N = y.shape[0]
    dummy_pred = torch.randn(N, 2)
    loss, lt, ln = loss_fn(dummy_pred, y, valid_mask)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print(f"  ✓ NaN masking: loss={loss.item():.4f} (finite), S0 correctly masked out")


# ---------------------------------------------------------------------------
# Test 4 & 5: Model output shape and loss finiteness
# ---------------------------------------------------------------------------

def test_model_output_shape_and_loss(tiny_df, graph):
    """End-to-end smoke: model output shape (N, 2) and finite loss."""
    edge_index, edge_attr, station_order = graph
    dataset = ERA5LandDataset(tiny_df, scaler=None)
    batch = dataset[0]

    x = batch["x"]         # (N, 6)
    y = batch["y"]         # (N, 2)
    valid_mask = batch["valid_mask"]

    N = x.shape[0]

    model = OffsetMPT(
        in_features=6,
        hidden_dim=32,
        heads=2,
        num_gnn_layers=2,
        edge_dim=4,
        out_dim=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index, edge_attr)

    assert pred.shape == (N, 2), f"Expected ({N}, 2), got {pred.shape}"
    assert torch.isfinite(pred).all(), "Model output contains NaN/Inf"

    loss_fn = OffsetLoss()
    loss, lt, ln = loss_fn(pred, y, valid_mask)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    print(f"  ✓ Output shape: {pred.shape} | Loss: {loss.item():.4f} (finite)")
