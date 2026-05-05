"""
factory.py — Model factory for the ERA5 Offset Prediction pipeline.

Centralises all model instantiation. Training scripts call build_model(cfg)
instead of constructing OffsetMPT directly, enabling model selection via config.

Supported model types:
    "baseline"      — OffsetMPT, now spatiotemporal-capable
    "multi_channel" — MultiChannelOffsetModel (new variant)

Channel ablation:
    Set cfg.model.active_channels to a comma-separated subset of channel names.
    Available names: "temperature", "humidity_stability", "wind", "terrain"
    "all" or "" uses all channels.
"""

import torch.nn as nn
from typing import List

from src.config import Config
from src.models.mpt import OffsetMPT
from src.models.multi_channel_model import MultiChannelOffsetModel

# Canonical feature index groups per channel name for FOG_FEATURE_COLUMNS:
# [mx2t, mn2t, era5_t2m, era5_d2m, UG_era5, dewpoint_spread_2m, rh_2m,
#  era5_u10, era5_v10, wind_speed_10m, theta_v_2m, theta_v_delta_1d,
#  t2m_delta_1d, dewpoint_spread_delta_1d, height, sin_doy, cos_doy]
_FOG_CHANNEL_REGISTRY = {
    "temperature": [0, 1, 2, 3, 12],
    "humidity_stability": [4, 5, 6, 10, 11, 13, 15, 16],
    "wind": [7, 8, 9],
    "terrain": [14],
}
_FOG_ALL_CHANNELS = ["temperature", "humidity_stability", "wind", "terrain"]

_LEGACY_CHANNEL_REGISTRY = {
    "temperature": [0, 1],
    "pressure": [2, 4, 5],
    "terrain": [3],
}
_LEGACY_ALL_CHANNELS = ["temperature", "pressure", "terrain"]


def _resolve_channels(active_channels: str, in_features: int):
    """
    Parse active_channels string → (channel_names, channel_feature_indices).

    Args:
        active_channels: "all" | "" | comma-separated channel names
                         e.g. "temperature,humidity_stability" or "terrain"

    Returns:
        channel_names:           list[str]
        channel_feature_indices: list[list[int]]

    Raises:
        ValueError: if an unknown channel name is given.
    """
    if in_features == 6:
        registry = _LEGACY_CHANNEL_REGISTRY
        all_channels = _LEGACY_ALL_CHANNELS
    else:
        registry = _FOG_CHANNEL_REGISTRY
        all_channels = _FOG_ALL_CHANNELS

    raw = active_channels.strip().lower()
    if raw in ("all", "", "all channels"):
        names = all_channels
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if in_features != 6:
            names = ["humidity_stability" if n == "pressure" else n for n in names]

    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(
            f"Unknown channel name(s): {unknown}. "
            f"Valid names: {list(registry.keys())}"
        )

    indices = [registry[n] for n in names]
    return names, indices


def build_model(cfg: Config, dropout_override: float | None = None) -> nn.Module:
    """
    Instantiate and return the correct model based on cfg.model.model_type.

    Args:
        cfg:              Central config object (src/config.py).
        dropout_override: If provided, overrides cfg.model.dropout.
                          Pass 0.0 at inference time to disable dropout.

    Returns:
        nn.Module with forward(x, edge_index, edge_attr) → (N, 2).

    Raises:
        ValueError: If cfg.model.model_type is not a recognised key,
                    or if an unknown channel name is specified.
    """
    dropout = dropout_override if dropout_override is not None else cfg.model.dropout
    model_type = cfg.model.model_type.lower()

    if model_type == "baseline":
        return OffsetMPT(
            in_features=cfg.model.in_features,
            hidden_dim=cfg.model.hidden_dim,
            heads=cfg.model.heads,
            num_gnn_layers=cfg.model.num_gnn_layers,
            temporal_layers=cfg.model.temporal_layers,
            max_seq_len=cfg.model.max_seq_len,
            temporal_pooling=cfg.model.temporal_pooling,
            edge_dim=cfg.model.edge_dim,
            out_dim=cfg.model.out_dim,
            enable_fog_head=cfg.model.enable_fog_head,
            fog_out_dim=cfg.model.fog_out_dim,
            dropout=dropout,
        )

    elif model_type == "multi_channel":
        channel_names, channel_feature_indices = _resolve_channels(
            cfg.model.active_channels,
            cfg.model.in_features,
        )
        num_channels = len(channel_names)

        return MultiChannelOffsetModel(
            in_features=cfg.model.in_features,
            hidden_dim=cfg.model.hidden_dim,
            heads=cfg.model.heads,
            num_gnn_layers=cfg.model.num_gnn_layers,
            edge_dim=cfg.model.edge_dim,
            out_dim=cfg.model.out_dim,
            dropout=dropout,
            num_channels=num_channels,
            aggregation=cfg.model.aggregation,
            # Inject resolved feature index groups into the attention layer
            _channel_feature_indices=channel_feature_indices,
            _channel_names=channel_names,
        )

    else:
        supported = ["baseline", "multi_channel"]
        raise ValueError(
            f"Unknown model_type '{cfg.model.model_type}'. "
            f"Supported types: {supported}"
        )
