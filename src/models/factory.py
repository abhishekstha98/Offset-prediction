"""
factory.py — Model factory for the ERA5 Offset Prediction pipeline.

Centralises all model instantiation. Training scripts call build_model(cfg)
instead of constructing OffsetMPT directly, enabling model selection via config.

Supported model types:
    "baseline"      — original OffsetMPT (unchanged)
    "multi_channel" — MultiChannelOffsetModel (new variant)

Channel ablation:
    Set cfg.model.active_channels to a comma-separated subset of channel names.
    Available names: "temperature", "pressure", "terrain"
    Example: "temperature,pressure" removes the terrain channel.
    "all" or "" uses all three channels.
"""

import torch.nn as nn
from typing import List

from src.config import Config
from src.models.mpt import OffsetMPT
from src.models.multi_channel_model import MultiChannelOffsetModel

# Canonical feature index groups per channel name
# Node features: [mx2t(0), mn2t(1), UG_era5(2), height(3), sin_doy(4), cos_doy(5)]
_CHANNEL_REGISTRY = {
    "temperature": [0, 1],       # mx2t, mn2t
    "pressure":    [2, 4, 5],    # UG_era5, sin_doy, cos_doy
    "terrain":     [3],          # height
}
_ALL_CHANNELS = ["temperature", "pressure", "terrain"]


def _resolve_channels(active_channels: str):
    """
    Parse active_channels string → (channel_names, channel_feature_indices).

    Args:
        active_channels: "all" | "" | comma-separated channel names
                         e.g. "temperature,pressure" or "terrain"

    Returns:
        channel_names:           list[str]
        channel_feature_indices: list[list[int]]

    Raises:
        ValueError: if an unknown channel name is given.
    """
    raw = active_channels.strip().lower()
    if raw in ("all", "", "all channels"):
        names = _ALL_CHANNELS
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]

    unknown = [n for n in names if n not in _CHANNEL_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown channel name(s): {unknown}. "
            f"Valid names: {list(_CHANNEL_REGISTRY.keys())}"
        )

    indices = [_CHANNEL_REGISTRY[n] for n in names]
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
            edge_dim=cfg.model.edge_dim,
            out_dim=cfg.model.out_dim,
            dropout=dropout,
        )

    elif model_type == "multi_channel":
        channel_names, channel_feature_indices = _resolve_channels(cfg.model.active_channels)
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
