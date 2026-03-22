"""
multi_channel_attention.py — Self-contained multi-channel graph attention layer.

Implements domain-aware parallel attention channels, each dedicated to a specific
group of environmental variables:

    Channel 0  — Temperature   : mx2t, mn2t            (feature indices 0, 1)
    Channel 1  — Pressure/Humid: UG_era5, sin_doy, cos  (feature indices 2, 4, 5)
    Channel 2  — Terrain       : height                 (feature index 3)

Each channel has:
  - Its own linear projection from channel-specific input features → hidden_dim.
  - Its own attention weight computation (source + target → scalar α).
  - Independent softmax-normalised message aggregation over neighbours.

After per-channel aggregation, all channel outputs are concatenated and projected
back to hidden_dim via a learnable linear layer.

No dependency on existing attention implementations. Compatible with PyTorch
Geometric via the MessagePassing base class.

Usage:
    layer = MultiChannelGraphAttention(in_dim=6, hidden_dim=64, num_channels=3)
    out, attn = layer(x, edge_index)        # attn is optional
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ---------------------------------------------------------------------------
# Default channel feature slices for the 6-feature ERA5 node vector
# [mx2t, mn2t, UG_era5, height, sin_doy, cos_doy]
# ---------------------------------------------------------------------------
DEFAULT_CHANNEL_FEATURE_INDICES: List[List[int]] = [
    [0, 1],         # Temperature channel  — mx2t, mn2t
    [2, 4, 5],      # Pressure/Humid channel — UG_era5, sin_doy, cos_doy
    [3],            # Terrain channel      — height
]

DEFAULT_CHANNEL_NAMES: List[str] = [
    "temperature",
    "pressure",
    "terrain",
]


# ---------------------------------------------------------------------------
# Single-channel attention layer (MessagePassing subclass)
# ---------------------------------------------------------------------------

class _SingleChannelAttention(MessagePassing):
    """
    One attention channel operating on a specific subset of node features.

    Implements scaled dot-product-style attention:
        e(i,j) = LeakyReLU( [h_i || h_j] · a )   (unnormalised score)
        α(i,j) = softmax over j∈N(i) of e(i,j)   (normalised per target node)
        m_i   = Σ_j  α(i,j) · W_v · h_j_full      (weighted message)

    where h_* are the projected channel-specific features, and the value
    projection W_v operates on the same projected features.

    Args:
        channel_in_dim: Number of input features for this channel.
        hidden_dim:     Projection / output dimension.
        dropout:        Dropout applied to attention coefficients.
    """

    def __init__(self, channel_in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")

        # Project channel features to hidden space (query/key)
        self.proj_q = nn.Linear(channel_in_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(channel_in_dim, hidden_dim, bias=False)
        # Value projection
        self.proj_v = nn.Linear(channel_in_dim, hidden_dim, bias=False)
        # Attention scoring: concatenated q||k → scalar
        self.attn_score = nn.Linear(2 * hidden_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self._last_attn_weights: Optional[torch.Tensor] = None  # for inspection

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.proj_k.weight, gain=1.0)
        nn.init.xavier_uniform_(self.proj_v.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attn_score.weight, gain=1.0)

    def forward(
        self,
        x_channel: torch.Tensor,   # (N, channel_in_dim)
        edge_index: torch.Tensor,  # (2, E)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_channel:   Node features for this channel only. Shape (N, channel_in_dim).
            edge_index:  Graph connectivity. Shape (2, E).
            return_attn: Whether to return attention weights per edge.

        Returns:
            out:  Aggregated node embeddings. Shape (N, hidden_dim).
            attn: Attention weights per edge (E,) if return_attn else None.
        """
        q = self.proj_q(x_channel)   # (N, hidden_dim)
        k = self.proj_k(x_channel)   # (N, hidden_dim)
        v = self.proj_v(x_channel)   # (N, hidden_dim)

        # propagate hands q, k, v to message()
        out = self.propagate(edge_index, q=q, k=k, v=v, size=None)

        attn = self._last_attn_weights if return_attn else None
        return out, attn

    def message(
        self,
        q_i: torch.Tensor,   # (E, hidden_dim)  — target query
        k_j: torch.Tensor,   # (E, hidden_dim)  — source key
        v_j: torch.Tensor,   # (E, hidden_dim)  — source value
        index: torch.Tensor, # edge target indices for softmax normalisation
    ) -> torch.Tensor:
        """Compute attention-weighted messages."""
        # Score: concat target query + source key → scalar
        score = self.attn_score(torch.cat([q_i, k_j], dim=-1))  # (E, 1)
        score = F.leaky_relu(score, negative_slope=0.2)

        # Softmax normalised per target node
        alpha = softmax(score, index=index, dim=0)               # (E, 1)
        alpha = self.dropout(alpha)

        # Cache for inspection
        self._last_attn_weights = alpha.squeeze(-1).detach()

        return alpha * v_j  # (E, hidden_dim)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Apply non-linearity after aggregation."""
        return F.elu(aggr_out)


# ---------------------------------------------------------------------------
# Multi-Channel Graph Attention (public API)
# ---------------------------------------------------------------------------

class MultiChannelGraphAttention(nn.Module):
    """
    Domain-aware multi-channel graph attention for environmental station data.

    Runs N independent attention channels, each operating on a different group
    of environmental features. Outputs are concatenated and projected back to
    the original hidden dimension.

    Default 3-channel split (for 6-feature ERA5 node vectors):
      - Temperature channel  : mx2t, mn2t            (feature indices 0, 1)
      - Pressure/Humid channel: UG_era5, sin_doy, cos (feature indices 2, 4, 5)
      - Terrain channel      : height                 (feature index 3)

    Args:
        in_dim:                  Total input node feature dimension (e.g. 6).
        hidden_dim:              Per-channel hidden dimension and final output dimension.
        num_channels:            Number of parallel channels. Must match
                                 len(channel_feature_indices) if that is provided.
        dropout:                 Dropout on attention coefficients.
        channel_feature_indices: List of index lists defining which input features
                                 each channel uses. Defaults to DEFAULT_CHANNEL_FEATURE_INDICES.
        channel_names:           Human-readable names for each channel (for logging).

    Constructor example:
        layer = MultiChannelGraphAttention(in_dim=6, hidden_dim=64, num_channels=3)

    Forward signature:
        out, attn_dict = layer(node_features, edge_index, return_attn=False)

        node_features: (N, in_dim)   — full node feature matrix
        edge_index:    (2, E)        — edge connectivity
        return_attn:   bool          — whether to return per-channel attention weights

    Output:
        out:       (N, hidden_dim)   — updated node embeddings after fusion
        attn_dict: dict[str, Tensor] — per-channel attention weights (E,), or {}
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_channels: int = 3,
        dropout: float = 0.1,
        channel_feature_indices: Optional[List[List[int]]] = None,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__()

        # Resolve feature index groups
        if channel_feature_indices is None:
            channel_feature_indices = DEFAULT_CHANNEL_FEATURE_INDICES[:num_channels]

        if len(channel_feature_indices) != num_channels:
            raise ValueError(
                f"len(channel_feature_indices)={len(channel_feature_indices)} "
                f"must equal num_channels={num_channels}"
            )

        self.channel_feature_indices = channel_feature_indices
        self.channel_names = (
            channel_names[:num_channels]
            if channel_names
            else DEFAULT_CHANNEL_NAMES[:num_channels]
        )
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # Instantiate one independent attention channel per feature group
        self.channels = nn.ModuleList([
            _SingleChannelAttention(
                channel_in_dim=len(indices),
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for indices in channel_feature_indices
        ])

        # Fusion: concatenated channel outputs → hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(num_channels * hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,   # (N, in_dim)
        edge_index: torch.Tensor,      # (2, E)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Message flow per channel:
            message_temp(i,j)     — temperature features only
            message_pressure(i,j) — pressure/humidity features only
            message_terrain(i,j)  — terrain features only

        Each channel independently aggregates messages, then all outputs are
        fused via concatenation and linear projection.

        Args:
            node_features: (N, in_dim)  Full node feature matrix.
            edge_index:    (2, E)       Graph connectivity.
            return_attn:   bool         Return per-channel attention weights.

        Returns:
            updated_node_embeddings: (N, hidden_dim)
            attn_dict: dict mapping channel name → attention weights (E,) per edge.
                       Empty dict if return_attn=False.
        """
        channel_outputs = []
        attn_dict = {}

        for name, indices, channel_layer in zip(
            self.channel_names, self.channel_feature_indices, self.channels
        ):
            # Slice the features relevant to this channel
            x_ch = node_features[:, indices]                     # (N, channel_in_dim)
            out_ch, attn_ch = channel_layer(x_ch, edge_index, return_attn=return_attn)
            channel_outputs.append(out_ch)                       # (N, hidden_dim)

            if return_attn and attn_ch is not None:
                attn_dict[name] = attn_ch                        # (E,)

        # Fuse: (N, num_channels * hidden_dim) → (N, hidden_dim)
        fused = torch.cat(channel_outputs, dim=-1)               # (N, K * hidden_dim)
        updated_node_embeddings = self.fusion(fused)             # (N, hidden_dim)

        return updated_node_embeddings, attn_dict
