"""
multi_channel_model.py — Full model using the new MultiChannelGraphAttention.

Reuses NodeEncoderMLP and OutputHeadMLP from mpt.py unchanged.
Satisfies the same forward() interface as OffsetMPT:

    model(x, edge_index, edge_attr) -> (N, 2)

Note on edge_attr:
  The new MultiChannelGraphAttention is self-contained and computes attention
  weights from projected node features alone (not edge_attr). The edge_attr
  argument is accepted and ignored so the model remains a transparent
  drop-in replacement inside train.py/inference.py without any callsite changes.
"""

import torch
import torch.nn as nn

from src.models.mpt import NodeEncoderMLP, OutputHeadMLP
from src.models.multi_channel_attention import MultiChannelGraphAttention
from typing import List, Optional


class MultiChannelOffsetModel(nn.Module):
    """
    ERA5 temperature offset model using domain-aware multi-channel graph attention.

    Architecture:
        NodeEncoderMLP (6 → hidden_dim)
          → [MultiChannelGraphAttention + Residual + LayerNorm] × num_gnn_layers
          → OutputHeadMLP (hidden_dim → 2)

    Conceptual message flow inside each layer:
        message_temp(i,j)     — temperature features of neighbours
        message_pressure(i,j) — pressure/humidity features of neighbours
        message_terrain(i,j)  — terrain features of neighbours
        → concat + linear projection → fused node update

    Tensor flow:
        x (N, in_features=6)
          └─▶ NodeEncoderMLP             → (N, hidden_dim)
                └─▶ Encode-first residual block × num_gnn_layers:
                      ┌───────────────────────────────────────┐
                      │  MultiChannelGraphAttention           │
                      │    Temperature channel (idx 0,1)      │
                      │    Pressure channel    (idx 2,4,5)    │
                      │    Terrain channel     (idx 3)        │
                      │    concat + proj → (N, hidden_dim)    │
                      │  + Residual (skip from encoder)       │
                      │  + LayerNorm                          │
                      └───────────────────────────────────────┘
                └─▶ OutputHeadMLP         → (N, 2)
        output: [ΔTmax, ΔTmin] per node

    Note: MultiChannelGraphAttention operates directly on the ORIGINAL node
    features (x) for its channel slicing. The encoded representation (h) is
    used for the residual skip connection and the final output head.

    Args:
        in_features:    Total input node feature dimension (default 6).
        hidden_dim:     Latent dimension throughout (default 64).
        heads:          Unused in this variant (kept for API compatibility with factory).
        num_gnn_layers: Number of attention+residual blocks (default 2).
        edge_dim:       Unused (kept for API compatibility; pass edge_attr through).
        out_dim:        Output dimension — 2 for [ΔTmax, ΔTmin].
        dropout:        Dropout on attention coefficients.
        num_channels:   Number of parallel attention channels (default 3).
        aggregation:    Unused (kept for factory API compatibility; fusion is always concat+proj).
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 64,
        heads: int = 4,              # kept for factory API compatibility
        num_gnn_layers: int = 2,
        edge_dim: int = 4,           # kept for forward() compatibility
        out_dim: int = 2,
        dropout: float = 0.1,
        num_channels: int = 3,
        aggregation: str = "mean",   # kept for factory API compatibility
        # Ablation: inject custom channel feature index groups from factory.
        # None → use MultiChannelGraphAttention defaults (all 3 channels).
        _channel_feature_indices: Optional[List[List[int]]] = None,
        _channel_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Baseline encoder and decoder reused unchanged from mpt.py
        self.node_encoder = NodeEncoderMLP(in_features, hidden_dim, dropout)
        self.output_head = OutputHeadMLP(hidden_dim, out_dim)

        # Multi-channel attention blocks
        self.conv_layers = nn.ModuleList([
            MultiChannelGraphAttention(
                in_dim=in_features,
                hidden_dim=hidden_dim,
                num_channels=num_channels,
                dropout=dropout,
                channel_feature_indices=_channel_feature_indices,  # None → use defaults
                channel_names=_channel_names,
            )
            for _ in range(num_gnn_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,     # accepted, passed through for API compatibility
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        Identical interface to OffsetMPT.forward() — transparent drop-in replacement.

        Args:
            x:           (N, in_features) node feature matrix.
            edge_index:  (2, E) edge connectivity.
            edge_attr:   (E, 4) edge features — accepted but not used internally;
                         channel attention is computed from node features only.
            return_attn: If True, prints attention weight dict (for debugging).

        Returns:
            (N, 2) predicted offsets [ΔTmax, ΔTmin] per node.
        """
        # Encode node features to latent space
        h = self.node_encoder(x)   # (N, hidden_dim)

        # Apply multi-channel attention blocks with residual connections
        # Each block takes the ORIGINAL x for channel slicing,
        # while h carries the latent residual.
        for conv, norm in zip(self.conv_layers, self.norms):
            h_new, attn_dict = conv(x, edge_index, return_attn=return_attn)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)    # residual skip + LayerNorm

        return self.output_head(h)  # (N, 2)
