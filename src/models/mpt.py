"""
mpt.py — Message Passing Transformer for ERA5 temperature offset prediction.

Architecture (concat=False keeps shapes consistent throughout):

    Input node features (N, 6)
         │
         ▼
    [Node Encoder MLP]     6 → hidden_dim
    Linear → GELU → LayerNorm → Linear → Dropout
         │
         ▼  ┌─── residual ──────────────┐
    [TransformerConv 1]   hidden_dim, edge_dim=4, heads=H, concat=False
    [LayerNorm + Dropout]              → (N, hidden_dim)
         │  └───────────────────────────┘
         ▼  ┌─── residual ──────────────┐
    [TransformerConv 2]   same config   → (N, hidden_dim)
    [LayerNorm + Dropout]
         │  └───────────────────────────┘
         ▼
    [Output Head MLP]     hidden_dim → hidden_dim//2 → 2
    Linear → GELU → Linear
         │
         ▼
    Predicted offsets [ΔTmax, ΔTmin] per node  (N, 2)

Notes:
  - concat=False: TransformerConv averages across heads, keeping output at
    hidden_dim (not hidden_dim * heads).  This makes residual connections
    trivial (no projection needed) and is suitable for the small 23-node graph.
  - Residual connections: applied after each TransformerConv block.
  - edge_dim=4 matches the graph_builder.py output: [dist_km, Δlat, Δlon, Δheight].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class NodeEncoderMLP(nn.Module):
    """Maps raw node features to the hidden_dim latent space."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class OutputHeadMLP(nn.Module):
    """Projects hidden_dim representations to [ΔTmax, ΔTmin] predictions."""

    def __init__(self, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        mid = max(hidden_dim // 2, out_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Linear(mid, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class OffsetMPT(nn.Module):
    """
    Message Passing Transformer for station temperature offset prediction.

    Args:
        in_features:    Number of input node features (default 6).
        hidden_dim:     Latent dimension (consistent throughout, since concat=False).
        heads:          Number of attention heads in TransformerConv.
        num_gnn_layers: Number of stacked TransformerConv layers (default 2).
        edge_dim:       Dimension of edge features (default 4).
        out_dim:        Output dimension — 2 for [ΔTmax, ΔTmin].
        dropout:        Dropout rate in encoder and conv layers.
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 64,
        heads: int = 4,
        num_gnn_layers: int = 2,
        edge_dim: int = 4,
        out_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_encoder = NodeEncoderMLP(in_features, hidden_dim, dropout)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            # concat=False → output shape is (N, hidden_dim), same as input
            self.conv_layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,        # average heads → shape stays hidden_dim
                    edge_dim=edge_dim,
                    dropout=dropout,
                    beta=True,           # enable skip-connection inside attention
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.output_head = OutputHeadMLP(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, in_features) node feature matrix.
            edge_index: (2, E) edge connectivity.
            edge_attr:  (E, edge_dim) edge feature matrix.

        Returns:
            (N, 2) predicted offsets [ΔTmax, ΔTmin] per node.
        """
        # Encode
        h = self.node_encoder(x)           # (N, hidden_dim)

        # Message-passing layers with residual connections
        for conv, norm in zip(self.conv_layers, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)  # (N, hidden_dim)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)            # residual + LN

        # Decode
        return self.output_head(h)         # (N, 2)
