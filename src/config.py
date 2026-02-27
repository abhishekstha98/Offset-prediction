"""
config.py — Centralized hyperparameter configuration for the ERA5 Offset MPT pipeline.
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Node feature dimension (mx2t, mn2t, UG_era5, height, sin_doy, cos_doy)
    in_features: int = 6
    # GNN hidden dimension (consistent through all layers because concat=False)
    hidden_dim: int = 64
    # Number of attention heads (concat=False → output is still hidden_dim)
    heads: int = 4
    # Number of TransformerConv message-passing layers
    num_gnn_layers: int = 2
    # Output dimension: [ΔTmax, ΔTmin]
    out_dim: int = 2
    # Dropout rate for node encoder and output head
    dropout: float = 0.1
    # Edge feature dimension (distance_km, Δlat, Δlon, Δheight)
    edge_dim: int = 4


@dataclass
class GraphConfig:
    # k for k-NN graph construction.
    # NOTE: With 23 Netherlands stations, k=5 connects each node to ~22% of the network.
    # Start with k=3; raise to 5 only if validation MAE plateaus.
    # When transferring to Nepal, retune k from scratch.
    k: int = 3


@dataclass
class SplitConfig:
    # Number of spatial K-Means blocks for SLOBO cross-validation.
    # For 23 stations, K=4 gives ~5-6 stations per fold.
    # Use K=4 or K=5; never exceed K=5 for this dataset size.
    n_blocks: int = 4
    # Number of temporal windows for ST-LOBO (T=2 recommended for 9 trainval years).
    # Window 0: 2015-2018, Window 1: 2019-2023.
    n_windows: int = 2
    # Years >= test_year are the final hold-out test set (never seen during training/val).
    # Dataset: 2015-2025. trainval = 2015-2023, test = 2024-2025.
    # NOTE: Previous value of 2022 stranded 2022-2023 outside both splits — corrected.
    test_year: int = 2024


@dataclass
class LossConfig:
    # MAE weight on ΔTmax component of the loss
    lambda_tmax: float = 1.0
    # MAE weight on ΔTmin component of the loss
    lambda_tmin: float = 1.0


@dataclass
class TrainConfig:
    data_path: str = "d:/Offset Prediction Research/comparison_all_years.csv"
    checkpoint_dir: str = "d:/Offset Prediction Research/checkpoints"
    scaler_path: str = "d:/Offset Prediction Research/checkpoints/scaler.pkl"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    # Run one SLOBO fold (0-indexed) or -1 to run all folds
    fold: int = 0
    # Use ST-LOBO (spatial × temporal blocking) instead of pure SLOBO.
    # False = SLOBO (primary for Netherlands)
    # True  = ST-LOBO (stretch goal for Netherlands, primary for Nepal)
    st_lobo: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Singleton default config — import this everywhere
cfg = Config()
