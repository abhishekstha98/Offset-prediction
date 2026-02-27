"""
loss.py — OffsetLoss for ERA5 temperature offset prediction.

Computes masked MAE separately for ΔTmax and ΔTmin, then combines with
configurable weights (λ_tmax, λ_tmin).

valid_mask (N, 2) is critical:
  - mask[:, 0] = True where offset_tmax is a valid (non-NaN) target.
  - mask[:, 1] = True where offset_tmin is a valid (non-NaN) target.

Nodes with False in the mask still participate in message passing but their
contribution is excluded from the loss to avoid training on imputed zeros.
"""

import torch
import torch.nn as nn


class OffsetLoss(nn.Module):
    """
    Masked MAE loss for temperature offset prediction.

    Args:
        lambda_tmax: Weight on the Tmax offset component.
        lambda_tmin: Weight on the Tmin offset component.
    """

    def __init__(self, lambda_tmax: float = 1.0, lambda_tmin: float = 1.0):
        super().__init__()
        self.lambda_tmax = lambda_tmax
        self.lambda_tmin = lambda_tmin

    def forward(
        self,
        pred: torch.Tensor,          # (N, 2) predicted [ΔTmax, ΔTmin]
        target: torch.Tensor,        # (N, 2) true      [ΔTmax, ΔTmin]
        valid_mask: torch.Tensor,    # (N, 2) bool, True = include in loss
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss:  Weighted sum of masked MAEs.
            loss_tmax:   Per-component MAE for Tmax (for logging).
            loss_tmin:   Per-component MAE for Tmin (for logging).
        """
        mask_tmax = valid_mask[:, 0]
        mask_tmin = valid_mask[:, 1]

        # If no valid targets exist for a component, return 0 to avoid NaN
        loss_tmax = self._masked_mae(pred[:, 0], target[:, 0], mask_tmax)
        loss_tmin = self._masked_mae(pred[:, 1], target[:, 1], mask_tmin)

        total = self.lambda_tmax * loss_tmax + self.lambda_tmin * loss_tmin
        return total, loss_tmax, loss_tmin

    @staticmethod
    def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute MAE only over masked-in elements."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (pred[mask] - target[mask]).abs().mean()
