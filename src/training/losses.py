"""
Loss functions for time series forecasting.
"""

import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss).
    
    Less sensitive to outliers than MSE.
    Quadratic for small errors, linear for large errors.
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Threshold for switching from L2 to L1 (default: 1.0)
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth L1 loss.
        
        Args:
            predictions: Predicted values, shape (batch, horizon)
            targets: True values, same shape
            
        Returns:
            Scalar loss
        """
        return nn.functional.smooth_l1_loss(predictions, targets, beta=self.beta)


class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(predictions, targets)


class MAELoss(nn.Module):
    """Mean Absolute Error loss (L1 loss)."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.l1_loss(predictions, targets)


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_name: Name of loss ('smooth_l1', 'mse', 'mae')
        **kwargs: Additional arguments for loss
        
    Returns:
        Loss module
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'smooth_l1':
        beta = kwargs.get('beta', 1.0)
        return SmoothL1Loss(beta=beta)
    elif loss_name == 'mse':
        return MSELoss()
    elif loss_name == 'mae':
        return MAELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

