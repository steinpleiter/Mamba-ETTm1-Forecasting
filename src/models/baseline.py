"""
Baseline models for time series forecasting.

Implements:
- SeasonalNaive: Copy values from seasonal_period steps ago
- DLinear: Decomposition + Linear layers (trend + seasonal)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SeasonalNaive(nn.Module):
    """
    Seasonal Naive baseline forecaster.
    Forecast: y_{t+h} = y_{t+h-m} where m is seasonal_period.
    For ETTm1 with 15-min data, seasonal_period=96 gives daily seasonality.
    """
    
    def __init__(self, seasonal_period: int = 96, forecast_horizon: int = 96):
        """
        Args:
            seasonal_period: Period for seasonal pattern (default: 96 for daily)
            forecast_horizon: Number of steps to forecast (H)
        """
        super().__init__()
        self.seasonal_period = seasonal_period
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate seasonal naive forecast.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, seq_len)
            
        Returns:
            Forecast of shape (batch, forecast_horizon)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Handle both (batch, seq, features) and (batch, seq)
        if len(x.shape) == 3:
            # Take last feature (OT is last column in ETTm1)
            x = x[:, :, -1]  # (batch, seq_len)
        
        if seq_len < self.seasonal_period:
            raise ValueError(
                f"Input sequence length ({seq_len}) must be >= seasonal_period ({self.seasonal_period})"
            )
        
        # Generate forecast by copying from seasonal_period steps ago
        forecast = []
        for h in range(self.forecast_horizon):
            # Index from the end: -seasonal_period + (h % seasonal_period)
            idx = -(self.seasonal_period - (h % self.seasonal_period))
            forecast.append(x[:, idx])
        
        forecast = torch.stack(forecast, dim=1)  # (batch, forecast_horizon)
        return forecast


class MovingAverage(nn.Module):
    """Moving average smoothing for trend-seasonal decomposition."""
    
    def __init__(self, kernel_size: int = 25):
        """
        Args:
            kernel_size: Size of moving average window (should be odd)
        """
        super().__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd")
        
        # Create moving average kernel
        self.padding = (kernel_size - 1) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply moving average.
        
        Args:
            x: Input of shape (batch, seq_len, features)
            
        Returns:
            Smoothed output of same shape
        """
        # x: (batch, seq_len, features)
        batch_size, seq_len, num_features = x.shape
        
        # Apply avg pooling along time dimension for each feature separately
        x_padded = torch.nn.functional.pad(
            x.transpose(1, 2),  # (batch, features, seq_len)
            (self.padding, self.padding),
            mode='replicate'
        )
        
        # Apply average pooling
        smoothed = torch.nn.functional.avg_pool1d(
            x_padded,
            kernel_size=self.kernel_size,
            stride=1
        )
        
        return smoothed.transpose(1, 2)  # (batch, seq_len, features)


class SeriesDecomposition(nn.Module):
    """Decompose time series into trend and seasonal components."""
    
    def __init__(self, kernel_size: int = 25):
        """
        Args:
            kernel_size: Size of moving average for trend extraction
        """
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose series into trend and seasonal.
        
        Args:
            x: Input of shape (batch, seq_len, features)
            
        Returns:
            Tuple of (seasonal, trend), each of shape (batch, seq_len, features)
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear model.
    Based on "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2022).
    Decomposes input into trend + seasonal, applies separate linear layers, then combines.
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
        enc_in: int = 7,
        kernel_size: int = 25
    ):
        """
        Args:
            seq_len: Input sequence length (L)
            pred_len: Prediction length (H)
            individual: If True, use separate linear layers per channel (CI mode)
            enc_in: Number of input features/channels
            kernel_size: Kernel size for moving average decomposition
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        
        # Decomposition
        self.decomposition = SeriesDecomposition(kernel_size)
        
        if individual:
            # Channel-independent: separate linear layer per channel
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            # Shared linear layers
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, features)
            
        Returns:
            Predictions of shape (batch, pred_len) for target variable (last feature)
        """
        # Decompose
        seasonal, trend = self.decomposition(x)
        
        # seasonal, trend: (batch, seq_len, features)
        
        if self.individual:
            # Channel-independent mode
            seasonal_output = []
            trend_output = []
            
            for i in range(self.enc_in):
                # Process each channel separately
                s_i = seasonal[:, :, i]  # (batch, seq_len)
                t_i = trend[:, :, i]     # (batch, seq_len)
                
                seasonal_output.append(self.linear_seasonal[i](s_i))
                trend_output.append(self.linear_trend[i](t_i))
            
            seasonal_output = torch.stack(seasonal_output, dim=-1).contiguous()  # (batch, pred_len, features)
            trend_output = torch.stack(trend_output, dim=-1).contiguous()        # (batch, pred_len, features)
        else:
            # Shared mode: apply same linear to all channels
            # Transpose to (batch, features, seq_len) for linear layer
            seasonal_t = seasonal.transpose(1, 2)  # (batch, features, seq_len)
            trend_t = trend.transpose(1, 2)        # (batch, features, seq_len)
            
            seasonal_output = self.linear_seasonal(seasonal_t)  # (batch, features, pred_len)
            trend_output = self.linear_trend(trend_t)           # (batch, features, pred_len)
            
            # Transpose back
            seasonal_output = seasonal_output.transpose(1, 2)  # (batch, pred_len, features)
            trend_output = trend_output.transpose(1, 2)        # (batch, pred_len, features)
        
        # Combine
        output = seasonal_output + trend_output  # (batch, pred_len, features)
        
        # Return only target variable (last feature = OT)
        # Use contiguous() to fix memory layout after transpose operations
        return output[:, :, -1].contiguous()  # (batch, pred_len)


class LinearRegression(nn.Module):
    """Simple linear regression baseline. Just a linear layer: y = Wx + b."""
    
    def __init__(self, seq_len: int, pred_len: int, enc_in: int = 7):
        """
        Args:
            seq_len: Input sequence length (L)
            pred_len: Prediction length (H)
            enc_in: Number of input features
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # Flatten input and map to output
        self.linear = nn.Linear(seq_len * enc_in, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, features)
            
        Returns:
            Predictions of shape (batch, pred_len)
        """
        batch_size = x.shape[0]
        
        # Flatten
        x_flat = x.reshape(batch_size, -1)  # (batch, seq_len * features)
        
        # Linear mapping
        output = self.linear(x_flat)  # (batch, pred_len)
        
        return output


class LastValue(nn.Module):
    """Last value baseline - repeat the last observed value."""
    
    def __init__(self, forecast_horizon: int = 96):
        """
        Args:
            forecast_horizon: Number of steps to forecast (H)
        """
        super().__init__()
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate forecast by repeating last value.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, seq_len)
            
        Returns:
            Forecast of shape (batch, forecast_horizon)
        """
        # Handle both (batch, seq, features) and (batch, seq)
        if len(x.shape) == 3:
            # Take last feature (OT)
            last_value = x[:, -1, -1]  # (batch,)
        else:
            last_value = x[:, -1]  # (batch,)
        
        # Repeat for forecast horizon
        forecast = last_value.unsqueeze(1).repeat(1, self.forecast_horizon)
        
        return forecast


def create_baseline_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    enc_in: int = 7,
    **kwargs
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_name: Name of model ('seasonal_naive', 'dlinear', 'linear', 'last_value')
        seq_len: Input sequence length
        pred_len: Prediction length
        enc_in: Number of input features
        **kwargs: Additional model-specific arguments
        
    Returns:
        Baseline model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'seasonal_naive':
        seasonal_period = kwargs.get('seasonal_period', 96)
        return SeasonalNaive(seasonal_period=seasonal_period, forecast_horizon=pred_len)
    
    elif model_name == 'dlinear':
        individual = kwargs.get('individual', False)
        kernel_size = kwargs.get('kernel_size', 25)
        return DLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
            enc_in=enc_in,
            kernel_size=kernel_size
        )
    
    elif model_name == 'linear':
        return LinearRegression(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    
    elif model_name == 'last_value':
        return LastValue(forecast_horizon=pred_len)
    
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

