"""
Mamba-based Time Series Forecasting Model.

Based on TSMamba paper: "A Mamba Foundation Model for Time Series Forecasting"
Architecture:
- Forward and backward Mamba encoders for temporal dependencies
- Channel-independent (CI) processing
- 1D conv patching for input embedding
- Compressed prediction head
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")


class PatchEmbedding(nn.Module):
    """Embed time series into patches using 1D convolution."""
    
    def __init__(self, patch_len: int = 16, d_model: int = 128, in_channels: int = 1):
        """
        Args:
            patch_len: Length of each patch
            d_model: Model dimension (embedding size)
            in_channels: Number of input channels (1 for CI mode)
        """
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        
        # 1D conv with stride=patch_len (non-overlapping patches)
        self.projection = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_len
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels)
            
        Returns:
            Patches: (batch, n_patches, d_model)
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
        x = self.projection(x)  # (batch, d_model, n_patches)
        x = x.transpose(1, 2)   # (batch, n_patches, d_model)
        return x


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks with normalization."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of Mamba layers
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor for inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed. Run: pip install mamba-ssm")
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                ),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
            reverse: If True, process in reverse order (backward encoder)
            
        Returns:
            Output: (batch, n_patches, d_model)
        """
        if reverse:
            # Flip sequence for backward processing
            x = torch.flip(x, dims=[1])
        
        for layer in self.layers:
            # Residual connection with LayerNorm
            residual = x
            x = layer['norm'](x)
            x = layer['mamba'](x)
            x = layer['dropout'](x)
            x = x + residual
        
        if reverse:
            # Flip back
            x = torch.flip(x, dims=[1])
        
        return x


class PredictionHead(nn.Module):
    """Compressed prediction head that maps patches to forecast horizon."""
    
    def __init__(
        self,
        d_model: int,
        n_patches: int,
        pred_len: int,
        compress_dim: int = 64
    ):
        """
        Args:
            d_model: Model dimension
            n_patches: Number of patches
            pred_len: Prediction horizon
            compress_dim: Compressed dimension
        """
        super().__init__()
        
        # Compress model dimension
        self.compress = nn.Sequential(
            nn.Linear(d_model, compress_dim),
            nn.GELU()
        )
        
        # Map compressed patches to forecast
        self.head = nn.Linear(n_patches * compress_dim, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
            
        Returns:
            Forecast: (batch, pred_len)
        """
        batch_size = x.shape[0]
        
        # Compress
        x = self.compress(x)  # (batch, n_patches, compress_dim)
        
        # Flatten and predict
        x = x.reshape(batch_size, -1)  # (batch, n_patches * compress_dim)
        x = self.head(x)  # (batch, pred_len)
        
        return x


class MambaForecaster(nn.Module):
    """
    Mamba-based time series forecasting model.
    
    Architecture:
    1. Patch embedding (1D conv)
    2. Forward Mamba encoder
    3. Backward Mamba encoder  
    4. Temporal alignment conv
    5. Compressed prediction head
    
    Processes each channel independently (CI mode).
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        pred_len: int = 96,
        enc_in: int = 7,
        d_model: int = 128,
        n_layers: int = 4,
        patch_len: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        compress_dim: int = 64
    ):
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction horizon
            enc_in: Number of input features (for reference, CI processes each separately)
            d_model: Model dimension
            n_layers: Number of Mamba layers
            patch_len: Patch length
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            dropout: Dropout rate
            compress_dim: Dimension for compressed prediction head
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.d_model = d_model
        
        # Calculate number of patches
        self.n_patches = seq_len // patch_len
        
        # Instance normalization (per channel)
        self.norm = nn.BatchNorm1d(1)
        
        # Patch embedding (shared across channels in CI mode)
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            d_model=d_model,
            in_channels=1  # CI mode: 1 channel at a time
        )
        
        # Forward and backward Mamba encoders
        self.forward_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        self.backward_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # Temporal alignment for backward encoder
        self.temporal_align = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )
        
        # Prediction head (separate for each channel in CI mode)
        self.prediction_heads = nn.ModuleList([
            PredictionHead(
                d_model=d_model,
                n_patches=self.n_patches,
                pred_len=pred_len,
                compress_dim=compress_dim
            )
            for _ in range(enc_in)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - only processes target channel (OT).

        Args:
            x: (batch, seq_len, enc_in) - multivariate time series

        Returns:
            Forecast: (batch, pred_len) - forecast for target variable (last channel)
        """
        # Only process target channel (last one = OT)
        x_channel = x[:, :, -1:]  # (batch, seq_len, 1)

        # Normalize
        x_channel = x_channel.transpose(1, 2)  # (batch, 1, seq_len)
        x_channel = self.norm(x_channel)
        x_channel = x_channel.transpose(1, 2)  # (batch, seq_len, 1)

        # Patch embedding
        patches = self.patch_embed(x_channel)  # (batch, n_patches, d_model)

        # Forward encoder
        forward_out = self.forward_encoder(patches, reverse=False)

        # Backward encoder
        backward_out = self.backward_encoder(patches, reverse=True)

        # Align backward output
        backward_out = backward_out.transpose(1, 2)  # (batch, d_model, n_patches)
        backward_out = self.temporal_align(backward_out)
        backward_out = backward_out.transpose(1, 2)  # (batch, n_patches, d_model)

        # Combine forward and backward
        combined = forward_out + backward_out  # (batch, n_patches, d_model)

        # Predict using the last prediction head
        return self.prediction_heads[-1](combined)


def create_mamba_forecaster(
    seq_len: int,
    pred_len: int,
    enc_in: int = 7,
    **kwargs
) -> MambaForecaster:
    """
    Factory function to create Mamba forecaster with default settings.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        enc_in: Number of input features
        **kwargs: Additional model arguments
        
    Returns:
        MambaForecaster instance
    """
    return MambaForecaster(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        **kwargs
    )

