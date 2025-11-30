"""
Mamba-based Time Series Forecasting Model.

Architecture:
- Multivariate input (all channels)
- Forward-only (causal) Mamba encoder
- Patch embedding for sequence compression
- Simple prediction head
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
    """Embed multivariate time series into patches using 1D convolution."""

    def __init__(self, patch_len: int = 16, d_model: int = 128, in_channels: int = 7):
        """
        Args:
            patch_len: Length of each patch
            d_model: Model dimension (embedding size)
            in_channels: Number of input channels
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
    """Stack of Mamba blocks with normalization (forward-only, causal)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)

        Returns:
            Output: (batch, n_patches, d_model)
        """
        for layer in self.layers:
            # Residual connection with LayerNorm (pre-norm)
            residual = x
            x = layer['norm'](x)
            x = layer['mamba'](x)
            x = layer['dropout'](x)
            x = x + residual

        return x


class MambaForecaster(nn.Module):
    """
    Mamba-based time series forecasting model.

    Architecture:
    1. LayerNorm on input
    2. Patch embedding (1D conv) - processes all channels together
    3. Forward-only Mamba encoder (causal)
    4. Prediction head

    Processes all input channels together (multivariate).
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
            enc_in: Number of input features
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

        # Input normalization (LayerNorm across features)
        self.input_norm = nn.LayerNorm(enc_in)

        # Patch embedding (processes all channels together)
        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            d_model=d_model,
            in_channels=enc_in
        )

        # Forward-only Mamba encoder (causal)
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model * self.n_patches, compress_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(compress_dim * 2, pred_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - processes all input channels.

        Args:
            x: (batch, seq_len, enc_in) - multivariate time series

        Returns:
            Forecast: (batch, pred_len) - forecast for target variable
        """
        batch_size = x.shape[0]

        # Normalize input
        x = self.input_norm(x)  # (batch, seq_len, enc_in)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, d_model)

        # Mamba encoder (forward-only, causal)
        x = self.encoder(x)  # (batch, n_patches, d_model)

        # Final norm
        x = self.final_norm(x)  # (batch, n_patches, d_model)

        # Flatten and predict
        x = x.reshape(batch_size, -1)  # (batch, n_patches * d_model)
        x = self.head(x)  # (batch, pred_len)

        return x


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
