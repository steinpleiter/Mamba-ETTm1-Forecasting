"""
LSTM-based time series forecasting model.

Simple but effective baseline using stacked LSTMs.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMForecaster(nn.Module):
    """
    LSTM-based forecasting model.
    
    Architecture:
    - Input projection
    - Stacked LSTM layers
    - Fully connected prediction head
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        pred_len: int = 96,
        enc_in: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction horizon
            enc_in: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, enc_in)
            
        Returns:
            Predictions of shape (batch, pred_len)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Predict
        output = self.fc(last_hidden)  # (batch, pred_len)
        
        return output


def create_lstm_forecaster(
    seq_len: int,
    pred_len: int,
    enc_in: int = 7,
    **kwargs
) -> LSTMForecaster:
    """Factory function to create LSTM forecaster."""
    return LSTMForecaster(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        **kwargs
    )

