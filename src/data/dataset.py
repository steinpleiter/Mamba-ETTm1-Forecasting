"""
PyTorch Dataset classes for ETTm1 time series forecasting.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional


class ETTDataset(Dataset):
    """
    PyTorch Dataset for ETTm1 time series forecasting.
    
    Returns:
        - input_data: (L, n_features) - Historical data
        - target_data: (H,) - Future OT values to predict
        - input_calendar: (L, 8) - Calendar features for historical window
        - future_calendar: (H, 8) - Calendar features for forecast window (known)
    """
    
    def __init__(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        input_calendar: np.ndarray,
        future_calendar: np.ndarray,
        use_calendar: bool = True
    ):
        """
        Args:
            input_data: (n_samples, L, n_features)
            target_data: (n_samples, H)
            input_calendar: (n_samples, L, 8)
            future_calendar: (n_samples, H, 8)
            use_calendar: Whether to include calendar features
        """
        self.input_data = torch.FloatTensor(input_data)
        self.target_data = torch.FloatTensor(target_data)
        self.input_calendar = torch.FloatTensor(input_calendar)
        self.future_calendar = torch.FloatTensor(future_calendar)
        self.use_calendar = use_calendar
        
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dictionary with:
            - 'input': (L, n_features) or (L, n_features + 8) with calendar
            - 'target': (H,)
            - 'future_calendar': (H, 8) if use_calendar else None
        """
        input_seq = self.input_data[idx]  # (L, n_features)
        target_seq = self.target_data[idx]  # (H,)
        
        if self.use_calendar:
            # Concatenate calendar features to input
            input_cal = self.input_calendar[idx]  # (L, 8)
            input_seq = torch.cat([input_seq, input_cal], dim=-1)  # (L, n_features + 8)
            future_cal = self.future_calendar[idx]  # (H, 8)
        else:
            future_cal = None
        
        return {
            'input': input_seq,
            'target': target_seq,
            'future_calendar': future_cal
        }


class ChannelIndependentDataset(Dataset):
    """
    Dataset that treats each channel as independent univariate series.
    Used for channel-independent (CI) modeling.
    
    Effectively multiplies number of samples by number of channels.
    """
    
    def __init__(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        input_calendar: np.ndarray,
        future_calendar: np.ndarray,
        target_channel_idx: int = 6,  # OT is index 6 in ETTm1
        use_calendar: bool = True
    ):
        """
        Args:
            input_data: (n_samples, L, n_features)
            target_data: (n_samples, H)
            input_calendar: (n_samples, L, 8)
            future_calendar: (n_samples, H, 8)
            target_channel_idx: Index of target channel (OT)
            use_calendar: Whether to use calendar features
        """
        self.input_data = torch.FloatTensor(input_data)
        self.target_data = torch.FloatTensor(target_data)
        self.input_calendar = torch.FloatTensor(input_calendar)
        self.future_calendar = torch.FloatTensor(future_calendar)
        self.target_channel_idx = target_channel_idx
        self.use_calendar = use_calendar
        
        self.n_samples, self.L, self.n_features = input_data.shape
        
    def __len__(self) -> int:
        # Each channel of each sample becomes a separate item
        return self.n_samples * self.n_features
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dictionary with:
            - 'input': (L,) or (L + 8,) with calendar - univariate
            - 'target': (H,)
            - 'future_calendar': (H, 8) if use_calendar else None
        """
        sample_idx = idx // self.n_features
        channel_idx = idx % self.n_features
        
        # Get univariate input for this channel
        input_seq = self.input_data[sample_idx, :, channel_idx]  # (L,)
        
        # Target is always OT (same for all channels of this sample)
        target_seq = self.target_data[sample_idx]  # (H,)
        
        if self.use_calendar:
            input_cal = self.input_calendar[sample_idx]  # (L, 8)
            # Flatten calendar features and concatenate
            input_cal_flat = input_cal.reshape(-1)  # (L * 8,)
            # For CI, we typically add calendar as additional context
            # Here we'll add it channel-wise
            input_seq = input_seq.unsqueeze(-1)  # (L, 1)
            input_seq = torch.cat([input_seq, input_cal], dim=-1)  # (L, 9)
            future_cal = self.future_calendar[sample_idx]  # (H, 8)
        else:
            input_seq = input_seq.unsqueeze(-1)  # (L, 1)
            future_cal = None
        
        return {
            'input': input_seq,
            'target': target_seq,
            'future_calendar': future_cal,
            'channel_idx': channel_idx
        }

