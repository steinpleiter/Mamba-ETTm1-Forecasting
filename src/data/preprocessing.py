"""
Data preprocessing utilities for ETTm1 dataset.

Handles:
- Timestamp parsing and validation
- Normalization (z-score)
- Sliding window creation
- Calendar feature engineering
- Missing value handling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from datetime import datetime


def load_ett_data(file_path: str) -> pd.DataFrame:
    """
    Load ETTm1 dataset from CSV.
    
    Args:
        file_path: Path to ETTm1.csv
        
    Returns:
        DataFrame with parsed datetime index
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def validate_timestamps(df: pd.DataFrame, expected_freq: str = '15min') -> pd.DataFrame:
    """
    Validate that timestamps are evenly spaced at expected frequency.
    
    Args:
        df: DataFrame with datetime index
        expected_freq: Expected frequency (default: '15min' for ETTm1)
        
    Returns:
        DataFrame with validated timestamps
    """
    # Check for gaps
    time_diffs = df.index.to_series().diff()
    expected_diff = pd.Timedelta(expected_freq)
    
    gaps = time_diffs[time_diffs != expected_diff].dropna()
    
    if len(gaps) > 0:
        print(f"Warning: Found {len(gaps)} timestamp gaps")
        print(f"First few gaps:\n{gaps.head()}")
        
        # Option 1: Reindex to fill gaps with NaN
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=expected_freq
        )
        df = df.reindex(full_index)
        print(f"Reindexed to {len(df)} rows with regular {expected_freq} spacing")
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'forward_fill',
    max_gap: int = 4
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with potential missing values
        method: 'forward_fill', 'interpolate', or 'drop'
        max_gap: Maximum gap size to fill (for forward_fill)
        
    Returns:
        DataFrame with missing values handled
    """
    n_missing_before = df.isna().sum().sum()
    
    if n_missing_before == 0:
        print("No missing values found")
        return df
    
    print(f"Missing values before: {n_missing_before}")
    
    if method == 'forward_fill':
        df = df.ffill(limit=max_gap)
        # Fill any remaining NaNs at the start with backfill
        df = df.bfill()
    elif method == 'interpolate':
        df = df.interpolate(method='linear', limit=max_gap)
        df = df.bfill()
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    n_missing_after = df.isna().sum().sum()
    print(f"Missing values after: {n_missing_after}")
    
    return df


def compute_normalization_stats(df: pd.DataFrame, features: list) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std for z-score normalization.
    
    Args:
        df: Training data DataFrame
        features: List of feature column names
        
    Returns:
        Dictionary with 'mean' and 'std' for each feature
    """
    stats = {}
    for feat in features:
        stats[feat] = {
            'mean': df[feat].mean(),
            'std': df[feat].std()
        }
    return stats


def normalize_data(
    df: pd.DataFrame,
    features: list,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    fit: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Apply z-score normalization to features.
    
    Args:
        df: DataFrame to normalize
        features: List of feature columns
        stats: Pre-computed normalization stats (if fit=False)
        fit: Whether to compute stats from this data
        
    Returns:
        Tuple of (normalized DataFrame, normalization stats)
    """
    df_norm = df.copy()
    
    if fit:
        stats = compute_normalization_stats(df, features)
    elif stats is None:
        raise ValueError("Must provide stats if fit=False")
    
    for feat in features:
        mean = stats[feat]['mean']
        std = stats[feat]['std']
        df_norm[feat] = (df[feat] - mean) / std
    
    return df_norm, stats


def denormalize_data(
    data: np.ndarray,
    feature: str,
    stats: Dict[str, Dict[str, float]]
) -> np.ndarray:
    """
    Reverse z-score normalization.
    
    Args:
        data: Normalized data array
        feature: Feature name
        stats: Normalization statistics
        
    Returns:
        Denormalized data
    """
    mean = stats[feature]['mean']
    std = stats[feature]['std']
    return data * std + mean


def create_calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Create sin/cos encoded calendar features.
    
    Features:
    - Minute of hour (0-59)
    - Hour of day (0-23)
    - Day of week (0-6)
    - Month of year (1-12)
    
    Args:
        timestamps: DatetimeIndex
        
    Returns:
        Array of shape (len(timestamps), 8) with sin/cos features
    """
    features = []
    
    # Minute of hour (0-59)
    minute = timestamps.minute
    features.append(np.sin(2 * np.pi * minute / 60))
    features.append(np.cos(2 * np.pi * minute / 60))
    
    # Hour of day (0-23)
    hour = timestamps.hour
    features.append(np.sin(2 * np.pi * hour / 24))
    features.append(np.cos(2 * np.pi * hour / 24))
    
    # Day of week (0-6)
    dayofweek = timestamps.dayofweek
    features.append(np.sin(2 * np.pi * dayofweek / 7))
    features.append(np.cos(2 * np.pi * dayofweek / 7))
    
    # Month of year (1-12)
    month = timestamps.month
    features.append(np.sin(2 * np.pi * (month - 1) / 12))
    features.append(np.cos(2 * np.pi * (month - 1) / 12))
    
    return np.stack(features, axis=1)


def create_sliding_windows(
    df: pd.DataFrame,
    context_length: int,
    forecast_horizon: int,
    target_feature: str,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series forecasting.
    
    Args:
        df: Normalized DataFrame
        context_length: Length of input window (L)
        forecast_horizon: Length of forecast window (H)
        target_feature: Name of target column (e.g., 'OT')
        stride: Stride for sliding window
        
    Returns:
        Tuple of (input_data, target_data, input_calendar, future_calendar)
        - input_data: (n_windows, L, n_features)
        - target_data: (n_windows, H)
        - input_calendar: (n_windows, L, 8)
        - future_calendar: (n_windows, H, 8)
    """
    feature_cols = df.columns.tolist()
    target_idx = feature_cols.index(target_feature)
    
    data_array = df.values
    timestamps = df.index
    
    # Calculate number of windows
    window_size = context_length + forecast_horizon
    n_windows = (len(df) - window_size) // stride + 1
    
    input_data = []
    target_data = []
    input_calendar = []
    future_calendar = []
    
    for i in range(0, len(df) - window_size + 1, stride):
        # Input window
        input_window = data_array[i:i + context_length]
        input_data.append(input_window)
        
        # Target window (only target feature)
        target_window = data_array[i + context_length:i + window_size, target_idx]
        target_data.append(target_window)
        
        # Calendar features for input
        input_times = timestamps[i:i + context_length]
        input_cal = create_calendar_features(input_times)
        input_calendar.append(input_cal)
        
        # Calendar features for future (known at prediction time)
        future_times = timestamps[i + context_length:i + window_size]
        future_cal = create_calendar_features(future_times)
        future_calendar.append(future_cal)
    
    return (
        np.array(input_data),
        np.array(target_data),
        np.array(input_calendar),
        np.array(future_calendar)
    )


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets (time-based, no shuffle).
    
    Args:
        df: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Data split:")
    print(f"  Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"  Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    
    return train_df, val_df, test_df

