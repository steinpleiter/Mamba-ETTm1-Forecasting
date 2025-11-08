"""
Evaluation metrics for time series forecasting.

Implements:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MASE (Mean Absolute Scaled Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
"""

import numpy as np
import torch
from typing import Union, Dict, Optional


def mae(predictions: Union[np.ndarray, torch.Tensor], 
        targets: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Absolute Error.
    
    MAE = mean(|y_pred - y_true|)
    
    Args:
        predictions: Predicted values, shape (n_samples, horizon) or (n_samples,)
        targets: True values, same shape as predictions
        
    Returns:
        MAE as float
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    return float(np.mean(np.abs(predictions - targets)))


def mse(predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Squared Error.
    
    MSE = mean((y_pred - y_true)^2)
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        MSE as float
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    return float(np.mean((predictions - targets) ** 2))


def rmse(predictions: Union[np.ndarray, torch.Tensor],
         targets: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Root Mean Squared Error.
    
    RMSE = sqrt(MSE)
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        RMSE as float
    """
    return float(np.sqrt(mse(predictions, targets)))


def mase(predictions: Union[np.ndarray, torch.Tensor],
         targets: Union[np.ndarray, torch.Tensor],
         train_data: Union[np.ndarray, torch.Tensor],
         seasonal_period: int = 96) -> float:
    """
    Mean Absolute Scaled Error.
    
    Compares prediction error to naive seasonal baseline.
    MASE < 1 means better than seasonal naive.
    
    MASE = MAE(predictions) / MAE(seasonal_naive_on_train)
    
    For ETTm1 (15-min data), seasonal_period=96 represents daily seasonality.
    
    Args:
        predictions: Predicted values, shape (n_samples, horizon)
        targets: True values, same shape
        train_data: Training data for computing naive baseline, shape (n_train,)
        seasonal_period: Seasonal period for naive forecast (default: 96 for daily)
        
    Returns:
        MASE as float
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.detach().cpu().numpy()
    
    # Flatten if needed
    predictions_flat = predictions.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    # MAE of predictions
    prediction_mae = np.mean(np.abs(predictions_flat - targets_flat))
    
    # MAE of naive seasonal forecast on training data
    # Naive forecast: y_t = y_{t-m} where m is seasonal period
    if len(train_data) <= seasonal_period:
        raise ValueError(f"Training data length ({len(train_data)}) must be > seasonal_period ({seasonal_period})")
    
    naive_errors = np.abs(train_data[seasonal_period:] - train_data[:-seasonal_period])
    naive_mae = np.mean(naive_errors)
    
    if naive_mae == 0:
        # Avoid division by zero (constant training data)
        return np.inf if prediction_mae > 0 else 0.0
    
    return float(prediction_mae / naive_mae)


def smape(predictions: Union[np.ndarray, torch.Tensor],
          targets: Union[np.ndarray, torch.Tensor],
          epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    sMAPE = 100 * mean(|y_pred - y_true| / (|y_pred| + |y_true|))
    
    Range: [0, 100], where 0 is perfect prediction.
    Symmetric because it treats over- and under-predictions equally.
    
    Args:
        predictions: Predicted values
        targets: True values
        epsilon: Small value to avoid division by zero
        
    Returns:
        sMAPE as float (percentage, 0-100)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    numerator = np.abs(predictions - targets)
    denominator = (np.abs(predictions) + np.abs(targets)) / 2.0 + epsilon
    
    return float(100.0 * np.mean(numerator / denominator))


def compute_all_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    train_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
    seasonal_period: int = 96
) -> Dict[str, float]:
    """
    Compute all forecasting metrics at once.
    
    Args:
        predictions: Predicted values
        targets: True values
        train_data: Training data (required for MASE)
        seasonal_period: Seasonal period for MASE
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mae': mae(predictions, targets),
        'mse': mse(predictions, targets),
        'rmse': rmse(predictions, targets),
        'smape': smape(predictions, targets)
    }
    
    if train_data is not None:
        try:
            metrics['mase'] = mase(predictions, targets, train_data, seasonal_period)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not compute MASE: {e}")
            metrics['mase'] = np.nan
    
    return metrics


class MetricsTracker:
    """Track metrics during training/evaluation."""
    
    def __init__(self, track_mase: bool = False, 
                 train_data: Optional[np.ndarray] = None,
                 seasonal_period: int = 96):
        """
        Args:
            track_mase: Whether to compute MASE (requires train_data)
            train_data: Training data for MASE computation
            seasonal_period: Seasonal period for MASE
        """
        self.track_mase = track_mase
        self.train_data = train_data
        self.seasonal_period = seasonal_period
        
        self.predictions = []
        self.targets = []
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: Union[np.ndarray, torch.Tensor],
               targets: Union[np.ndarray, torch.Tensor]):
        """
        Add a batch of predictions and targets.
        
        Args:
            predictions: Batch predictions
            targets: Batch targets
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        self.predictions.append(predictions)
        self.targets.append(targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute metrics over all accumulated predictions.
        
        Returns:
            Dictionary with all metrics
        """
        if len(self.predictions) == 0:
            return {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'smape': np.nan,
                'mase': np.nan
            }
        
        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        train_data = self.train_data if self.track_mase else None
        
        return compute_all_metrics(
            all_preds,
            all_targets,
            train_data=train_data,
            seasonal_period=self.seasonal_period
        )
    
    def compute_and_reset(self) -> Dict[str, float]:
        """Compute metrics and reset tracker."""
        metrics = self.compute()
        self.reset()
        return metrics


def seasonal_naive_forecast(
    historical_data: Union[np.ndarray, torch.Tensor],
    forecast_horizon: int,
    seasonal_period: int = 96
) -> np.ndarray:
    """
    Generate seasonal naive forecast.
    
    Forecast: y_{t+h} = y_{t+h-m} where m is seasonal period
    
    Args:
        historical_data: Historical time series, shape (context_length,)
        forecast_horizon: Number of steps to forecast (H)
        seasonal_period: Seasonal period (default: 96 for daily in 15-min data)
        
    Returns:
        Forecast of shape (forecast_horizon,)
    """
    if isinstance(historical_data, torch.Tensor):
        historical_data = historical_data.detach().cpu().numpy()
    
    if len(historical_data) < seasonal_period:
        raise ValueError(f"Historical data length ({len(historical_data)}) < seasonal_period ({seasonal_period})")
    
    # Take the last seasonal_period values and repeat/tile as needed
    forecast = []
    for h in range(forecast_horizon):
        # Index from the end of historical data
        idx = -(seasonal_period - (h % seasonal_period))
        forecast.append(historical_data[idx])
    
    return np.array(forecast)


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix (e.g., "Train", "Val", "Test")
    """
    if prefix:
        print(f"\n{prefix} Metrics:")
    else:
        print("\nMetrics:")
    
    print("-" * 40)
    for name, value in metrics.items():
        if np.isnan(value):
            print(f"  {name.upper():10s}: N/A")
        else:
            print(f"  {name.upper():10s}: {value:.6f}")
    print("-" * 40)


def compare_metrics(baseline_metrics: Dict[str, float],
                   model_metrics: Dict[str, float],
                   metric_name: str = "MAE") -> float:
    """
    Compare model to baseline and compute improvement.
    
    Args:
        baseline_metrics: Baseline metrics dict
        model_metrics: Model metrics dict
        metric_name: Metric to compare (default: "MAE")
        
    Returns:
        Improvement percentage (positive = model is better)
    """
    baseline_val = baseline_metrics.get(metric_name.lower())
    model_val = model_metrics.get(metric_name.lower())
    
    if baseline_val is None or model_val is None:
        return np.nan
    
    if baseline_val == 0:
        return np.nan
    
    improvement = ((baseline_val - model_val) / baseline_val) * 100
    return improvement

