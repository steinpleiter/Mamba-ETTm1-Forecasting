"""
Unit tests for evaluation metrics.
"""

import pytest
import numpy as np
import torch
from src.evaluation.metrics import (
    mae, mse, rmse, mase, smape,
    compute_all_metrics, MetricsTracker,
    seasonal_naive_forecast
)


class TestBasicMetrics:
    """Test basic metrics (MAE, MSE, RMSE)."""
    
    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert mae(preds, targets) == 0.0
    
    def test_mae_known_values(self):
        """Test MAE with known values."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        # MAE = mean([0.5, 0.5, 0.5]) = 0.5
        assert np.isclose(mae(preds, targets), 0.5)
    
    def test_mse_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        assert mse(preds, targets) == 0.0
    
    def test_mse_known_values(self):
        """Test MSE with known values."""
        preds = np.array([1.0, 2.0])
        targets = np.array([2.0, 4.0])
        # MSE = mean([1.0, 4.0]) = 2.5
        assert np.isclose(mse(preds, targets), 2.5)
    
    def test_rmse_known_values(self):
        """Test RMSE with known values."""
        preds = np.array([1.0, 2.0])
        targets = np.array([2.0, 4.0])
        # RMSE = sqrt(2.5) ≈ 1.58
        assert np.isclose(rmse(preds, targets), np.sqrt(2.5))
    
    def test_torch_tensor_input(self):
        """Test that metrics work with PyTorch tensors."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])
        assert np.isclose(mae(preds, targets), 0.5)


class TestMASE:
    """Test Mean Absolute Scaled Error."""
    
    def test_mase_better_than_naive(self):
        """Test MASE < 1 when model beats seasonal naive."""
        # Create seasonal data with period 4
        train = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        
        # Perfect predictions (MASE should be 0)
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = mase(preds, targets, train, seasonal_period=4)
        assert result == 0.0
    
    def test_mase_worse_than_naive(self):
        """Test MASE > 1 when model is worse than naive."""
        # Constant training data
        train = np.array([5.0] * 10)
        
        # Bad predictions
        preds = np.array([10.0, 10.0])
        targets = np.array([5.0, 5.0])
        
        result = mase(preds, targets, train, seasonal_period=2)
        assert result > 1.0 or np.isinf(result)
    
    def test_mase_multidimensional(self):
        """Test MASE with 2D predictions."""
        train = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        
        # Multiple samples
        preds = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        targets = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = mase(preds, targets, train, seasonal_period=4)
        assert result == 0.0


class TestSMAPE:
    """Test Symmetric Mean Absolute Percentage Error."""
    
    def test_smape_perfect_prediction(self):
        """Test sMAPE with perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        assert smape(preds, targets) == 0.0
    
    def test_smape_known_values(self):
        """Test sMAPE with known values."""
        preds = np.array([100.0])
        targets = np.array([150.0])
        # |100-150| / ((100+150)/2) = 50/125 = 0.4
        # sMAPE = 40%
        expected = 100.0 * 0.4
        assert np.isclose(smape(preds, targets), expected)
    
    def test_smape_symmetric(self):
        """Test that sMAPE is symmetric."""
        preds1 = np.array([100.0])
        targets1 = np.array([150.0])
        
        preds2 = np.array([150.0])
        targets2 = np.array([100.0])
        
        assert np.isclose(smape(preds1, targets1), smape(preds2, targets2))


class TestSeasonalNaive:
    """Test seasonal naive forecast."""
    
    def test_seasonal_naive_simple(self):
        """Test seasonal naive with simple pattern."""
        # Pattern: [1, 2, 3, 4] repeating
        historical = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        
        forecast = seasonal_naive_forecast(historical, forecast_horizon=4, seasonal_period=4)
        
        # Should forecast [1, 2, 3, 4]
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(forecast, expected)
    
    def test_seasonal_naive_longer_horizon(self):
        """Test seasonal naive with horizon > period."""
        historical = np.array([1, 2, 1, 2, 1, 2])
        
        forecast = seasonal_naive_forecast(historical, forecast_horizon=6, seasonal_period=2)
        
        # Should repeat [1, 2] pattern
        expected = np.array([1, 2, 1, 2, 1, 2])
        np.testing.assert_array_equal(forecast, expected)


class TestMetricsTracker:
    """Test MetricsTracker class."""
    
    def test_tracker_single_batch(self):
        """Test tracker with single batch."""
        tracker = MetricsTracker()
        
        preds = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        tracker.update(preds, targets)
        metrics = tracker.compute()
        
        assert metrics['mae'] == 0.0
        assert metrics['mse'] == 0.0
    
    def test_tracker_multiple_batches(self):
        """Test tracker accumulating multiple batches."""
        tracker = MetricsTracker()
        
        # Batch 1
        preds1 = np.array([[1.0], [2.0]])
        targets1 = np.array([[1.5], [2.5]])
        tracker.update(preds1, targets1)
        
        # Batch 2
        preds2 = np.array([[3.0], [4.0]])
        targets2 = np.array([[3.5], [4.5]])
        tracker.update(preds2, targets2)
        
        metrics = tracker.compute()
        
        # All errors are 0.5, so MAE should be 0.5
        assert np.isclose(metrics['mae'], 0.5)
    
    def test_tracker_reset(self):
        """Test tracker reset."""
        tracker = MetricsTracker()
        
        preds = np.array([[1.0, 2.0]])
        targets = np.array([[2.0, 3.0]])
        
        tracker.update(preds, targets)
        tracker.reset()
        
        # After reset, should have no data
        metrics = tracker.compute()
        assert np.isnan(metrics['mae'])


class TestComputeAllMetrics:
    """Test compute_all_metrics function."""
    
    def test_all_metrics_without_mase(self):
        """Test computing all metrics without MASE."""
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        
        metrics = compute_all_metrics(preds, targets)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'smape' in metrics
        assert 'mase' not in metrics
    
    def test_all_metrics_with_mase(self):
        """Test computing all metrics including MASE."""
        train = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
        preds = np.array([1.0, 2.0])
        targets = np.array([1.5, 2.5])
        
        metrics = compute_all_metrics(preds, targets, train_data=train, seasonal_period=4)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'smape' in metrics
        assert 'mase' in metrics
        assert not np.isnan(metrics['mase'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

