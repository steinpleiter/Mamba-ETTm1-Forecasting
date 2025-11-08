"""
Unit tests for baseline models.
"""

import pytest
import torch
from src.models.baseline import (
    SeasonalNaive,
    DLinear,
    LinearRegression,
    LastValue,
    create_baseline_model
)


class TestSeasonalNaive:
    """Test SeasonalNaive model."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        model = SeasonalNaive(seasonal_period=4, forecast_horizon=3)
        
        # Input: (batch=2, seq_len=10, features=7)
        x = torch.randn(2, 10, 7)
        output = model(x)
        
        # Output should be (batch=2, forecast_horizon=3)
        assert output.shape == (2, 3)
    
    def test_seasonal_pattern(self):
        """Test that it correctly copies seasonal pattern."""
        model = SeasonalNaive(seasonal_period=4, forecast_horizon=4)
        
        # Create input with clear pattern: [1, 2, 3, 4, 1, 2, 3, 4]
        pattern = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x = pattern.repeat(2).unsqueeze(0).unsqueeze(-1)  # (1, 8, 1)
        
        output = model(x)
        
        # Should forecast [1, 2, 3, 4]
        expected = pattern.unsqueeze(0)
        assert torch.allclose(output, expected)


class TestDLinear:
    """Test DLinear model."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        model = DLinear(seq_len=10, pred_len=5, enc_in=7, individual=False)
        
        x = torch.randn(2, 10, 7)
        output = model(x)
        
        # Output: (batch=2, pred_len=5)
        assert output.shape == (2, 5)
    
    def test_individual_mode(self):
        """Test channel-independent mode."""
        model = DLinear(seq_len=10, pred_len=5, enc_in=7, individual=True)
        
        x = torch.randn(2, 10, 7)
        output = model(x)
        
        assert output.shape == (2, 5)
    
    def test_parameters_exist(self):
        """Test that model has learnable parameters."""
        model = DLinear(seq_len=10, pred_len=5, enc_in=7)
        
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


class TestLinearRegression:
    """Test LinearRegression baseline."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        model = LinearRegression(seq_len=10, pred_len=5, enc_in=7)
        
        x = torch.randn(2, 10, 7)
        output = model(x)
        
        assert output.shape == (2, 5)
    
    def test_parameters_exist(self):
        """Test model has parameters."""
        model = LinearRegression(seq_len=10, pred_len=5, enc_in=7)
        
        n_params = sum(p.numel() for p in model.parameters())
        # Should have seq_len * enc_in * pred_len + pred_len params
        expected = 10 * 7 * 5 + 5
        assert n_params == expected


class TestLastValue:
    """Test LastValue baseline."""
    
    def test_output_shape(self):
        """Test output shape."""
        model = LastValue(forecast_horizon=5)
        
        x = torch.randn(2, 10, 7)
        output = model(x)
        
        assert output.shape == (2, 5)
    
    def test_repeats_last_value(self):
        """Test that it repeats last value."""
        model = LastValue(forecast_horizon=3)
        
        # Input where last OT value is 42.0
        x = torch.zeros(1, 5, 7)
        x[0, -1, -1] = 42.0
        
        output = model(x)
        
        # All outputs should be 42.0
        expected = torch.tensor([[42.0, 42.0, 42.0]])
        assert torch.allclose(output, expected)


class TestModelFactory:
    """Test create_baseline_model factory."""
    
    def test_creates_seasonal_naive(self):
        """Test creating SeasonalNaive."""
        model = create_baseline_model('seasonal_naive', seq_len=10, pred_len=5)
        assert isinstance(model, SeasonalNaive)
    
    def test_creates_dlinear(self):
        """Test creating DLinear."""
        model = create_baseline_model('dlinear', seq_len=10, pred_len=5, enc_in=7)
        assert isinstance(model, DLinear)
    
    def test_creates_linear(self):
        """Test creating LinearRegression."""
        model = create_baseline_model('linear', seq_len=10, pred_len=5, enc_in=7)
        assert isinstance(model, LinearRegression)
    
    def test_creates_last_value(self):
        """Test creating LastValue."""
        model = create_baseline_model('last_value', seq_len=10, pred_len=5)
        assert isinstance(model, LastValue)
    
    def test_unknown_model_raises_error(self):
        """Test that unknown model name raises error."""
        with pytest.raises(ValueError, match="Unknown baseline model"):
            create_baseline_model('unknown_model', seq_len=10, pred_len=5)


class TestGradientFlow:
    """Test that trainable models have gradient flow."""
    
    def test_dlinear_gradients(self):
        """Test DLinear has gradients."""
        model = DLinear(seq_len=10, pred_len=5, enc_in=7)
        
        x = torch.randn(2, 10, 7)
        targets = torch.randn(2, 5)
        
        output = model(x)
        loss = ((output - targets) ** 2).mean()
        loss.backward()
        
        # Check that parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
    
    def test_linear_gradients(self):
        """Test LinearRegression has gradients."""
        model = LinearRegression(seq_len=10, pred_len=5, enc_in=7)
        
        x = torch.randn(2, 10, 7)
        targets = torch.randn(2, 5)
        
        output = model(x)
        loss = ((output - targets) ** 2).mean()
        loss.backward()
        
        for param in model.parameters():
            assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

