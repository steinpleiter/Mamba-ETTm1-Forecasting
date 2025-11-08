# Model architectures module

from .baseline import DLinear, SeasonalNaive, LinearRegression, LastValue
from .lstm_forecaster import LSTMForecaster

try:
    from .mamba_forecaster import MambaForecaster
except ImportError:
    # Mamba not available (e.g., on macOS)
    MambaForecaster = None

__all__ = [
    'DLinear',
    'SeasonalNaive', 
    'LinearRegression',
    'LastValue',
    'LSTMForecaster',
    'MambaForecaster'
]

