"""
Evaluate baseline models on ETTm1 dataset.

Usage:
    rye run python scripts/evaluate_baseline.py --model seasonal_naive
    rye run python scripts/evaluate_baseline.py --model dlinear
"""

import argparse
import pickle
from pathlib import Path
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ETTDataset
from src.models.baseline import create_baseline_model
from src.evaluation.metrics import MetricsTracker, print_metrics


def load_processed_data(data_path: Path):
    """Load processed data from pickle file."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def evaluate_model(model, dataloader, device, use_mase=False, train_data=None, seasonal_period=96):
    """
    Evaluate model on a dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device to run on
        use_mase: Whether to compute MASE
        train_data: Training data for MASE (target variable only)
        seasonal_period: Seasonal period for MASE
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    tracker = MetricsTracker(track_mase=use_mase, train_data=train_data, seasonal_period=seasonal_period)
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            predictions = model(inputs)
            
            # Update metrics
            tracker.update(predictions, targets)
    
    return tracker.compute()


def main(args):
    print("=" * 80)
    print(f"Evaluating Baseline Model: {args.model}")
    print("=" * 80)
    
    # Load config
    print(f"\nLoading configuration")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Paths
    processed_data_path = Path(data_config['paths']['processed_data'])
    context_length = data_config['context_length']
    forecast_horizon = data_config['forecast_horizon']
    dataset_name = data_config.get('dataset', 'ETTm1')

    train_path = processed_data_path / f"{dataset_name}_train_L{context_length}_H{forecast_horizon}.pkl"
    val_path = processed_data_path / f"{dataset_name}_val_L{context_length}_H{forecast_horizon}.pkl"
    test_path = processed_data_path / f"{dataset_name}_test_L{context_length}_H{forecast_horizon}.pkl"
    
    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Using device: {device}")
    
    # Load data
    print(f"\nLoading processed data")
    train_data = load_processed_data(train_path)
    val_data = load_processed_data(val_path)
    test_data = load_processed_data(test_path)
    
    print(f"  Train: {len(train_data['input'])} samples")
    print(f"  Val:   {len(val_data['input'])} samples")
    print(f"  Test:  {len(test_data['input'])} samples")
    
    # Create datasets
    print(f"\nCreating datasets")
    use_calendar = data_config.get('use_calendar_features', True)
    
    train_dataset = ETTDataset(
        train_data['input'],
        train_data['target'],
        train_data['input_calendar'],
        train_data['future_calendar'],
        use_calendar=use_calendar
    )
    
    val_dataset = ETTDataset(
        val_data['input'],
        val_data['target'],
        val_data['input_calendar'],
        val_data['future_calendar'],
        use_calendar=use_calendar
    )
    
    test_dataset = ETTDataset(
        test_data['input'],
        test_data['target'],
        test_data['input_calendar'],
        test_data['future_calendar'],
        use_calendar=use_calendar
    )
    
    # Create dataloaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\nCreating {args.model} model")
    
    # Input features: 7 original + 8 calendar if used
    enc_in = 7 + (8 if use_calendar else 0)
    
    model = create_baseline_model(
        model_name=args.model,
        seq_len=context_length,
        pred_len=forecast_horizon,
        enc_in=enc_in,
        seasonal_period=args.seasonal_period,
        individual=args.individual
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # For MASE computation, extract target variable from training data
    train_targets_flat = train_data['target'].reshape(-1)
    
    # Evaluate
    print(f"\nEvaluating model")
    
    print("\n  Validation set...")
    val_metrics = evaluate_model(
        model, val_loader, device,
        use_mase=True,
        train_data=train_targets_flat,
        seasonal_period=args.seasonal_period
    )
    
    print("\n  Test set...")
    test_metrics = evaluate_model(
        model, test_loader, device,
        use_mase=True,
        train_data=train_targets_flat,
        seasonal_period=args.seasonal_period
    )
    
    # Print results
    print(f"\nResults")
    print_metrics(val_metrics, prefix="Validation")
    print_metrics(test_metrics, prefix="Test")
    
    # Save results if requested
    if args.save_results:
        results = {
            'model': args.model,
            'config': {
                'context_length': context_length,
                'forecast_horizon': forecast_horizon,
                'seasonal_period': args.seasonal_period,
                'individual': args.individual
            },
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'n_parameters': n_params
        }
        
        results_path = Path('results') / f'{args.model}_baseline_results.pkl'
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['seasonal_naive', 'dlinear', 'linear', 'last_value'],
        help='Baseline model to evaluate'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='mps',  # Default to MPS for Mac users
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (mps for Mac M1/M2/M3)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--seasonal_period',
        type=int,
        default=96,
        help='Seasonal period for MASE (96 = daily for 15-min data)'
    )
    
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Use channel-independent mode for DLinear'
    )
    
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save results to pickle file'
    )
    
    args = parser.parse_args()
    main(args)

