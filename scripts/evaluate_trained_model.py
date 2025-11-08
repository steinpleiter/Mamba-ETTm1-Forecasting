"""
Evaluate a trained model checkpoint on test set.

Usage:
    rye run python scripts/evaluate_trained_model.py --checkpoint results/checkpoints/dlinear_best.pt
"""

import argparse
import pickle
from pathlib import Path
import sys
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ETTDataset
from src.models.baseline import DLinear
from src.models.mamba_forecaster import MambaForecaster
from src.evaluation.metrics import MetricsTracker, print_metrics


def load_processed_data(data_path: Path):
    """Load processed data from pickle file."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def evaluate(model, dataloader, device, tracker=None):
    """Evaluate model on dataloader."""
    model.eval()
    
    if tracker is not None:
        tracker.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(inputs)
            
            if tracker is not None:
                tracker.update(predictions, targets)
    
    return tracker.compute() if tracker is not None else {}


def main(args):
    print("=" * 80)
    print("Evaluating Trained Model")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    model_config = checkpoint['config']
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Validation MAE: {checkpoint['val_mae']:.6f}")
    
    # Load config
    config_path = model_config.get('config', 'configs/base_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Paths
    processed_data_path = Path(data_config['paths']['processed_data'])
    context_length = data_config['context_length']
    forecast_horizon = data_config['forecast_horizon']
    dataset_name = data_config.get('dataset', 'ETTm1')
    
    train_path = processed_data_path / f"{dataset_name}_train_L{context_length}_H{forecast_horizon}.pkl"
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
    test_data = load_processed_data(test_path)
    
    print(f"  Test: {len(test_data['input'])} samples")
    
    # Create datasets
    print(f"\nCreating test dataset")
    use_calendar = data_config.get('use_calendar_features', True)
    
    test_dataset = ETTDataset(
        test_data['input'],
        test_data['target'],
        test_data['input_calendar'],
        test_data['future_calendar'],
        use_calendar=use_calendar
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\nLoading model")
    enc_in = 7 + (8 if use_calendar else 0)
    
    # Determine model type from checkpoint
    checkpoint_name = Path(args.checkpoint).name
    
    if 'mamba' in checkpoint_name.lower():
        # Mamba model
        model = MambaForecaster(
            seq_len=context_length,
            pred_len=forecast_horizon,
            enc_in=enc_in,
            d_model=model_config.get('d_model', 128),
            n_layers=model_config.get('n_layers', 4),
            patch_len=model_config.get('patch_len', 16),
            d_state=model_config.get('d_state', 16),
            dropout=model_config.get('dropout', 0.1),
            compress_dim=model_config.get('compress_dim', 64)
        )
        print(f"  Model type: Mamba")
    else:
        # DLinear model
        model = DLinear(
            seq_len=context_length,
            pred_len=forecast_horizon,
            individual=model_config.get('individual', False),
            enc_in=enc_in,
            kernel_size=model_config.get('kernel_size', 25)
        )
        print(f"  Model type: DLinear")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set")
    
    # For MASE computation
    train_targets_flat = train_data['target'].reshape(-1)
    
    tracker = MetricsTracker(
        track_mase=True,
        train_data=train_targets_flat,
        seasonal_period=96
    )
    
    test_metrics = evaluate(model, test_loader, device, tracker=tracker)
    
    # Print results
    print_metrics(test_metrics, prefix="Test")
    
    # Compare to baseline
    print("\nComparison to Seasonal Naive:")
    print(f"  Seasonal Naive Test MAE: 1.123")
    print(f"  DLinear Test MAE:        {test_metrics['mae']:.3f}")
    improvement = ((1.123 - test_metrics['mae']) / 1.123) * 100
    print(f"  Improvement:             {improvement:.2f}%")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model checkpoint")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/checkpoints/dlinear_best.pt',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    main(args)

