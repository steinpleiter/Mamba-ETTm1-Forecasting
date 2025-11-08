"""
Train DLinear baseline model on ETTm1 dataset.

Usage:
    rye run python scripts/train_dlinear.py
    rye run python scripts/train_dlinear.py --individual --epochs 50
"""

import argparse
import pickle
from pathlib import Path
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ETTDataset
from src.models.baseline import DLinear
from src.training.losses import get_loss_function
from src.evaluation.metrics import MetricsTracker, print_metrics


def load_processed_data(data_path: Path):
    """Load processed data from pickle file."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        use_amp: Use automatic mixed precision
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                loss = criterion(predictions, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device, tracker=None):
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        tracker: Optional MetricsTracker
        
    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    if tracker is not None:
        tracker.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            n_batches += 1
            
            if tracker is not None:
                tracker.update(predictions, targets)
    
    avg_loss = total_loss / n_batches
    metrics = tracker.compute() if tracker is not None else {}
    
    return avg_loss, metrics


def main(args):
    print("=" * 80)
    print("Training DLinear Model")
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
    
    print(f"  Train: {len(train_data['input'])} samples")
    print(f"  Val:   {len(val_data['input'])} samples")
    
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multi-worker dataloading well
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    print(f"\nCreating DLinear model")
    enc_in = 7 + (8 if use_calendar else 0)
    
    model = DLinear(
        seq_len=context_length,
        pred_len=forecast_horizon,
        individual=args.individual,
        enc_in=enc_in,
        kernel_size=args.kernel_size
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Channel-independent mode: {args.individual}")
    print(f"  Kernel size: {args.kernel_size}")
    
    # Loss function
    criterion = get_loss_function(args.loss)
    print(f"  Loss function: {args.loss}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training tracker
    train_targets_flat = train_data['target'].reshape(-1)
    val_tracker = MetricsTracker(
        track_mase=True,
        train_data=train_targets_flat,
        seasonal_period=96
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs")
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_mse': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=(device.type == 'cuda')
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, tracker=val_tracker
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val MAE:    {val_metrics['mae']:.6f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6e}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_mse'].append(val_metrics['mse'])
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_dir = Path('results/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f'dlinear_{dataset_name}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_metrics['mae'],
                'config': vars(args)
            }, checkpoint_path)
            
            print(f"Saved best model (MAE: {best_val_mae:.6f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    print(f"\nLoading best model")
    checkpoint = torch.load(checkpoint_dir / f'dlinear_{dataset_name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Best validation MAE: {checkpoint['val_mae']:.6f}")
    
    # Final validation
    print(f"\nFinal evaluation")
    _, val_metrics = validate(model, val_loader, criterion, device, tracker=val_tracker)
    print_metrics(val_metrics, prefix="Validation")
    
    # Save results
    print(f"\nSaving results")
    results = {
        'model': 'dlinear',
        'config': vars(args),
        'val_metrics': val_metrics,
        'history': history,
        'n_parameters': n_params
    }
    
    results_path = Path('results') / f'dlinear_{dataset_name}_training_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"  Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nBest validation MAE: {best_val_mae:.6f}")
    print(f"Improvement over seasonal naive: {((1.732 - best_val_mae) / 1.732 * 100):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DLinear model")
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to configuration file'
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
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Use channel-independent mode'
    )
    
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=25,
        help='Kernel size for moving average decomposition'
    )
    
    parser.add_argument(
        '--loss',
        type=str,
        default='smooth_l1',
        choices=['smooth_l1', 'mse', 'mae'],
        help='Loss function'
    )
    
    args = parser.parse_args()
    main(args)

