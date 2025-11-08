"""
Data preprocessing script for ETTm1 dataset.

Usage:
    rye run preprocess
    or
    rye run python scripts/preprocess_data.py --config configs/base_config.yaml
"""

import argparse
import pickle
from pathlib import Path
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    load_ett_data,
    validate_timestamps,
    handle_missing_values,
    normalize_data,
    create_sliding_windows,
    split_data
)


def main(config_path: str = "configs/base_config.yaml"):
    """
    Main preprocessing pipeline.
    
    Steps:
    1. Load raw ETTm1 data
    2. Validate timestamps
    3. Handle missing values
    4. Split into train/val/test
    5. Normalize (fit on train only)
    6. Create sliding windows
    7. Save processed datasets
    """
    print("="  * 80)
    print("ETTm1 Data Preprocessing Pipeline")
    print("=" * 80)
    
    # Load config
    print(f"\nLoading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Paths
    dataset_name = data_config['dataset']
    raw_data_path = Path(data_config['paths']['raw_data']) / f"{dataset_name}.csv"
    processed_data_path = Path(data_config['paths']['processed_data'])
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    target_var = data_config['target_variable']
    context_length = data_config['context_length']
    forecast_horizon = data_config['forecast_horizon']
    train_ratio = data_config['train_ratio']
    val_ratio = data_config['val_ratio']
    test_ratio = data_config['test_ratio']
    use_calendar = data_config.get('use_calendar_features', True)
    
    print(f"  Target variable: {target_var}")
    print(f"  Context length (L): {context_length}")
    print(f"  Forecast horizon (H): {forecast_horizon}")
    print(f"  Calendar features: {use_calendar}")
    
    # Load data
    print(f"\nLoading data from {raw_data_path}")
    df = load_ett_data(str(raw_data_path))
    print(f"  Loaded {len(df)} samples")
    print(f"  Features: {df.columns.tolist()}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Validate timestamps
    print("\nValidating timestamps")
    df = validate_timestamps(df, expected_freq='15min')
    
    # Handle missing values
    print("\nHandling missing values")
    df = handle_missing_values(
        df,
        method=data_config.get('handle_missing', 'forward_fill')
    )
    
    # Split data
    print("\nSplitting data")
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Normalize
    print("\nNormalizing data")
    features = df.columns.tolist()
    
    # Fit normalization on train set only
    train_df_norm, norm_stats = normalize_data(
        train_df,
        features=features,
        fit=True
    )
    print(f"  Computed normalization stats from training set")
    
    # Apply to val and test
    val_df_norm, _ = normalize_data(val_df, features=features, stats=norm_stats, fit=False)
    test_df_norm, _ = normalize_data(test_df, features=features, stats=norm_stats, fit=False)
    
    # Create sliding windows
    print("\nCreating sliding windows")
    
    print("  Training set...")
    train_input, train_target, train_input_cal, train_future_cal = create_sliding_windows(
        train_df_norm,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        target_feature=target_var
    )
    print(f"    Created {len(train_input)} windows")
    
    print("  Validation set...")
    val_input, val_target, val_input_cal, val_future_cal = create_sliding_windows(
        val_df_norm,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        target_feature=target_var
    )
    print(f"    Created {len(val_input)} windows")
    
    print("  Test set...")
    test_input, test_target, test_input_cal, test_future_cal = create_sliding_windows(
        test_df_norm,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        target_feature=target_var
    )
    print(f"    Created {len(test_input)} windows")
    
    # Save processed data
    print("\nSaving processed data")
    
    train_data = {
        'input': train_input,
        'target': train_target,
        'input_calendar': train_input_cal,
        'future_calendar': train_future_cal,
        'norm_stats': norm_stats,
        'config': data_config
    }
    
    val_data = {
        'input': val_input,
        'target': val_target,
        'input_calendar': val_input_cal,
        'future_calendar': val_future_cal
    }
    
    test_data = {
        'input': test_input,
        'target': test_target,
        'input_calendar': test_input_cal,
        'future_calendar': test_future_cal
    }
    
    # Include dataset name in filename
    train_path = processed_data_path / f"{dataset_name}_train_L{context_length}_H{forecast_horizon}.pkl"
    val_path = processed_data_path / f"{dataset_name}_val_L{context_length}_H{forecast_horizon}.pkl"
    test_path = processed_data_path / f"{dataset_name}_test_L{context_length}_H{forecast_horizon}.pkl"
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"  Saved {train_path}")
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    print(f"  Saved {val_path}")
    
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  Saved {test_path}")
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"\nDataset shapes:")
    print(f"  Train: input{train_input.shape}, target{train_target.shape}")
    print(f"  Val:   input{val_input.shape}, target{val_target.shape}")
    print(f"  Test:  input{test_input.shape}, target{test_target.shape}")
    print(f"\nFiles saved to: {processed_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ETTm1 dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)

