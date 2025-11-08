"""
Master script to run complete experiment pipeline.

Runs:
1. Data preprocessing
2. Seasonal Naive baseline
3. DLinear training
4. LSTM training
5. Model comparison

Usage:
    python scripts/run_all.py --device mps
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed!")
        sys.exit(1)
    
    print(f"\nCompleted in {elapsed:.1f} seconds")
    return elapsed


def main(args):
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE EXPERIMENT PIPELINE")
    print("=" * 80)
    
    total_start = time.time()
    timings = {}
    
    # Step 1: Preprocess data
    timings['preprocess'] = run_command(
        ['python', 'scripts/preprocess_data.py', '--config', args.config],
        "1. Data Preprocessing"
    )
    
    # Step 2: Evaluate Seasonal Naive baseline
    timings['seasonal_naive'] = run_command(
        ['python', 'scripts/evaluate_baseline.py',
         '--model', 'seasonal_naive',
         '--device', args.device,
         '--save_results'],
        "2. Seasonal Naive Baseline"
    )
    
    # Step 3: Train DLinear
    timings['dlinear'] = run_command(
        ['python', 'scripts/train_dlinear.py',
         '--device', args.device,
         '--individual',
         '--epochs', str(args.epochs),
         '--batch_size', str(args.batch_size)],
        "3. Train DLinear Model"
    )
    
    # Step 4: Train LSTM
    timings['lstm'] = run_command(
        ['python', 'scripts/train_lstm.py',
         '--device', args.device,
         '--epochs', str(args.epochs),
         '--batch_size', str(args.batch_size),
         '--hidden_size', str(args.hidden_size),
         '--num_layers', str(args.num_layers)],
        "4. Train LSTM Model"
    )
    
    # Step 5: Compare models
    timings['compare'] = run_command(
        ['python', 'scripts/compare_models.py'],
        "5. Compare All Models"
    )
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    
    print("\nTiming Summary:")
    for step, duration in timings.items():
        print(f"  {step:20s}: {duration:6.1f}s")
    print(f"  {'TOTAL':20s}: {total_elapsed:6.1f}s ({total_elapsed/60:.1f} min)")
    
    print("\nResults saved in:")
    print("  results/seasonal_naive_baseline_results.pkl")
    print("  results/dlinear_ETTm1_training_results.pkl")
    print("  results/lstm_ETTm1_training_results.pkl")
    print("  results/checkpoints/")
    print("  results/figures/")
    
    print("\nNext steps:")
    print("  1. Check results/figures/ for comparison plots")
    print("  2. Update progress report with actual metrics")
    print("  3. Add sample forecast plots if needed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete experiment pipeline")
    
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
        '--hidden_size',
        type=int,
        default=128,
        help='LSTM hidden size'
    )
    
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of LSTM layers'
    )
    
    args = parser.parse_args()
    main(args)

