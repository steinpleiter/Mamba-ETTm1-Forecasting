"""
Compare performance of different trained models.

Usage:
    rye run python scripts/compare_models.py
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(filename):
    """Load results from pickle file."""
    path = Path('results') / filename
    if not path.exists():
        return None
    
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    
    # Load all available results
    models = {}
    
    # Baseline
    seasonal_naive = load_results('seasonal_naive_baseline_results.pkl')
    if seasonal_naive:
        models['Seasonal Naive'] = seasonal_naive
    
    # DLinear
    dlinear = load_results('dlinear_training_results.pkl')
    if dlinear:
        models['DLinear'] = dlinear
    
    # DLinear ETTm2
    dlinear_ettm2 = load_results('dlinear_ETTm2_training_results.pkl')
    if dlinear_ettm2:
        models['DLinear (ETTm2)'] = dlinear_ettm2
    
    # LSTM
    lstm = load_results('lstm_training_results.pkl')
    if lstm:
        models['LSTM'] = lstm
    
    # LSTM ETTm1
    lstm_ettm1 = load_results('lstm_ETTm1_training_results.pkl')
    if lstm_ettm1:
        models['LSTM (ETTm1)'] = lstm_ettm1
    
    # Mamba
    mamba = load_results('mamba_training_results.pkl')
    if mamba:
        models['Mamba'] = mamba
    
    if len(models) == 0:
        print("No results found! Train models first.")
        return
    
    print(f"\nFound {len(models)} model results:")
    for name in models.keys():
        print(f"  - {name}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("VALIDATION SET COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MASE':>10} {'Parameters':>15}")
    print("-" * 80)
    
    for name, result in models.items():
        metrics = result.get('val_metrics', {})
        n_params = result.get('n_parameters', 0)
        
        mae = metrics.get('mae', np.nan)
        rmse = metrics.get('rmse', np.nan)
        mase = metrics.get('mase', np.nan)
        
        print(f"{name:<20} {mae:>10.4f} {rmse:>10.4f} {mase:>10.2f} {n_params:>15,}")
    
    # Test set comparison if available
    print("\n" + "=" * 80)
    print("TEST SET COMPARISON")
    print("=" * 80)
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MASE':>10} {'vs Naive':>12}")
    print("-" * 80)
    
    baseline_mae = 1.123  # Seasonal naive test MAE
    
    for name, result in models.items():
        metrics = result.get('test_metrics', result.get('val_metrics', {}))
        mae = metrics.get('mae', np.nan)
        rmse = metrics.get('rmse', np.nan)
        mase = metrics.get('mase', np.nan)
        
        if not np.isnan(mae):
            improvement = ((baseline_mae - mae) / baseline_mae) * 100
            vs_naive = f"+{improvement:.1f}%"
        else:
            vs_naive = "N/A"
        
        print(f"{name:<20} {mae:>10.4f} {rmse:>10.4f} {mase:>10.2f} {vs_naive:>12}")
    
    # Plot training curves if available
    trainable_models = {k: v for k, v in models.items() if 'history' in v}
    
    if len(trainable_models) > 0:
        print("\n" + "=" * 80)
        print("Generating training curve plots...")
        print("=" * 80)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1 = axes[0]
        for name, result in trainable_models.items():
            history = result['history']
            ax1.plot(history['val_loss'], label=name, linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Loss', fontsize=12)
        ax1.set_title('Training Curves - Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # MAE curves
        ax2 = axes[1]
        for name, result in trainable_models.items():
            history = result['history']
            ax2.plot(history['val_mae'], label=name, linewidth=2)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation MAE', fontsize=12)
        ax2.set_title('Training Curves - Validation MAE', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = Path('results/figures')
        fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path / 'model_comparison.png', dpi=150, bbox_inches='tight')
        
        print(f"  Saved: results/figures/model_comparison.png")
        plt.show()
        
        # Bar chart of final performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = list(models.keys())
        test_maes = [models[name].get('test_metrics', models[name].get('val_metrics', {})).get('mae', np.nan) 
                     for name in model_names]
        
        colors = ['red' if 'Naive' in name else 'orange' if 'DLinear' in name else 'green' 
                  for name in model_names]
        
        bars = ax.bar(model_names, test_maes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, mae in zip(bars, test_maes):
            if not np.isnan(mae):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mae:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Test MAE (lower is better)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(test_maes) * 1.15)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(fig_path / 'final_performance.png', dpi=150, bbox_inches='tight')
        
        print(f"  Saved: results/figures/final_performance.png")
        plt.show()
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    
    # Summary
    best_model = min(models.items(), 
                    key=lambda x: x[1].get('val_metrics', {}).get('mae', float('inf')))
    
    print(f"\nBest Model: {best_model[0]}")
    print(f"  Validation MAE: {best_model[1]['val_metrics']['mae']:.4f}")
    
    if 'test_metrics' in best_model[1]:
        print(f"  Test MAE: {best_model[1]['test_metrics']['mae']:.4f}")


if __name__ == "__main__":
    main()

