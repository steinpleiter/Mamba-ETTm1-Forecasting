# Mamba vs. LSTM: Long-Horizon Time-Series Forecasting on ETTm1

**Author:** Stein Pleiter (1012872009)
**Email:** stein.pleiter@mail.utoronto.ca
**Repository:** https://github.com/steinpleiter/Mamba-ETTm1-Forecasting

---

## Introduction

Efficient forecasting of electricity transformer oil temperature (OT) is critical for grid stability and preventing equipment failure. Long-horizon forecasting (H = 96 steps) is particularly challenging due to error accumulation and complex temporal dependencies.

This project evaluates **Mamba**, a novel linear-time State Space Model (SSM), against a robust **LSTM** baseline and **DLinear** on the ETTm1 dataset. We investigate whether Mamba's computational benefits translate to performance gains on mid-sized time-series data.

---

## Results Summary

| Model | MAE | RMSE | MASE | Params | vs. Naive |
|-------|-----|------|------|--------|-----------|
| Seasonal Naive | 1.123 | 1.406 | 28.72 | 0 | 0.0% |
| DLinear | 0.625 | 0.717 | 15.99 | 1.48M | +44.3% |
| Mamba | 0.469 | 0.534 | 11.99 | 161k | +58.2% |
| **LSTM** | **0.280** | **0.342** | **7.16** | 221k | **+75.1%** |

**Key Finding:** LSTM outperformed Mamba on this mid-sized dataset (L=512), achieving a 40% reduction in MAE. While Mamba offers linear scaling benefits for very long sequences, LSTM's recurrence mechanism proved more effective at this context length.

---

## Dataset: ETTm1

- **Source:** Electricity Transformer Temperature dataset
- **Resolution:** 15-minute intervals (96 steps = 24 hours)
- **Features:** HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (7 total)
- **Target:** OT (Oil Temperature)

### Data Statistics

| Split | Samples | Mean OT | Std OT |
|-------|---------|---------|--------|
| Train | 41,808 | 0.00 | 1.00 |
| Val | 13,936 | −1.21 | 0.52 |
| Test | 13,936 | −1.12 | 0.41 |

*Z-score normalized using training set statistics only (no leakage).*

### Preprocessing

1. **Cleaning:** No missing values found
2. **Split:** Strict chronological 60/20/20 split
3. **Normalization:** Z-score (μ=0, σ=1) fitted on training data only
4. **Windowing:** Input L=512, Horizon H=96
5. **Calendar Features:** Sin/cos encodings of Minute, Hour, Day of Week, Month

---

## Architecture

### LSTM (Baseline)
- 2-layer stacked LSTM
- hidden_size=128
- Linear projection head
- **220,832 parameters**

### Mamba (Primary)
- d_model=32, n_layers=1
- patch_len=16, dropout=0.3
- **161,374 parameters**

### DLinear
- Decomposition-linear model
- Trend + seasonal components
- **1,477,440 parameters**

### Seasonal Naive
- Non-learning baseline
- ŷ[t+h] = y[t+h−96] (copies from 24h ago)

---

## Key Observations

### Distribution Shift
The systematic offset in predictions is caused by non-stationarity:
- **Train period:** July 2016 – Sept 2017 (warmer, ~17°C mean)
- **Test period:** Feb 2018 – June 2018 (colder, ~8°C mean)

Normalizing with training statistics anchored models to warmer temperatures, causing positive bias on colder test data.

### Mamba vs. LSTM
- LSTM captures sharp peaks and troughs more precisely
- Mamba tends to smooth high-frequency variations
- For L=512, LSTM's recurrence mechanism provides stronger inductive bias

---

## Project Structure

```
Mamba-ETTm1-Forecasting/
├── configs/                 # Configuration files
│   └── base_config.yaml
├── data/
│   ├── raw/                 # Original ETTm1.csv
│   └── processed/           # Preprocessed .pkl files
├── notebooks/               # Jupyter notebooks
│   └── Colab_Setup.ipynb    # Google Colab training notebook
├── scripts/
│   ├── train_mamba.py       # Mamba training script
│   ├── train_lstm.py        # LSTM training script
│   ├── train_dlinear.py     # DLinear training script
│   └── preprocess_data.py   # Data preprocessing
├── src/
│   ├── data/                # Dataset and preprocessing
│   ├── models/              # Model architectures
│   ├── training/            # Training utilities
│   └── evaluation/          # Metrics
└── results/
    ├── checkpoints/         # Saved models
    └── figures/             # Visualizations
```

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/steinpleiter/Mamba-ETTm1-Forecasting.git
cd Mamba-ETTm1-Forecasting

# Using Rye (recommended)
rye sync

# Or using pip
pip install -r requirements.txt
```

### Training

```bash
# Preprocess data
rye run python scripts/preprocess_data.py

# Train LSTM
rye run python scripts/train_lstm.py --device mps --epochs 30

# Train Mamba (requires CUDA)
rye run python scripts/train_mamba.py --device cuda --epochs 50 --d_model 32 --n_layers 1

# Train DLinear
rye run python scripts/train_dlinear.py --device mps --epochs 30
```

### Google Colab
For Mamba training with CUDA support, use `notebooks/Colab_Setup.ipynb`.

---

## Ethical Considerations

- **Distribution shift risk:** Models trained on historical data may fail during anomalous events (e.g., extreme weather)
- **Mitigation:** Strict time-based splits, comprehensive baseline comparisons, transparent reporting
- **Efficiency:** Mamba's linear complexity offers a path toward lower-carbon AI training

---

## References

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*
2. Cui, Y., et al. (2021). A Neglected but Powerful Baseline for Long Sequence Time-Series Forecasting. *arXiv:2103.16349*
3. Zeng, A., et al. (2022). Are Transformers Effective for Time Series Forecasting? *arXiv:2205.13504*
4. Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). *arXiv:2211.14730*
5. Das, A., et al. (2023). TiDE: Time-series Dense Encoder for Long-term Forecasting. *arXiv:2304.08424*
6. Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI*
