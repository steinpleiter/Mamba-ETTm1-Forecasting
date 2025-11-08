# Mamba for Long-Horizon Time-Series Forecasting on ETTm1

**Author:** Stein Pleiter (1012872009)
**Email:** stein.pleiter@mail.utoronto.ca
**Repository:** https://github.com/steinpleiter/Mamba-ETTm1-Forecasting

---

## Introduction

### Goal

Build a forecasting model for the Electricity Transformer Temperature dataset (ETTm1) that predicts the next **H** time steps of oil temperature (OT) from a long history **L**. This project leverages **Mamba**, a recent state-space sequence model that runs in linear time and handles long context efficiently.

### Why This Is Interesting

Many real-world systems (energy, traffic, sensors) need forecasts over long horizons but have limited compute. While Transformers deliver strong results, they suffer from quadratic complexity with sequence length. Mamba is designed to be faster while maintaining accuracy, making it well-suited for long-horizon forecasting tasks.

## System Overview

```
Past L steps + calendar features 
    ↓
Mamba Stack (Linear-time SSM with N=6 blocks)
    ↓
Conv1D head (L → H mapping)
    ↓
Forecast Ŷ_{t+1:t+H} (OT)
```

**Optional components:**

- Future calendar conditioning
- Residual connection with daily seasonal-naive baseline (m=96)

---

## Dataset: ETTm1

- **Source:** Electricity Transformer Temperature dataset
- **Resolution:** 15-minute intervals
- **Input Features:** HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- **Target Variable:** OT (Oil Temperature)
- **Split:** Time-based train/validation/test split (no data leakage)

### Preprocessing Pipeline

1. **Timestamp parsing:** Verify 15-min spacing; handle gaps via forward-fill or window dropping
2. **Standardization:** Z-score normalization using train set statistics only
3. **Sliding windows:**
   - Context lengths: L ∈ {336, 512, 720}
   - Forecast horizons: H ∈ {24, 48, 96}
4. **Calendar features:** Sin/cos encodings for minute, hour, day-of-week, month (known future information)
5. **Leakage prevention:** Only past values and known future calendars used; no future targets

---

## Architecture

### Backbone: Mamba

- **Blocks:** N = 6 stacked Mamba blocks
- **Model dimension:** d_model = 256
- **Input projection:** d_in → d_model
- **Dropout:** Applied for regularization
- **Positional encoding:** Optional learned positional bias (ablation study)

### Prediction Head

- **One-shot forecasting:** Predict all H steps simultaneously
- **Architecture:**
  - 1D convolution: sequence length L → H
  - Linear layer: output OT predictions
- **Residual option:** Learn delta from daily seasonal baseline (m=96)

### Exogenous Feature Fusion

- Calendar features condition hidden states via small MLP
- Treated as ablation component

---

## Training Configuration

- **Optimizer:** AdamW with cosine learning rate schedule + warmup
- **Mixed Precision:** AMP (Automatic Mixed Precision)
- **Loss Function:** SmoothL1
- **Early Stopping:** Based on validation MAE
- **Metrics:** MAE, MSE, RMSE, MASE, sMAPE
- **Efficiency Metrics:** Throughput, latency, peak VRAM

---

## Baseline Models

1. **DLinear-style model:** Lightweight linear decomposition baseline
2. **Seasonal-naive:** Copy values from H steps earlier (m=96 for daily seasonality)

These baselines demonstrate the value-add of the Mamba architecture.

---

## Evaluation Metrics

- **MAE (Mean Absolute Error):** Primary metric for early stopping
- **MSE (Mean Squared Error):** Standard regression metric
- **RMSE (Root Mean Squared Error):** Same units as target variable
- **MASE (Mean Absolute Scaled Error):** Compares against seasonal baseline
- **sMAPE (Symmetric Mean Absolute Percentage Error):** Scale-independent metric

---

## Experiments & Ablations

Planned ablation studies:

- [ ] Context length: L ∈ {336, 512, 720}
- [ ] Forecast horizon: H ∈ {24, 48, 96}
- [ ] Positional encoding: with vs. without
- [ ] Calendar features: with vs. without
- [ ] Residual baseline: with vs. without
- [ ] Model depth: varying N (number of Mamba blocks)

---

## Ethical Considerations

- **Distribution shift:** Model trained on one period may underperform on different periods
- **Decision impact:** Forecasts can influence planning decisions in energy systems
- **Mitigation strategies:**
  - Strict time-based splits (no future information leakage)
  - Clear reporting of metrics and confidence intervals
  - Code and configuration release for reproducibility

---

## Project Motivation

This project explores the intersection of modern state-space models and practical time-series forecasting. After working on image classification tasks (CAPTCHA detection at 98% accuracy, breast cancer histopathology at 80-95% accuracy), this project deliberately shifts focus to long-horizon sequence forecasting.

**Why Mamba?** It offers a genuinely new approach—linear-time complexity as an alternative to attention mechanisms—while remaining solo-feasible and scientifically interesting. The project pairs solid engineering practices (clean splits, leakage checks, comprehensive baselines) with a cutting-edge model family.

---

## Repository Structure

```
Mamba-ETTm1-Forecasting/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup (optional)
├── .gitignore               # Git ignore rules
│
├── data/                    # Data directory (not committed)
│   ├── raw/                 # Original ETTm1 data
│   ├── processed/           # Preprocessed datasets
│   └── README.md            # Data documentation
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py       # PyTorch Dataset classes
│   │   └── preprocessing.py # Preprocessing utilities
│   │
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── mamba.py         # Mamba backbone
│   │   ├── baseline.py      # DLinear and seasonal-naive
│   │   └── utils.py         # Model utilities
│   │
│   ├── training/            # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── losses.py        # Loss functions
│   │
│   └── evaluation/          # Evaluation and metrics
│       ├── __init__.py
│       └── metrics.py       # MAE, MSE, MASE, etc.
│
├── configs/                 # Configuration files
│   ├── base_config.yaml     # Base configuration
│   └── experiments/         # Experiment-specific configs
│
├── scripts/                 # Executable scripts
│   ├── preprocess_data.py   # Data preprocessing
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── run_ablations.sh     # Ablation study runner
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_metrics.py
│
└── results/                 # Experiment results (not committed)
    ├── logs/                # Training logs
    ├── checkpoints/         # Model checkpoints
    └── figures/             # Generated plots
```

---

## Getting Started

### Installation with Rye

This project uses [Rye](https://rye-up.com/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/steinpleiter/Mamba-ETTm1-Forecasting.git
cd Mamba-ETTm1-Forecasting

# Sync dependencies (automatically creates venv and installs everything)
rye sync

# Activate the virtual environment (optional, rye run handles this)
. .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Quick Start

```bash
# 1. Download and preprocess data
rye run preprocess

# 2. Train the model
rye run train --config configs/base_config.yaml

# 3. Evaluate
rye run evaluate --checkpoint results/checkpoints/best_model.pt

# 4. Run tests
rye run test

# 5. Format code
rye run format

# 6. Lint code
rye run lint
```

---

## References

**Gu, A., & Dao, T.** (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*

**Cui, Y., et al.** (2021). A Neglected but Powerful Baseline for Long Sequence Time-Series Forecasting. *arXiv:2103.16349*

**Zeng, A., et al.** (2022). Are Transformers Effective for Time Series Forecasting? *arXiv:2205.13504*

**Nie, Y., et al.** (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). *ICLR 2023, arXiv:2211.14730*

**Das, A., et al.** (2024). TiDE: Time-series Dense Encoder for Long-term Forecasting. *arXiv:2304.08424*

**Tiezzi, M., et al.** (2025). State-Space Modeling in Long Sequence Processing: A Survey. *arXiv:2406.09062*

**Patro, B. N., et al.** (2024). Mamba-360: Survey of State Space Models as Transformer Alternatives. *arXiv:2404.16112*

**Hyndman, R. J., & Koehler, A. B.** (2006). Another Look at Forecast-Accuracy Metrics for Intermittent Demand. *Foresight*
