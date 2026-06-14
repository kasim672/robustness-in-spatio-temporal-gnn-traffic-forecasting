# 🧠 Source Code (`src/`)

Core library implementing the complete traffic forecasting and robustness evaluation pipeline — from data loading and graph construction through model definition, training, corruption injection, evaluation, and visualization.

---

## Architecture Overview

```
src/
├── __init__.py              # Package initialization
├── config.py                # Central configuration hub
├── data_loader.py           # Data loading, normalization, sequence creation
├── graph_builder.py         # Graph construction and support matrix computation
├── train.py                 # Training loop with AMP and early stopping
├── evaluate.py              # Multi-horizon, multi-metric evaluation
├── robustness.py            # Corruption injection (random missing + sensor failure)
├── visualize.py             # Plot generation for all figures
├── interpretability.py      # Adjacency heatmaps and case study visualizations
└── models/                  # Forecasting model implementations
    ├── __init__.py
    ├── sanity_baselines.py  # Persistence, Historical Average
    ├── arima_model.py       # ARIMA(2,1,2) per-sensor forecasting
    ├── rf_model.py          # Random Forest (100 trees, max depth 15)
    ├── lstm_model.py        # 2-layer LSTM (hidden dim 64)
    ├── stgcn.py             # Spatio-Temporal Graph Convolutional Network
    └── dcrnn.py             # Diffusion Convolutional Recurrent Neural Network
```

---

## Module Reference

### `config.py` — Central Configuration

All hyperparameters, file paths, and reproducibility settings in one place. No magic numbers scattered across the codebase.

| Setting | Value | Purpose |
|---|---|---|
| `SEED` | 42 | Global random seed for reproducibility |
| `SEQ_LEN` | 12 | Input window (12 × 5 min = 60 min) |
| `PRED_LEN` | 12 | Forecast horizon (12 × 5 min = 60 min) |
| `BATCH_SIZE` | 128 | Training batch size (optimized from 64) |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `EPOCHS` | 100 | Max training epochs (early stopping applies) |
| `PATIENCE` | 15 | Early stopping patience |
| `TRAIN_RATIO` | 0.7 | Chronological train split |
| `VAL_RATIO` | 0.1 | Chronological validation split |
| `TEST_RATIO` | 0.2 | Chronological test split |
| `USE_AMP` | True | Automatic Mixed Precision (fp16) |
| `NUM_WORKERS` | 4 | DataLoader parallel workers |

```python
from src.config import *  # All settings accessible project-wide
```

---

### `data_loader.py` — Data Loading & Preprocessing

Handles the complete data pipeline with **leakage-free** design:

1. **CSV Loading** — reads `METR-LA.csv` or `PEMS-BAY.csv`
2. **Missing Value Handling** — forward fill → backward fill → column mean (no zero-speed artifacts)
3. **Chronological Split** — 70/10/20 train/val/test (preserves temporal order)
4. **Z-Score Normalization** — using **training statistics only**
5. **Sliding Window** — generates (X, Y) pairs with `SEQ_LEN=12` input and `PRED_LEN=12` target
6. **DataLoader Creation** — with persistent workers and prefetch for GPU throughput

**Key Classes:**
- `TrafficDataset(Dataset)` — PyTorch Dataset wrapping (X, Y) tensor pairs
- `load_and_preprocess(dataset_name)` — Full pipeline, returns DataLoaders + metadata

```python
from src.data_loader import load_and_preprocess

train_loader, val_loader, test_loader, scaler, metadata = load_and_preprocess("METR-LA")
```

---

### `graph_builder.py` — Graph Construction

Builds the spatial graph from sensor time-series data, with full support for STGCN (Chebyshev) and DCRNN (diffusion) architectures.

**Pipeline:**
1. Compute pairwise **Pearson correlation** between sensor time series
2. Apply **Gaussian kernel**: `w_ij = exp(-((1 - corr_ij)^2) / (2σ^2))` where σ = 0.1
3. Apply **sparsity threshold**: entries below ε = 0.3 set to zero
4. Add **self-loops**: `A_ii = 1`
5. Generate **support matrices**:
   - STGCN: symmetric normalized Laplacian → Chebyshev polynomials T₀, T₁, ..., Tₖ (K=3)
   - DCRNN: forward/backward random-walk transition matrices P_f^k, P_b^k (K=2)

**Key Functions:**
- `compute_correlation_adj(data, sigma, epsilon)` → adjacency matrix
- `compute_chebyshev_supports(adj, K)` → list of Chebyshev polynomial matrices
- `compute_diffusion_supports(adj, K)` → list of diffusion transition matrices
- `build_identity_graph(N)` → identity matrix (ablation)
- `build_random_graph(N, density)` → random symmetric graph (ablation)

> ⚠️ **Leakage prevention:** Graph is built from training data only — never from the full dataset.

---

### `train.py` — Training Loop

Handles model training with research-grade features:

| Feature | Implementation |
|---|---|
| **AMP Training** | `torch.cuda.amp.GradScaler` + `autocast` for fp16 speedup |
| **Early Stopping** | Monitors validation MAE with configurable patience |
| **LR Scheduling** | `ReduceLROnPlateau` for adaptive learning rate |
| **Checkpointing** | Saves best model based on validation MAE |
| **Teacher Forcing** | DCRNN-specific: gradual decay from 1.0 → 0.0 over training |
| **Timing** | Per-epoch wall-clock and throughput logging |

**Key Classes:**
- `EarlyStopping` — tracks best validation loss and triggers stopping
- `train_model(model, train_loader, val_loader, ...)` — complete training loop

```python
from src.train import train_model

train_model(model, train_loader, val_loader, scaler, config)
```

---

### `evaluate.py` — Multi-Metric Evaluation

Computes **de-normalized** metrics in original speed units (mph). All predictions are inverse-transformed before metric computation.

**Metrics:**
- **MAE** — Mean Absolute Error (primary metric)
- **RMSE** — Root Mean Square Error (penalizes outliers)
- **MAPE** — Mean Absolute Percentage Error (scale-independent)

**Horizons:** Evaluated at steps 3, 6, and 12 (corresponding to 15 min, 30 min, 60 min forecasts).

**Key Functions:**
- `masked_mae(preds, targets)` — MAE ignoring null values
- `masked_rmse(preds, targets)` — RMSE ignoring null values
- `masked_mape(preds, targets)` — MAPE ignoring null values
- `evaluate_model(model, test_loader, scaler)` → per-horizon + overall metrics dict

---

### `robustness.py` — Corruption Injection

Implements the two corruption scenarios evaluated in the paper. Operates on test-set **input** sequences only — targets remain clean.

**Corruption Types:**

| Type | Function | What it does |
|---|---|---|
| Random Missing | `inject_random_missing(data, ratio)` | Each entry independently zeroed with probability `ratio` |
| Sensor Failure | `inject_sensor_failure(data, ratio)` | `⌊ratio × N⌋` sensors have ALL historical readings zeroed |

**Design Principles:**
- Corruption uses independent random seeds (not the training seed)
- Binary mask `M` applied element-wise: `X_corrupted = M ⊙ X`
- 5 seeds per corruption ratio for statistical robustness (reports mean ± std)

```python
from src.robustness import inject_random_missing, inject_sensor_failure

corrupted_X = inject_random_missing(X_test, ratio=0.4, fill_value=0.0)
```

---

### `visualize.py` — Plot Generation

Generates all 27 figures in `results/plots/` using matplotlib and seaborn with a consistent research-paper style.

**Plot Categories:**
- Baseline comparison bar charts (MAE, RMSE, MAPE)
- Per-horizon line plots
- Robustness degradation curves with ±1σ confidence bands
- Adjacency matrix heatmaps
- Time-series prediction overlays
- Data distribution histograms
- Graph property analysis

**Style:** `seaborn-v0_8-darkgrid` theme with non-interactive `Agg` backend.

---

### `interpretability.py` — Research Visualizations

Additional visualization utilities for generating paper-quality explanatory figures:

- `plot_adjacency_heatmap(adj, ...)` — heatmap of the learned adjacency matrix
- Case study plots for specific sensor predictions

---

## 📂 models/

### `sanity_baselines.py` — Persistence & Historical Average

Zero-parameter baselines that establish the minimum performance threshold.

- **Persistence** — predicts `Y(t+h) = X(t)` for all horizons
- **Historical Average** — predicts `Y(t+h) = mean(training values at same time-of-day)`

### `arima_model.py` — ARIMA(2,1,2)

Per-sensor statistical model fitted independently to each of the N sensor time series. Uses `statsmodels` ARIMA with order (2,1,2).

### `rf_model.py` — Random Forest

Scikit-learn `RandomForestRegressor` with 100 trees, max depth 15. Input is the flattened sliding window `(SEQ_LEN × N_sensors)` → predicts each horizon independently.

### `lstm_model.py` — LSTM

Two-layer PyTorch LSTM with hidden dimension 64. Processes the sensor vector as a multivariate time series — no explicit spatial modeling.

```
Input (batch, 12, N) → LSTM(2 layers, hidden=64) → Linear → Output (batch, 12, N)
```

### `stgcn.py` — Spatio-Temporal Graph Convolutional Network

Fully convolutional architecture with two stacked ST-Conv blocks:

```
Input → [Temporal Conv → ChebGraphConv(K=3) → Temporal Conv → Norm → Dropout] ×2 → Output Conv → Predictions
```

- **Graph convolution:** Chebyshev polynomials of order K=3 on the scaled Laplacian
- **Temporal convolution:** 1D convolution (kernel=3) with Gated Linear Unit (GLU) activation
- **Channel progression:** [1, 16, 32, 64]
- **Output:** All 12 horizons generated simultaneously (parallel)

### `dcrnn.py` — Diffusion Convolutional Recurrent Neural Network

Encoder-decoder architecture with Diffusion Convolutional GRU (DCGRU) cells:

```
Encoder: Input → DCGRU(2 layers, hidden=64) → Final hidden state
Decoder: Hidden state → DCGRU(2 layers) → Sequential predictions (with teacher forcing)
```

- **Diffusion convolution:** bidirectional random-walk transition matrices (K=2 diffusion steps)
- **Gating:** standard GRU gates (reset, update) with diffusion conv replacing linear layers
- **Teacher forcing:** decays from 1.0 → 0.0 over training
- **Output:** 12 horizons generated sequentially (autoregressive)

---

## Data Flow

```
CSV Data
  │
  ▼
data_loader.py ──── Clean, normalize, split, create sequences
  │
  ├──▶ graph_builder.py ──── Build adjacency + support matrices
  │
  ▼
train.py ──── Train models (AMP, early stopping)
  │
  ▼
evaluate.py ──── Compute MAE/RMSE/MAPE (de-normalized)
  │
  ├──▶ robustness.py ──── Inject corruption, re-evaluate
  │
  ▼
visualize.py ──── Generate all plots
  │
  ▼
results/metrics/*.json + results/plots/*.png
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Train-only normalization** | Prevents information leakage from validation/test statistics |
| **Train-only graph** | Adjacency built from training partition only |
| **Chronological splits** | Time-series data cannot be randomly shuffled |
| **Per-chunk sequences** | Sliding windows never cross train/val/test boundaries |
| **De-normalized metrics** | All metrics computed in original mph — not normalized units |
| **AMP training** | ~1.5–1.8× speedup with minimal precision impact |
| **Separate ablation checkpoints** | Ablation runs never overwrite production models |
