# 🗺️ Traffic Speed Datasets

This directory contains the two public benchmark datasets used in the robustness evaluation: **METR-LA** and **PEMS-BAY**. Both datasets record highway traffic speed from inductive loop detectors in California at 5-minute intervals.

<div align="center">

<img src="../assets/dataset.png" alt="Sensor Network Maps" width="90%">

*Sensor locations: (a) METR-LA — 207 loop detectors on Los Angeles highways; (b) PEMS-BAY — 325 detectors in the San Francisco Bay Area.*

</div>

---

## Dataset Summary

| Property | METR-LA | PEMS-BAY |
|---|---|---|
| **File** | `METR-LA.csv` | `PEMS-BAY.csv` |
| **File Size** | ~69 MB | ~82 MB |
| **Sensors (columns)** | 207 | 325 |
| **Timesteps (rows)** | 34,272 | 52,116 |
| **Duration** | Mar 1 – Jun 30, 2012 (4 months) | Jan 1 – May 31, 2017 (6 months) |
| **Sampling Interval** | 5 minutes | 5 minutes |
| **Feature** | Speed (mph) | Speed (mph) |
| **Region** | Los Angeles County, CA | San Francisco Bay Area, CA |
| **Source Infrastructure** | Caltrans highway loop detectors | Caltrans PeMS loop detectors |

---

## CSV Format

Both files follow an identical structure:

```
<timestamp>, <sensor_1>, <sensor_2>, ..., <sensor_N>
```

| Column | Type | Description |
|---|---|---|
| Column 0 | `datetime` | Timestamp in `YYYY-MM-DD HH:MM:SS` format |
| Columns 1–N | `float64` | Traffic speed (mph) at each sensor for that 5-min interval |

### Example (METR-LA, first 2 rows)

```
,773869,767541,767542,...,769373
2012-03-01 00:00:00,64.375,67.625,67.125,...,61.875
2012-03-01 00:05:00,62.667,68.556,65.444,...,62.875
```

- Column headers are **sensor IDs** (Caltrans detector station numbers)
- The first column (index) contains the timestamp
- Speed values are aggregated over each 5-minute window

## Data Splits

All splits are **strictly chronological** to prevent temporal information leakage:

| Split | Fraction | METR-LA Timesteps | PEMS-BAY Timesteps |
|---|---|---|---|
| **Train** | 70% | 23,990 | 36,481 |
| **Validation** | 10% | 3,427 | 5,212 |
| **Test** | 20% | 6,855 | 10,423 |

> ⚠️ **No random shuffling.** The train set always precedes validation, which always precedes test. This mirrors real deployment where models cannot see future data.

---

## Citation

If you use these datasets, please cite the original source:

```bibtex
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```

---

## Usage

```python
from src.data_loader import load_and_preprocess

# Load METR-LA with full preprocessing pipeline
train_loader, val_loader, test_loader, scaler, metadata = load_and_preprocess("METR-LA")

# Load PEMS-BAY
train_loader, val_loader, test_loader, scaler, metadata = load_and_preprocess("PEMS-BAY")
```

The `load_and_preprocess` function handles all steps: CSV reading, missing value imputation, chronological splitting, Z-score normalization (train stats only), sliding window creation, and DataLoader setup.
