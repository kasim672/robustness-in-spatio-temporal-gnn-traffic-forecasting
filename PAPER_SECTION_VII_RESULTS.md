# VII. RESULTS

## A. Clean-Data Accuracy

DCRNN obtains the best overall MAE on both datasets. On METR-LA, DCRNN achieves 3.548 mph overall MAE, followed by STGCN at 3.634 mph and LSTM at 3.707 mph. On PEMS-BAY, DCRNN achieves 1.905 mph, followed by LSTM at 2.071 mph, Random Forest at 2.113 mph, and Persistence at 2.170 mph. STGCN performs strongly on METR-LA but is less competitive on PEMS-BAY; one possible explanation is over-smoothing in the denser graph (7.6 avg. connections/node vs. 2.2), though this hypothesis requires further validation (see Section VIII).

Notably, STGCN achieves the best 60-minute MAE on METR-LA (4.242 vs. DCRNN's 4.541), a 6.6% advantage at the longest horizon. STGCN also achieves the lowest overall RMSE on METR-LA (6.291 vs. DCRNN's 6.672), indicating fewer extreme prediction errors despite a higher average error. All learned models substantially outperform the Persistence and Historical Average baselines, confirming that both temporal dynamics and learned representations contribute meaningfully to forecasting accuracy.

### TABLE III: Clean-Data Forecasting Performance (Lower Is Better)

| Dataset | Model | Type | 15 min MAE | 30 min MAE | 60 min MAE | Overall MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|---|---|---|
| METR-LA | **DCRNN** | GNN | **2.868** | **3.540** | 4.541 | **3.548** | 6.672 | 9.75 |
| METR-LA | STGCN | GNN | 3.214 | 3.605 | **4.242** | 3.634 | **6.291** | 9.75 |
| METR-LA | LSTM | DL | 2.988 | 3.685 | 4.759 | 3.707 | 7.027 | 10.66 |
| METR-LA | Random Forest | ML | 3.049 | 3.753 | 4.798 | 3.758 | 7.142 | 10.80 |
| METR-LA | Persistence | Base | 3.161 | 3.831 | 4.905 | 3.856 | 7.806 | 9.94 |
| METR-LA | ARIMA(2,1,2) | Stat. | 3.324 | 3.947 | 5.009 | 3.990 | 8.126 | 10.38 |
| METR-LA | Historical Avg. | Base | 5.120 | 5.120 | 5.120 | 5.120 | 8.777 | 15.93 |
| PEMS-BAY | **DCRNN** | GNN | **1.472** | **1.942** | **2.519** | **1.905** | **4.026** | **4.37** |
| PEMS-BAY | LSTM | DL | 1.523 | 2.083 | 2.864 | 2.071 | 4.500 | 4.90 |
| PEMS-BAY | Random Forest | ML | 1.553 | 2.135 | 2.920 | 2.113 | 4.665 | 5.01 |
| PEMS-BAY | Persistence | Base | 1.591 | 2.171 | 3.039 | 2.170 | 5.123 | 4.67 |
| PEMS-BAY | ARIMA(2,1,2) | Stat. | 1.747 | 2.292 | 3.119 | 2.294 | 5.489 | 5.00 |
| PEMS-BAY | STGCN | GNN | 2.077 | 2.302 | 2.604 | 2.299 | 4.232 | 5.53 |
| PEMS-BAY | Historical Avg. | Base | 3.559 | 3.559 | 3.559 | 3.559 | 6.827 | 8.76 |

> **Fig. 3.** Overall clean-data MAE on METR-LA and PEMS-BAY. DCRNN achieves the lowest MAE on both datasets, but clean accuracy alone does not determine robustness. (`results/plots/METR-LA_baselines_comparison.png`, `results/plots/PEMS-BAY_baselines_comparison.png`)

> **Fig. 4.** Per-horizon MAE comparison on METR-LA. STGCN surpasses DCRNN at the 60-minute horizon, suggesting that parallel temporal convolution degrades more gracefully over longer prediction windows than autoregressive decoding. (`results/plots/METR-LA_baselines_horizon.png`, `results/plots/PEMS-BAY_baselines_horizon.png`)

---

## B. Random Missing Data Robustness

Table IV shows that STGCN is the most robust GNN under random missing data on both datasets. On METR-LA, STGCN degrades by 27.0% at 40% corruption, while DCRNN degrades by 34.5%. On PEMS-BAY, the contrast is more pronounced: STGCN degrades 39.1% versus DCRNN's 70.3%. This indicates that spectral graph smoothing and non-autoregressive temporal convolution are more stable under scattered input corruption than diffusion-recurrent decoding.

The standard deviations under random missing are very small across all models (σ ≤ 0.008), indicating that this corruption type produces consistent degradation regardless of which specific entries are removed. This contrasts sharply with sensor failure (Section C), where standard deviations are 10–50× larger.

### TABLE IV: Robustness Under Random Missing Data (mean ± std over 5 seeds)

| Dataset | Model | Clean MAE | 10% MAE | 20% MAE | 30% MAE | 40% MAE | Δ40% | Rank | Interpretation |
|---|---|---|---|---|---|---|---|---|---|
| METR-LA | ARIMA | 3.990 | 4.249±0.004 | 4.513±0.006 | 4.777±0.006 | 5.040±0.008 | +26.3% | 1 | Most robust overall |
| METR-LA | **STGCN** | 3.634 | 3.842±0.002 | 4.082±0.004 | 4.337±0.002 | **4.615±0.004** | **+27.0%** | **2** | **Most robust GNN** |
| METR-LA | Persistence | 3.856 | 4.134±0.003 | 4.414±0.002 | 4.692±0.001 | 4.970±0.003 | +28.9% | 3 | Strong naive baseline |
| METR-LA | RF | 3.758 | 4.036±0.001 | 4.319±0.001 | 4.606±0.001 | 4.897±0.002 | +30.3% | 4 | Flattened input limits corruption spread |
| METR-LA | LSTM | 3.707 | 4.014±0.001 | 4.306±0.001 | 4.593±0.001 | 4.885±0.001 | +31.8% | 5 | Stable temporal baseline |
| METR-LA | DCRNN | 3.548 | 3.873±0.001 | 4.180±0.001 | 4.478±0.002 | 4.772±0.004 | +34.5% | 6 | Best clean, weakest robustness |
| PEMS-BAY | **STGCN** | 2.299 | 2.462±0.002 | 2.688±0.003 | 2.935±0.003 | **3.199±0.002** | **+39.1%** | **1** | **Most robust** |
| PEMS-BAY | Persistence | 2.170 | 2.493±0.001 | 2.817±0.001 | 3.141±0.002 | 3.464±0.001 | +59.6% | 2 | Baseline competitive |
| PEMS-BAY | ARIMA | 2.294 | 2.640±0.002 | 2.980±0.004 | 3.307±0.006 | 3.627±0.005 | +58.1% | 3 | Lower than METR-LA ranking |
| PEMS-BAY | LSTM | 2.071 | 2.434±0.001 | 2.739±0.001 | 3.040±0.001 | 3.351±0.001 | +61.8% | 4 | Moderate robustness |
| PEMS-BAY | RF | 2.113 | 2.480±0.001 | 2.844±0.001 | 3.203±0.002 | 3.555±0.001 | +68.2% | 5 | High degradation |
| PEMS-BAY | DCRNN | 1.905 | 2.240±0.001 | 2.562±0.002 | 2.895±0.003 | 3.244±0.001 | +70.3% | 6 | Largest relative loss |

> **Fig. 5.** METR-LA random missing-data robustness curves. STGCN has higher clean MAE than DCRNN but a consistently shallower degradation slope under increasing corruption. (`results/plots/METR-LA_robustness_curves.png`)

> **Fig. 6.** Relative degradation (%) at 40% random missing data across both datasets. DCRNN experiences the largest relative degradation on both datasets, while STGCN is consistently the most robust GNN. The gap widens substantially on the denser PEMS-BAY graph.

---

## C. Sensor Failure Robustness

Sensor failure is more structured than random missing data because an entire node's history becomes unavailable, creating spatially correlated degradation. The robustness ranking changes substantially compared to random missing.

On METR-LA, LSTM degrades by 31.1%, outperforming both GNNs: DCRNN degrades by 33.0% and STGCN by 34.9%. This reversal suggests that graph message passing can transmit a failed sensor's corrupted zero-valued representation to neighboring nodes, amplifying the damage beyond the failed sensor itself. LSTM, which has no explicit spatial connectivity, is affected only at the failed sensor without propagation.

On PEMS-BAY, STGCN remains comparatively robust under sensor failure (48.3% degradation), possibly because the denser graph (7.6 connections/node) provides alternative spatial paths for information flow around failed sensors. This interpretation is consistent with the higher graph connectivity but requires further investigation.

The notably higher standard deviation for STGCN under sensor failure (±0.213 at 40% on METR-LA, vs. ±0.004 under random missing) indicates that STGCN's performance is highly sensitive to *which specific sensors* fail, suggesting a strong dependence on the spatial distribution of failures relative to the graph topology.

### TABLE V: Robustness Under Sensor Failure at 40% Corruption (mean ± std over 5 seeds)

| Dataset | Model | Clean MAE | 40% MAE | Δ MAE | Degradation | Rank | Interpretation |
|---|---|---|---|---|---|---|---|
| METR-LA | ARIMA | 3.990 | 5.014±0.178 | +1.024 | 25.7% | 1 | Most robust (no spatial dep.) |
| METR-LA | Persistence | 3.856 | 4.934±0.181 | +1.078 | 28.0% | 2 | Copy-last unaffected by graph |
| METR-LA | RF | 3.758 | 4.888±0.165 | +1.130 | 30.0% | 3 | Flattened input limits propagation |
| METR-LA | **LSTM** | **3.707** | **4.858±0.164** | **+1.151** | **31.1%** | **4** | **Best learned model** |
| METR-LA | DCRNN | 3.548 | 4.718±0.169 | +1.170 | 33.0% | 5 | Graph propagates failure |
| METR-LA | STGCN | 3.634 | 4.903±0.213 | +1.269 | 34.9% | 6 | Spectral conv amplifies failure |
| PEMS-BAY | STGCN | 2.299 | 3.409±0.210 | +1.110 | 48.3% | 1 | Dense graph routes around failure |
| PEMS-BAY | ARIMA | 2.294 | 3.478±0.043 | +1.184 | 51.6% | 2 | Robust per-sensor model |
| PEMS-BAY | Persistence | 2.170 | 3.404±0.041 | +1.234 | 56.9% | 3 | Baseline |
| PEMS-BAY | RF | 2.113 | 3.387±0.040 | +1.274 | 60.3% | 4 | Moderate degradation |
| PEMS-BAY | LSTM | 2.071 | 3.398±0.040 | +1.327 | 64.1% | 5 | Higher PEMS-BAY sensitivity |
| PEMS-BAY | DCRNN | 1.905 | 3.189±0.041 | +1.284 | 67.4% | 6 | Largest relative loss |

### TABLE VI: Corruption-Type Comparison — Key Reversal (METR-LA at 40%)

| Model | Random Missing Deg. | Sensor Failure Deg. | Rank Change | Interpretation |
|---|---|---|---|---|
| STGCN | 27.0% (Rank 2) | 34.9% (Rank 6) | **↓4** | Graph smoothing helps random, hurts spatial failure |
| DCRNN | 34.5% (Rank 6) | 33.0% (Rank 5) | ↑1 | Consistently vulnerable |
| LSTM | 31.8% (Rank 5) | 31.1% (Rank 4) | ↑1 | No graph = immune to spatial propagation |

This reversal is a central finding: graph convolution acts as a spatial low-pass filter that smooths randomly scattered corruption but amplifies spatially concentrated failures. The same architectural feature that provides robustness under one corruption type becomes a vulnerability under another.

> **Fig. 7.** Sensor failure robustness curves on METR-LA. Standard deviation bands are substantially wider than under random missing, reflecting sensitivity to which specific sensors fail. (`results/plots/METR-LA_robustness_curves.png`)

---

## D. Graph Structure Ablation

The ablation study compares learned graph, identity graph, and random graph settings on METR-LA. The learned graph improves DCRNN by 1.92% over identity, while STGCN improves by only 0.95%. Random graphs perform comparably, and sometimes marginally better, than the learned graph.

### TABLE VII: Graph Structure Ablation on METR-LA

| Model | Graph | Overall MAE | 15 min | 30 min | 60 min | RMSE | MAPE (%) | Δ vs. Learned |
|---|---|---|---|---|---|---|---|---|
| STGCN | Learned | 3.634 | 3.214 | 3.605 | 4.242 | 6.291 | 9.75 | — |
| STGCN | Identity | 3.669 | — | — | — | — | — | +0.95% |
| STGCN | Random | 3.630 | — | — | — | — | — | −0.11% |
| DCRNN | Learned | 3.548 | 2.940 | 3.555 | 4.541 | 6.672 | 9.75 | — |
| DCRNN | Identity | 3.616 | 2.940 | 3.604 | 4.618 | 7.012 | 10.25 | +1.92% |
| DCRNN | Random | 3.532 | 2.943 | 3.555 | 4.374 | 6.584 | 9.92 | −0.45% |

Two plausible interpretations exist for the modest identity-graph degradation:

1. **Temporal dominance:** The temporal modules (gated CNN for STGCN, DCGRU for DCRNN) capture the majority of the predictive signal, so removing spatial edges has limited impact.
2. **Weak graph quality:** The learned correlation-based graph at ε=0.3 produces a very sparse topology (2.2 connections/node, 48.3% isolated nodes), which may already provide insufficient spatial information.

The random-graph configuration achieves comparable or marginally better performance than the learned graph. This comparison is based on a single training run per configuration; definitive conclusions about the relative importance of learned versus random graph topology would require multiple training seeds and statistical significance testing.

### TABLE VIII: Graph Sparsity Analysis (METR-LA)

| ε Threshold | Edges | Avg. Conn./Node | Isolated Nodes (%) | λ₂ (Algebraic Connectivity) | Spectral Gap |
|---|---|---|---|---|---|
| 0.1 | 424 | 0.83 | 33.8% | 0.292 | 0.0252 |
| 0.2 | 308 | 0.75 | 39.6% | 0.288 | 0.0111 |
| **0.3 (used)** | **240** | **0.67** | **48.3%** | **0.294** | **0.0034** |
| 0.5 | 148 | 0.50 | 59.9% | 0.329 | 0.0069 |

The selected threshold ε=0.3 produces a graph where nearly half the nodes are isolated. This raises the question of whether the limited ablation impact reflects temporal dominance or simply insufficient graph quality. The spectral gap (λ₂ − λ₁) decreases monotonically from 0.025 to 0.003 as ε increases, indicating weakening graph connectivity.

> **Fig. 8.** Graph sparsity analysis showing the relationship between ε threshold, number of edges, and model performance. (`results/plots/METR-LA_sparsity_analysis.png`)

---

## E. Summary of Key Findings

The experimental results yield three principal observations:

1. **Accuracy-robustness trade-off.** DCRNN achieves the best clean accuracy on both datasets (MAE 3.548 METR-LA, 1.905 PEMS-BAY) but exhibits the highest degradation under random missing corruption (34.5% and 70.3% respectively). STGCN sacrifices 2.4% clean accuracy on METR-LA but achieves 22% lower degradation under the same corruption.

2. **Corruption-type dependence.** The robustness ranking reverses between random missing and sensor failure. Under random missing on METR-LA, STGCN ranks 2nd (27.0%); under sensor failure, it drops to 6th (34.9%). The non-graph LSTM (31.1% sensor failure degradation) outperforms both GNNs because graph message passing propagates spatially correlated failures to neighboring nodes.

3. **Marginal graph topology benefit.** In our ablation experiments, the learned correlation-based graph provides a modest improvement (0.95–1.92%) over an identity graph on METR-LA. Random-graph performance is comparable in a single-run comparison. These observations are preliminary; multiple seeds and statistical testing are required before definitive claims about graph topology importance can be made.

---

## Available Figures

| Figure | File | Description |
|---|---|---|
| Fig. 3 | `results/plots/METR-LA_baselines_comparison.png` | Multi-metric bar chart — METR-LA |
| Fig. 4 | `results/plots/METR-LA_baselines_horizon.png` | Per-horizon MAE comparison — METR-LA |
| Fig. 5 | `results/plots/METR-LA_robustness_curves.png` | Robustness degradation curves with ±1σ bands |
| Fig. 6 | `results/plots/PEMS-BAY_baselines_horizon.png` | Per-horizon MAE comparison — PEMS-BAY |
| Fig. 7 | `results/plots/PEMS-BAY_baselines_comparison.png` | Multi-metric bar chart — PEMS-BAY |
| Fig. 8 | `results/plots/METR-LA_sparsity_analysis.png` | Graph sparsity threshold vs. performance |
| Fig. 9 | `results/plots/METR-LA_pred_sensor0.png` | Prediction vs. ground truth — sensor 0 |
| Fig. 10 | `results/plots/METR-LA_pred_sensor50.png` | Prediction vs. ground truth — sensor 50 |
| Fig. 11 | `results/plots/final_horizon_analysis.png` | Full horizon analysis |
| Fig. 12 | `results/plots/adjacency_matrix_detailed.png` | Adjacency matrix heatmap |

## Available Data Files

| File | Contents |
|---|---|
| `results/paper_assets/table_main_results.tex` | LaTeX table for Table III |
| `results/paper_assets/table_robustness.tex` | LaTeX tables for Tables IV–V |
| `results/paper_assets/table_dataset_stats.tex` | LaTeX table for Table I |
| `results/paper_assets/table_architecture.tex` | LaTeX table for Table II |
| `results/paper_assets/main_results.csv` | Machine-readable clean results |
| `results/paper_assets/robustness_random_missing.csv` | Full robustness data (random) |
| `results/paper_assets/robustness_sensor_failure.csv` | Full robustness data (sensor failure) |

## ⚠️ Action Required: Re-run Robustness for RMSE + MAPE

The existing robustness JSON files contain **MAE only**. The script `run_robustness.py` has been updated to collect all three metrics (MAE, RMSE, MAPE) per seed. To populate the full data:

```bash
python run_robustness.py --dataset METR-LA     # ~15 min
python run_robustness.py --dataset PEMS-BAY    # ~25 min
```

After re-running, the JSON will contain:
```json
"STGCN": {
  "mean": 4.6145, "std": 0.0036,     // legacy MAE keys (backward compat)
  "mae":  {"mean": 4.6145, "std": 0.0036, "all": [...]},
  "rmse": {"mean": 8.234,  "std": 0.012,  "all": [...]},
  "mape": {"mean": 12.45,  "std": 0.08,   "all": [...]}
}
```

**Why RMSE matters for robustness:** If DCRNN's autoregressive decoder compounds errors, RMSE should increase disproportionately faster than MAE under corruption — confirming the error-compounding hypothesis with a second independent metric.

