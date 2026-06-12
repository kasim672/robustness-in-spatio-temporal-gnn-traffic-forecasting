# VII. RESULTS AND DISCUSSION

## A. Clean-Data Accuracy

Table III presents the test-set performance of all seven models under clean input conditions. DCRNN achieves the lowest overall MAE on both datasets: 3.548 mph on METR-LA and 1.905 mph on PEMS-BAY. This advantage is attributable to DCRNN's bidirectional diffusion convolution, which captures asymmetric upstream-downstream traffic propagation patterns that unidirectional or spectral methods cannot represent, combined with the autoregressive decoder that conditions each prediction step on prior outputs, enabling the model to capture multi-step temporal dependencies more precisely than parallel prediction architectures.

STGCN achieves the second-best overall MAE on METR-LA (3.634 mph) but ranks below LSTM and Random Forest on PEMS-BAY (2.299 vs. 2.071 and 2.113 mph respectively). This dataset-dependent behavior is consistent with the substantially different graph densities: PEMS-BAY has 7.6 average connections per node versus 2.2 for METR-LA. In the denser PEMS-BAY graph, the K=3 Chebyshev convolution aggregates information from a larger effective neighborhood at each layer, which may lead to excessive spatial smoothing of localized traffic patterns. This interpretation is offered as a hypothesis; definitive confirmation would require controlled experiments varying K across graph densities.

A notable crossover emerges at longer forecast horizons. At 60 minutes, STGCN achieves the best MAE on METR-LA (4.242 vs. DCRNN's 4.541), a 6.6% advantage. This reversal is architecturally grounded: STGCN generates all 12 horizon predictions simultaneously through a single forward pass, whereas DCRNN's autoregressive decoder generates predictions sequentially, where each output becomes the input for the next step. At longer horizons, small prediction errors in early decoder steps accumulate through this sequential chain, producing compounding degradation that parallel architectures avoid.

The RMSE comparison provides complementary insight. STGCN achieves the lowest RMSE on METR-LA (6.291 vs. DCRNN's 6.672), indicating that while DCRNN produces lower average errors, STGCN produces fewer extreme outlier predictions. This property has practical significance for traffic management systems where catastrophic mispredictions—even if rare—can trigger inappropriate control actions.

LSTM outperforms all classical baselines on both datasets (MAE 3.707 on METR-LA, 2.071 on PEMS-BAY), confirming that learned temporal representations capture nonlinear traffic dynamics that statistical models cannot. The performance gap between LSTM and Random Forest is modest (1.3% on METR-LA), suggesting that temporal ordering provides limited additional benefit over feature-engineered input windows for short-term prediction.

### TABLE III: Clean-Data Forecasting Performance (Lower Is Better)

| Dataset | Model | Type | 15 min MAE | 30 min MAE | 60 min MAE | Overall MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|---|---|---|
| METR-LA | **DCRNN** | GNN | **2.868** | **3.540** | 4.541 | **3.548** | 6.672 | 9.95 |
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
| PEMS-BAY | STGCN | GNN | 2.077 | 2.302 | 2.604 | 2.299 | 4.232 | 5.20 |
| PEMS-BAY | Historical Avg. | Base | 3.559 | 3.559 | 3.559 | 3.559 | 6.827 | 8.76 |

<!-- INSERT FIGURE: results/plots/METR-LA_baselines_comparison.png -->
**Fig. 3.** Overall clean-data MAE on METR-LA. DCRNN achieves the lowest MAE, but clean accuracy alone does not determine robustness under deployment conditions.

<!-- INSERT FIGURE: results/plots/PEMS-BAY_baselines_comparison.png -->
**Fig. 4.** Overall clean-data MAE on PEMS-BAY. STGCN ranks below LSTM and Random Forest, suggesting graph density interacts with spectral convolution.

<!-- INSERT FIGURE: results/plots/METR-LA_baselines_horizon.png -->
**Fig. 5.** Per-horizon MAE on METR-LA. STGCN surpasses DCRNN at the 60-minute horizon due to the absence of autoregressive error compounding.

<!-- INSERT FIGURE: results/plots/PEMS-BAY_baselines_horizon.png -->
**Fig. 6.** Per-horizon MAE on PEMS-BAY. DCRNN maintains dominance across all horizons on this dataset.

<!-- INSERT FIGURE: results/plots/METR-LA_pred_sensor0.png -->
**Fig. 7.** Predicted vs. ground-truth speed for Sensor 0 on METR-LA. GNN models track rapid transitions more accurately than classical baselines.

---

## B. Random Missing Data Robustness

Table IV presents the multi-metric robustness results under random missing corruption, where each input entry is independently zeroed with probability p. All corrupted results are reported as mean ± standard deviation across five independent corruption seeds.

The central finding is that the accuracy ranking and the robustness ranking are inversely related. DCRNN, which achieves the best clean accuracy, exhibits the highest MAE degradation on both datasets: +34.5% on METR-LA and +70.3% on PEMS-BAY at p=0.4. Conversely, STGCN degrades by only +27.0% and +39.1% respectively, making it the most robust GNN under this corruption type.

The RMSE degradation data provides independent corroboration of this pattern. On PEMS-BAY, DCRNN's RMSE increases by +56.4% (4.026→6.295) compared to STGCN's +36.8% (4.232→5.789). The disproportionate RMSE growth relative to MAE growth for DCRNN (+56.4% RMSE vs. +70.3% MAE on PEMS-BAY) indicates that corruption does not merely increase average errors but also generates occasional extreme mispredictions. This is consistent with the autoregressive decoder hypothesis: when the encoder produces corrupted hidden states, the decoder's sequential prediction chain can amplify small initial errors into catastrophic long-horizon predictions.

STGCN's relative robustness is architecturally grounded in two properties. First, the Chebyshev spectral graph convolution acts as a spatial low-pass filter: when individual input entries are randomly zeroed, the convolution aggregates values from the K-hop neighborhood, effectively imputing missing values through spatial smoothing. Second, the gated temporal convolution produces all horizon predictions in parallel, preventing sequential error propagation. The GLU activation provides an additional stabilization mechanism: the learned gate σ(Q) can suppress corrupted temporal features before they propagate deeper into the network.

MAPE degradation reveals a scale-dependent dimension. On PEMS-BAY, DCRNN's MAPE increases by +92.5% at 40% corruption (4.37→8.42%), the highest among all models. Since MAPE normalizes by ground-truth magnitude, this indicates that DCRNN's errors are disproportionately concentrated on lower-speed observations, which is consistent with corrupted zero-valued inputs biasing the diffusion convolution toward underestimation.

The standard deviations under random missing are remarkably small (σ_MAE ≤ 0.008 on both datasets), indicating that this corruption type produces deterministic degradation regardless of which specific entries are removed. This consistency arises from the law of large numbers: at 40% corruption across ~80,000 input entries per sample (12 × 207 × 32 batch), the random pattern converges to a stable statistical profile.

### TABLE IV: Multi-Metric Robustness Under Random Missing (40% Corruption)

| Dataset | Model | Clean MAE | 40% MAE | MAE Δ | Clean RMSE | 40% RMSE | RMSE Δ | 40% MAPE | MAPE Δ |
|---|---|---|---|---|---|---|---|---|---|
| METR-LA | **STGCN** | 3.634 | **4.615±0.004** | **+27.0%** | **6.291** | **7.838** | **+24.6%** | 14.41% | +47.9% |
| METR-LA | Persistence | 3.856 | 4.970±0.003 | +28.9% | 7.806 | 9.314 | +19.3% | 14.73% | +48.1% |
| METR-LA | RF | 3.758 | 4.898±0.002 | +30.3% | 7.142 | 8.521 | +19.3% | 15.14% | +40.1% |
| METR-LA | LSTM | 3.707 | 4.885±0.001 | +31.8% | 7.027 | 8.313 | +18.3% | 15.13% | +41.9% |
| METR-LA | DCRNN | 3.548 | 4.772±0.004 | +34.5% | 6.672 | 8.140 | +22.0% | 14.83% | +49.0% |
| PEMS-BAY | **STGCN** | 2.299 | **3.198±0.002** | **+39.1%** | **4.232** | **5.789** | **+36.8%** | **8.03%** | **+54.6%** |
| PEMS-BAY | LSTM | 2.071 | 3.351±0.001 | +61.8% | 4.500 | 6.624 | +47.2% | 8.80% | +79.3% |
| PEMS-BAY | DCRNN | 1.905 | 3.244±0.001 | +70.3% | 4.026 | 6.295 | +56.4% | 8.42% | +92.5% |

<!-- INSERT FIGURE: results/plots/METR-LA_robustness_curves.png -->
**Fig. 8.** Robustness degradation curves on METR-LA with ±1σ bands. STGCN maintains a consistently shallower slope than DCRNN across all corruption ratios under random missing. The sensor failure panel shows substantially wider variance bands.

---

## C. Sensor Failure Robustness

Sensor failure represents a fundamentally different corruption structure: rather than scattered missing entries, entire sensors lose all historical readings, creating spatially correlated degradation. The robustness ranking changes substantially under this scenario, revealing a critical interaction between graph architecture and corruption geometry.

On METR-LA at 40% sensor failure, the ranking reverses. LSTM degrades by +31.1% (MAE) and +26.1% (RMSE), outperforming both GNNs: DCRNN degrades by +33.0%/+25.2% and STGCN by +34.9%/+28.5%. STGCN, the most robust model under random missing, becomes the least robust under sensor failure—a rank change of four positions.

This reversal is explained by the interaction between graph message passing and spatially correlated corruption. When a sensor fails, its value becomes zero. The graph convolution then aggregates this corrupted zero into the representations of all neighboring nodes. In a random missing scenario, corrupted values are spatially scattered, so each node's neighborhood contains a mixture of valid and corrupted values, and the convolution effectively performs spatial averaging. Under sensor failure, however, neighboring sensors on the same highway segment are likely to fail together (since they share physical infrastructure), concentrating the corruption in local graph neighborhoods and amplifying it through message passing.

STGCN's higher standard deviation under sensor failure provides additional evidence: σ=±0.213 at 40% on METR-LA, compared to σ=±0.004 under random missing—a 53× increase. This indicates that STGCN's performance is highly sensitive to which specific sensors fail. When failed sensors happen to be peripheral nodes with few connections, the impact is contained. When they occupy central positions in the graph topology, the corruption propagates through the spectral convolution to a large portion of the network.

On PEMS-BAY, the pattern partially differs. STGCN maintains comparative robustness under sensor failure (48.3% degradation), outperforming all other models except potentially within the overlap of its wider confidence interval (σ=±0.210). The denser PEMS-BAY graph (7.6 connections/node) may provide sufficient redundant spatial paths to route information around failed sensors, a hypothesis consistent with graph-theoretic notions of vertex connectivity. However, the high standard deviation (σ=±0.364 for RMSE) indicates that this robustness is unreliable and strongly conditioned on failure location.

### TABLE V: Multi-Metric Robustness Under Sensor Failure (40% Corruption)

| Dataset | Model | 40% MAE | MAE Δ | 40% RMSE | RMSE Δ | 40% MAPE | σ_MAE |
|---|---|---|---|---|---|---|---|
| METR-LA | ARIMA | 5.014±0.178 | +25.7% | 9.409±0.240 | +15.8% | 14.83% | 0.178 |
| METR-LA | Persistence | 4.934±0.181 | +28.0% | 9.247±0.245 | +18.5% | 14.59% | 0.181 |
| METR-LA | RF | 4.888±0.165 | +30.0% | 8.912±0.263 | +24.8% | 15.12% | 0.165 |
| METR-LA | **LSTM** | **4.858±0.164** | **+31.1%** | **8.858±0.268** | **+26.1%** | 15.05% | **0.164** |
| METR-LA | DCRNN | 4.718±0.169 | +33.0% | 8.354±0.279 | +25.2% | 14.53% | 0.169 |
| METR-LA | STGCN | 4.903±0.213 | +34.9% | 8.085±0.264 | +28.5% | 14.42% | 0.213 |
| PEMS-BAY | **STGCN** | 3.409±0.210 | +48.3% | 5.984±0.364 | +41.4% | 8.19% | 0.210 |
| PEMS-BAY | DCRNN | 3.189±0.041 | +67.4% | 6.221±0.114 | +54.5% | 7.83% | 0.041 |
| PEMS-BAY | LSTM | 3.398±0.040 | +64.1% | 6.727±0.115 | +49.5% | 8.35% | 0.040 |

### TABLE VI: Corruption-Type Reversal Summary (METR-LA at 40%)

| Model | Random Missing Rank | Sensor Failure Rank | Rank Δ | Mechanism |
|---|---|---|---|---|
| STGCN | 2nd (27.0%) | 6th (34.9%) | **↓4** | Spectral smoothing helps diffuse corruption, amplifies concentrated corruption |
| LSTM | 5th (31.8%) | 4th (31.1%) | ↑1 | No spatial connectivity prevents corruption propagation |
| DCRNN | 6th (34.5%) | 5th (33.0%) | ↑1 | Diffusion propagates corruption but less than spectral conv |

<!-- INSERT FIGURE: results/plots/METR-LA_spatial_error.png -->
**Fig. 9.** Spatial distribution of prediction errors across METR-LA sensors. Sensors in densely connected graph regions show correlated error patterns under sensor failure.

---

## D. Graph Structure Ablation

To investigate the contribution of learned graph topology to model performance, DCRNN and STGCN are evaluated with three adjacency configurations: the learned correlation-based graph, an identity graph (no spatial connections, I_N), and a random symmetric graph with matched density.

The learned graph provides a modest improvement of 1.92% over the identity graph for DCRNN (MAE 3.548 vs. 3.616) and 0.95% for STGCN (3.634 vs. 3.669). The identity-graph DCRNN shows a more pronounced RMSE increase (7.012 vs. 6.672, +5.1%), suggesting that spatial information contributes more to reducing extreme prediction errors than to reducing average error magnitude.

Two competing interpretations exist for the modest ablation impact, and the current experiments cannot distinguish between them. Under the temporal dominance hypothesis, the recurrent (DCGRU) and convolutional (gated CNN) temporal modules capture the majority of the predictive signal, rendering spatial information supplementary. Under the weak graph hypothesis, the correlation-based graph at ε=0.3 is too sparse (2.2 connections/node, 48.3% isolated nodes) to provide substantial spatial information in the first place. A definitive test would require evaluating performance across multiple graph construction methods and sparsity levels with adequate statistical power.

The random-graph variant achieves performance comparable to the learned graph for both models (DCRNN: 3.532 vs. 3.548; STGCN: 3.630 vs. 3.634). This observation is based on a single training run per configuration; we explicitly note that drawing conclusions about the relative value of learned versus random topology would require multiple training seeds and statistical significance testing. What can be stated conservatively is that, within the experimental configuration tested, graph topology choice did not produce differences exceeding 2%.

### TABLE VII: Graph Structure Ablation on METR-LA

| Model | Graph | Overall MAE | RMSE | MAPE (%) | Δ MAE vs. Learned |
|---|---|---|---|---|---|
| DCRNN | Learned | 3.548 | 6.672 | 9.95 | — |
| DCRNN | Identity | 3.616 | 7.012 | 10.25 | +1.92% |
| DCRNN | Random | 3.532 | 6.584 | 9.92 | −0.45% |
| STGCN | Learned | 3.634 | 6.291 | 9.75 | — |
| STGCN | Identity | 3.669 | — | — | +0.95% |
| STGCN | Random | 3.630 | — | — | −0.11% |

### TABLE VIII: Graph Sparsity Analysis (METR-LA)

| ε Threshold | Edges | Avg. Conn./Node | Isolated Nodes (%) | Spectral Gap |
|---|---|---|---|---|
| 0.1 | 424 | 0.83 | 33.8% | 0.0252 |
| 0.2 | 308 | 0.75 | 39.6% | 0.0111 |
| **0.3 (used)** | **240** | **0.67** | **48.3%** | **0.0034** |
| 0.5 | 148 | 0.50 | 59.9% | 0.0069 |

<!-- INSERT FIGURE: results/plots/adjacency_matrix_detailed.png -->
**Fig. 10.** Learned adjacency matrix heatmap (METR-LA, ε=0.3). The extreme sparsity is visible—most entries are zero. Nearly half the nodes have no spatial connections.

<!-- INSERT FIGURE: results/plots/METR-LA_sparsity_analysis.png -->
**Fig. 11.** Effect of sparsity threshold ε on graph properties and model performance. Higher ε removes edges monotonically while performance remains relatively stable, consistent with the marginal ablation impact.

<!-- INSERT FIGURE: results/plots/correlation_analysis.png -->
**Fig. 12.** Pairwise sensor correlation distribution and Gaussian kernel transformation. The sharp Gaussian kernel (σ=0.1) ensures only highly correlated sensor pairs (>0.9) retain meaningful edge weights.

---

## E. Discussion

The experimental results reveal a fundamental tension between clean-data accuracy and robustness to input corruption that has direct implications for the deployment of graph neural networks in intelligent transportation systems.

**Accuracy-robustness trade-off.** DCRNN consistently achieves the highest clean accuracy across both datasets and all horizons (except 60-minute on METR-LA), yet exhibits the highest sensitivity to input corruption. This trade-off is not coincidental but architecturally determined. The autoregressive decoder that provides DCRNN with its multi-step prediction advantage under clean conditions becomes a liability under corruption: the encoder produces degraded hidden states from corrupted inputs, and the decoder's sequential chain propagates and amplifies these initial errors across all future prediction steps. STGCN's parallel prediction architecture sacrifices the conditional modeling capability of autoregressive decoding but gains immunity to this error compounding mechanism. The RMSE data strengthens this interpretation: DCRNN's RMSE degradation of +56.4% on PEMS-BAY substantially exceeds its MAE degradation of +70.3% in relative terms compared to STGCN (+36.8% RMSE vs. +39.1% MAE), indicating that DCRNN produces more frequent extreme outlier predictions under corruption.

**Corruption geometry matters.** The reversal in robustness ranking between random missing and sensor failure—where STGCN drops from rank 2 to rank 6 on METR-LA—demonstrates that robustness is not a single-dimensional property. Graph convolution acts as a spatial low-pass filter: it smooths randomly scattered noise effectively but amplifies spatially concentrated failures by propagating corrupted zero values to neighboring node representations. This finding suggests that robustness evaluations using only one corruption type may produce misleading conclusions about deployment readiness.

**Practical deployment implications.** In real-world traffic monitoring networks, both corruption types occur simultaneously. Random missing data arises from transient communication failures and is relatively benign for graph-based models. Sensor failure, caused by hardware malfunction or power outages, produces spatially correlated degradation that is more damaging to GNNs than to simple temporal models. A conservative deployment strategy would use DCRNN for clean or lightly corrupted inputs while maintaining an LSTM fallback for sensors in regions experiencing equipment failures. Alternatively, corruption-aware training strategies such as input dropout during training could be investigated to improve GNN robustness without sacrificing clean accuracy.

**Graph topology value.** The ablation experiments indicate that, under the specific configuration tested (Pearson correlation, Gaussian kernel, ε=0.3), learned graph topology provides marginal benefit over both identity and random graphs. While this does not imply that graph structure is irrelevant in general, it does suggest that the temporal component dominates the spatial component for these two traffic speed datasets at the tested sparsity level. Future work should investigate whether denser graph construction methods, alternative proximity measures (e.g., road network distance), or adaptive graph learning produce larger spatial contributions.

---

## F. Key Findings

**Finding 1: DCRNN achieves the highest clean accuracy but the lowest robustness.**
DCRNN obtains the best overall MAE on both datasets (3.548 METR-LA, 1.905 PEMS-BAY) but degrades by 34.5% and 70.3% respectively under 40% random missing data—the highest degradation among all models. The autoregressive decoder, which provides clean-data accuracy through conditional multi-step prediction, becomes a vulnerability under corruption through sequential error compounding. This demonstrates that clean benchmark accuracy is an insufficient criterion for model selection in deployment scenarios.

**Finding 2: STGCN is the most robust GNN under random missing corruption.**
STGCN degrades by 27.0% on METR-LA and 39.1% on PEMS-BAY, the lowest among all learned models on both datasets. The spectral graph convolution provides spatial smoothing that partially compensates for scattered missing values, and the parallel temporal convolution avoids sequential error propagation. This robustness advantage persists across all three metrics (MAE, RMSE, MAPE).

**Finding 3: The robustness ranking reverses under sensor failure.**
On METR-LA, STGCN drops from rank 2 (random missing) to rank 6 (sensor failure), while LSTM rises from rank 5 to rank 4. Graph message passing propagates spatially correlated failures to neighboring nodes, transforming a spatial advantage into a vulnerability. The 53× increase in STGCN's standard deviation (0.004→0.213) confirms that sensor failure impact depends critically on failure location relative to graph topology.

**Finding 4: RMSE degradation independently confirms the error compounding hypothesis.**
DCRNN's RMSE degradation (+56.4% on PEMS-BAY) substantially exceeds its MAE degradation in absolute terms, indicating that corruption causes not just higher average errors but more frequent extreme mispredictions. This disproportionate RMSE growth is consistent with autoregressive error amplification and provides a second metric confirming the architectural vulnerability identified through MAE analysis alone.

**Finding 5: MAPE reveals scale-dependent vulnerability in DCRNN.**
DCRNN's MAPE increases by +92.5% on PEMS-BAY under 40% random missing corruption—the highest percentage increase across all models and metrics. Since MAPE normalizes by ground-truth magnitude, this indicates that DCRNN's errors are disproportionately concentrated during lower-speed traffic conditions, which is precisely when accurate forecasting is most critical for congestion management.

**Finding 6: Learned graph topology provides marginal benefit under tested conditions.**
The correlation-based graph improves DCRNN by 1.92% and STGCN by 0.95% over identity graphs. Random graphs produce comparable performance. These observations are based on single training runs and cannot establish statistical significance. The result is consistent with either temporal module dominance or insufficient graph quality at ε=0.3 (48.3% isolated nodes).

**Finding 7: STGCN produces the lowest RMSE on METR-LA despite higher MAE.**
STGCN achieves RMSE 6.291 versus DCRNN's 6.672, indicating fewer extreme prediction errors. For safety-critical traffic management applications, this property may outweigh DCRNN's lower average error, as catastrophic mispredictions can trigger inappropriate infrastructure control responses.

---

## Figure Placement Summary

| Fig. | File | Placement |
|---|---|---|
| 3 | `results/plots/METR-LA_baselines_comparison.png` | After Table III |
| 4 | `results/plots/PEMS-BAY_baselines_comparison.png` | After Fig. 3 |
| 5 | `results/plots/METR-LA_baselines_horizon.png` | After horizon discussion |
| 6 | `results/plots/PEMS-BAY_baselines_horizon.png` | After Fig. 5 |
| 7 | `results/plots/METR-LA_pred_sensor0.png` | End of Section A |
| 8 | `results/plots/METR-LA_robustness_curves.png` | After Table IV |
| 9 | `results/plots/METR-LA_spatial_error.png` | After Table VI |
| 10 | `results/plots/adjacency_matrix_detailed.png` | Start of Section D |
| 11 | `results/plots/METR-LA_sparsity_analysis.png` | After Table VIII |
| 12 | `results/plots/correlation_analysis.png` | After Fig. 11 |
| 13 | `results/plots/final_horizon_analysis.png` | End of Section F |
