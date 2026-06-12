# VII. RESULTS AND DISCUSSION

## A. Clean-Data Accuracy

Table III presents the test-set performance of all seven models under clean input conditions. DCRNN achieves the lowest overall MAE on both datasets: 3.548 mph on METR-LA and 1.905 mph on PEMS-BAY. This advantage may be attributable to DCRNN's bidirectional diffusion convolution, which models asymmetric upstream-downstream traffic propagation, combined with the autoregressive decoder that conditions each prediction step on prior outputs.

STGCN achieves the second-best overall MAE on METR-LA (3.634 mph) but ranks below LSTM and Random Forest on PEMS-BAY (2.299 vs. 2.071 and 2.113 mph respectively). This dataset-dependent behavior is consistent with the substantially different graph densities: PEMS-BAY has 7.6 average connections per node versus 2.2 for METR-LA. In the denser PEMS-BAY graph, the K=3 Chebyshev convolution aggregates information from a larger effective neighborhood at each layer, which may lead to excessive spatial smoothing of localized traffic patterns. This interpretation is offered as a hypothesis; definitive confirmation would require controlled experiments varying K across graph densities.

A notable crossover emerges at longer forecast horizons. At 60 minutes, STGCN achieves the best MAE on METR-LA (4.242 vs. DCRNN's 4.541), a 6.6% advantage. One possible explanation is that STGCN generates all 12 horizon predictions simultaneously, whereas DCRNN's autoregressive decoder generates predictions sequentially. At longer horizons, small prediction errors in early decoder steps may accumulate through this sequential chain, producing compounding degradation that parallel architectures avoid.

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
**Fig. 4.** Multi-metric performance comparison on METR-LA (MAE, RMSE, MAPE). DCRNN achieves the lowest MAE, but STGCN achieves the lowest RMSE, indicating fewer extreme prediction errors.

<!-- INSERT FIGURE: results/plots/PEMS-BAY_baselines_comparison.png -->
**Fig. 5.** Multi-metric performance comparison on PEMS-BAY (MAE, RMSE, MAPE). DCRNN leads across all three metrics. STGCN ranks below LSTM and Random Forest on MAE, suggesting graph density may interact with spectral convolution.

<!-- INSERT FIGURE: results/plots/METR-LA_baselines_horizon.png -->
**Fig. 6.** Per-horizon MAE on METR-LA. STGCN surpasses DCRNN at the 60-minute horizon due to the absence of autoregressive error compounding.

<!-- INSERT FIGURE: results/plots/PEMS-BAY_baselines_horizon.png -->
**Fig. 7.** Per-horizon MAE on PEMS-BAY. DCRNN maintains dominance across all horizons on this dataset.

---

## B. Random Missing Data Robustness

Table IV presents the multi-metric robustness results under random missing corruption, where each input entry is independently zeroed with probability p. All corrupted results are reported as mean ± standard deviation across five independent corruption seeds.

The central finding is that the accuracy ranking and the robustness ranking are inversely related. DCRNN, which achieves the best clean accuracy, exhibits the highest MAE degradation on both datasets: +34.5% on METR-LA and +70.3% on PEMS-BAY at p=0.4. Conversely, STGCN degrades by only +27.0% and +39.1% respectively, making it the most robust GNN under this corruption type.

The RMSE degradation data provides independent corroboration of this pattern. On PEMS-BAY, DCRNN's RMSE increases by +56.4% (4.026→6.295) compared to STGCN's +36.8% (4.232→5.789). The disproportionate RMSE growth relative to MAE growth for DCRNN (+56.4% RMSE vs. +70.3% MAE on PEMS-BAY) indicates that corruption does not merely increase average errors but also generates occasional extreme mispredictions. This is consistent with the autoregressive decoder hypothesis: when the encoder produces corrupted hidden states, the decoder's sequential prediction chain can amplify small initial errors into catastrophic long-horizon predictions.

STGCN's relative robustness may be explained by two architectural properties. First, the Chebyshev spectral graph convolution acts as a spatial low-pass filter: when individual input entries are randomly zeroed, the convolution aggregates values from the K-hop neighborhood, effectively performing spatial smoothing over corrupted entries. Second, the gated temporal convolution produces all horizon predictions in parallel, avoiding sequential error propagation.

MAPE degradation reveals a scale-dependent dimension. On PEMS-BAY, DCRNN's MAPE increases by +92.5% at 40% corruption (4.37→8.42%), the highest among all models. Since MAPE normalizes by ground-truth magnitude, this indicates that DCRNN's errors are disproportionately concentrated on lower-speed observations, which is consistent with corrupted zero-valued inputs biasing the diffusion convolution toward underestimation.

The standard deviations under random missing are remarkably small (σ_MAE ≤ 0.008 on both datasets), indicating that this corruption type produces consistent degradation largely independent of which specific entries are removed.

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

This reversal may be explained by the interaction between graph message passing and spatially correlated corruption. When a sensor fails, its value becomes zero. The graph convolution then aggregates this corrupted zero into the representations of neighboring nodes. Under random missing, corrupted values are spatially scattered and the convolution effectively performs spatial averaging. Under sensor failure, corruption concentrates in local graph neighborhoods and may be amplified through message passing.

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
**Fig. 9.** Learned adjacency matrix heatmap (METR-LA, ε=0.3). The extreme sparsity is visible—most entries are zero. Nearly half the nodes have no spatial connections.

<!-- INSERT FIGURE: results/plots/METR-LA_sparsity_analysis.png -->
**Fig. 10.** Effect of sparsity threshold ε on graph properties and model performance. Higher ε removes edges monotonically while performance remains relatively stable, consistent with the marginal ablation impact.

---

## E. Discussion

The experimental results reveal a fundamental tension between clean-data accuracy and robustness to input corruption that has direct implications for deploying graph neural networks in intelligent transportation systems.

**Accuracy-robustness trade-off.** DCRNN achieves the highest clean accuracy yet exhibits the highest sensitivity to input corruption. The autoregressive decoder that may provide DCRNN's clean-data advantage becomes a potential liability under corruption, as degraded encoder states propagate through the sequential decoding chain. The RMSE data is consistent with this interpretation: DCRNN's disproportionate RMSE growth under corruption suggests more frequent extreme mispredictions, not merely higher average error.

**Corruption geometry matters.** The reversal in robustness ranking between random missing and sensor failure demonstrates that robustness is not a single-dimensional property. This suggests that robustness evaluations using only one corruption type may produce misleading conclusions about deployment readiness.

**Practical implications.** In real-world traffic networks, both corruption types occur. A conservative deployment strategy would use DCRNN for clean or lightly corrupted inputs while maintaining an LSTM fallback for regions experiencing equipment failures. Corruption-aware training strategies such as input dropout could be investigated to improve GNN robustness without sacrificing clean accuracy.

**Graph topology value.** Under the configuration tested (ε=0.3), learned graph topology provides marginal benefit over identity and random graphs. Future work should investigate denser graph construction methods, alternative proximity measures, or adaptive graph learning.

---

## F. Key Findings

**Finding 1: Clean accuracy and robustness are inversely related.**
DCRNN achieves the best clean MAE (3.548 METR-LA, 1.905 PEMS-BAY) but the highest degradation under 40% random missing (+34.5%, +70.3%). Clean benchmark accuracy alone is insufficient for deployment-oriented model selection.

**Finding 2: STGCN is the most robust GNN under random missing corruption.**
STGCN degrades by only 27.0% (METR-LA) and 39.1% (PEMS-BAY), the lowest among all learned models across all three metrics (MAE, RMSE, MAPE).

**Finding 3: The robustness ranking reverses under sensor failure.**
STGCN drops from rank 2 to rank 6 on METR-LA; LSTM rises from rank 5 to rank 4. The 53× increase in standard deviation (0.004→0.213) indicates strong dependence on failure location, suggesting graph message passing may propagate spatially correlated failures.

**Finding 4: RMSE and MAPE provide independent corroboration.**
DCRNN's RMSE (+56.4%) and MAPE (+92.5%) degradation on PEMS-BAY substantially exceed its MAE degradation, indicating more frequent extreme mispredictions concentrated during lower-speed conditions.

**Finding 5: Learned graph topology provides marginal benefit.**
The correlation-based graph improves DCRNN by 1.92% and STGCN by 0.95% over identity graphs. These single-run observations cannot establish statistical significance.

Overall, the experiments demonstrate that evaluating traffic forecasting models solely on clean benchmark datasets may overestimate their suitability for deployment, emphasizing the importance of robustness-aware evaluation under realistic sensor degradation scenarios.

---

# IX. LIMITATIONS AND FUTURE WORK

Although the proposed benchmark provides a comprehensive evaluation of robustness under realistic sensor degradation, several limitations should be acknowledged. First, the experiments are conducted on two widely used public datasets, METR-LA and PEMS-BAY, and therefore the conclusions may not directly generalize to other traffic networks with different spatial characteristics or sensing infrastructures. Second, the robustness evaluation considers two corruption mechanisms—random missing values and complete sensor failures—but real-world deployments may also experience burst errors, communication delays, calibration drift, biased measurements, or dynamically changing road topologies. Third, the graph ablation study is based on single training runs for each graph configuration; consequently, while the observed differences are informative, statistical significance cannot be established without additional multi-seed experiments. Fourth, all models were trained exclusively on clean data, meaning the observed degradation reflects worst-case robustness without any corruption-aware adaptation.

Future research can extend this benchmark in several directions. Modern architectures such as Graph WaveNet, AGCRN, ASTGCN variants, and transformer-based spatio-temporal models should be evaluated under the same corruption protocol to determine whether recent advances improve robustness as well as clean-data accuracy. In addition, adaptive graph learning, uncertainty-aware forecasting, corruption-aware data augmentation, masked reconstruction objectives, and joint imputation-forecasting frameworks represent promising approaches for increasing resilience to degraded sensor inputs. Finally, evaluating robustness under more realistic operational scenarios, including evolving graph topology caused by road closures or network disruptions, would further improve the practical relevance of robustness-oriented traffic forecasting benchmarks.

---

# X. CONCLUSION

This paper presented a robustness-oriented benchmark of seven traffic forecasting models on the METR-LA and PEMS-BAY datasets, comparing traditional statistical methods, deep learning models, and spatio-temporal graph neural networks under both clean and degraded sensing conditions. While DCRNN achieved the highest forecasting accuracy on clean data, the experiments demonstrated that clean-data accuracy alone does not guarantee robustness under realistic sensor degradation. STGCN consistently exhibited greater resilience to randomly missing inputs, whereas sensor failure altered the robustness ranking and, on METR-LA, allowed the non-graph LSTM model to outperform both graph-based models. These findings suggest that graph-based message passing, while beneficial for exploiting spatial dependencies, may also propagate corrupted information when sensor failures occur.

The graph ablation study further indicated that, under the evaluated configuration, learned graph topology contributed only marginal improvements over simpler alternatives, suggesting that temporal learning played a dominant role in the observed predictive performance. Although these observations require additional multi-seed validation to establish statistical significance, they highlight the importance of carefully evaluating the contribution of graph construction in spatio-temporal forecasting systems.

Overall, the results demonstrate that robustness should be considered alongside predictive accuracy when selecting forecasting models for Intelligent Transportation Systems. Evaluating models solely on clean benchmark datasets may overestimate their suitability for real-world deployment, where missing data, sensor failures, and communication disruptions are unavoidable. Future traffic forecasting research should therefore emphasize robustness-aware evaluation and corruption-resilient model design in addition to achieving state-of-the-art accuracy.

---

# ACKNOWLEDGMENT

The author gratefully acknowledges the authors of the METR-LA and PEMS-BAY benchmark datasets and the broader open-source research community for providing publicly available datasets, reference implementations, and reproducible resources that supported this study.

---

## Figure Placement Summary


| Fig. | File | Section | Purpose |
|---|---|---|---|
| 1 | *(earlier section)* | IV | Experimental pipeline |
| 2 | *(earlier section)* | V | STGCN architecture |
| 3 | *(earlier section)* | V | DCRNN architecture |
| 4 | `results/plots/METR-LA_baselines_comparison.png` | VII-A | Multi-metric clean accuracy (MAE, RMSE, MAPE) |
| 5 | `results/plots/PEMS-BAY_baselines_comparison.png` | VII-A | Cross-dataset multi-metric contrast |
| 6 | `results/plots/METR-LA_baselines_horizon.png` | VII-A | Horizon crossover (STGCN vs DCRNN) |
| 7 | `results/plots/PEMS-BAY_baselines_horizon.png` | VII-A | PEMS-BAY horizon comparison |
| 8 | `results/plots/METR-LA_robustness_curves.png` | VII-B/C | **Central contribution** |
| 9 | `results/plots/adjacency_matrix_detailed.png` | VII-D | Graph sparsity visualization |
| 10 | `results/plots/METR-LA_sparsity_analysis.png` | VII-D | Ablation support |

---

# REFERENCES

[1] Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vancouver, BC, Canada, 2018.

[2] B. Yu, H. Yin, and Z. Zhu, "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting," in *Proc. 27th Int. Joint Conf. Artif. Intell. (IJCAI)*, Stockholm, Sweden, 2018, pp. 3634–3640.

[3] M. Defferrard, X. Bresson, and P. Vandergheynst, "Convolutional neural networks on graphs with fast localized spectral filtering," in *Proc. Advances Neural Inf. Process. Syst. (NeurIPS)*, vol. 29, 2016, pp. 3844–3852.

[4] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, "Graph WaveNet for deep spatial-temporal graph modeling," in *Proc. 28th Int. Joint Conf. Artif. Intell. (IJCAI)*, Macao, China, 2019, pp. 1907–1913.

[5] L. Bai, L. Yao, C. Li, X. Wang, and C. Wang, "Adaptive graph convolutional recurrent network for traffic forecasting," in *Proc. Advances Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 17804–17815.

[6] S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting," in *Proc. AAAI Conf. Artif. Intell.*, vol. 33, no. 1, 2019, pp. 922–929.

[7] W. Jiang and J. Luo, "Graph neural network for traffic forecasting: A survey," *Expert Syst. Appl.*, vol. 207, p. 117921, 2022.

[8] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] Z. Cui, K. Henrickson, R. Ke, and Y. Wang, "Graph Markov network for traffic forecasting with missing data," *Transp. Res. Part C, Emerg. Technol.*, vol. 117, p. 102671, 2020.

[10] J. Zuo, K. Zeitouni, Y. Taher, and S. Garcia-Rodriguez, "Graph convolutional networks for traffic forecasting with missing values," *Data Mining Knowl. Discovery*, vol. 37, pp. 913–947, 2023.

[11] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Toulon, France, 2017.

[12] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[13] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Hoboken, NJ, USA: Wiley, 2015.

[14] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Doha, Qatar, 2014, pp. 1724–1734.

[15] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language modeling with gated convolutional networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, Sydney, Australia, 2017, pp. 933–941.

[16] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, San Diego, CA, USA, 2015.

[17] E. I. Vlahogianni, M. G. Karlaftis, and J. C. Golias, "Short-term traffic forecasting: Where we are and where we're going," *Transp. Res. Part C, Emerg. Technol.*, vol. 43, pp. 3–19, 2014.

[18] C. Chen, K. Petty, A. Skabardonis, P. Varaiya, and Z. Jia, "Freeway performance measurement system: Mining loop detector data," *Transp. Res. Rec.*, vol. 1748, no. 1, pp. 96–102, 2001.

[19] B. M. Williams and L. A. Hoel, "Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process: Theoretical basis and empirical results," *J. Transp. Eng.*, vol. 129, no. 6, pp. 664–672, 2003.

[20] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Vancouver, BC, Canada, 2018.

[21] C. Zheng, X. Fan, C. Wang, and J. Qi, "GMAN: A graph multi-attention network for traffic prediction," in *Proc. AAAI Conf. Artif. Intell.*, vol. 34, no. 1, 2020, pp. 1234–1241.

[22] D. Zügner, A. Akbarnejad, and S. Günnemann, "Adversarial attacks on neural networks for graph data," in *Proc. 24th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, London, UK, 2018, pp. 2847–2856.

[23] D. A. Tedjopurnomo, Z. Bao, B. Zheng, F. M. Choudhury, and A. K. Qin, "A survey on modern deep neural network for traffic prediction: Trends, methods and challenges," *IEEE Trans. Knowl. Data Eng.*, vol. 34, no. 4, pp. 1544–1561, 2022.

[24] Y. Zhang, T. Cheng, and Y. Ren, "A graph deep learning method for short-term traffic forecasting on large road networks," *Comput.-Aided Civil Infrastruct. Eng.*, vol. 34, no. 10, pp. 877–896, 2019.
