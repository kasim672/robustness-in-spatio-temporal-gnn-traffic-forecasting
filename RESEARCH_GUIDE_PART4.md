# RESEARCH GUIDE — PART 4: VIVA QUESTIONS

## BEGINNER (50 Questions)

**1. What is traffic forecasting?**
Predicting future traffic speed/flow at sensor locations using historical data.

**2. What datasets do you use?**
METR-LA (207 sensors, LA highways, 4 months) and PEMS-BAY (325 sensors, Bay Area, 6 months).

**3. What is the sampling interval?**
5 minutes per timestep.

**4. What does speed=0 mean in your data?**
Usually sensor failure, not actual zero speed. We replace with forward-fill/column mean.

**5. How many models do you compare?**
7: Persistence, Historical Average, ARIMA(2,1,2), Random Forest, LSTM, STGCN, DCRNN.

**6. What is Persistence baseline?**
Copy last observed value as prediction for all future timesteps.

**7. Why use Persistence?**
It's the minimum bar — any useful model must beat "just copy the last value."

**8. What is Z-score normalization?**
z = (x - μ) / σ. Scales data to zero mean, unit variance per sensor.

**9. Why normalize?**
Different sensors have different speed ranges. Without it, high-speed sensors dominate the loss.

**10. Why use training statistics only for normalization?**
Using test data stats would leak future information into the model.

**11. What is data leakage?**
When test set information influences training, inflating reported performance.

**12. Why chronological splits?**
Time-series has causal ordering. Random splits create look-ahead bias.

**13. What are your split ratios?**
70% train, 10% validation, 20% test (chronological).

**14. What is a sliding window?**
Moving through time: X=[t₀..t₁₁], Y=[t₁₂..t₂₃], then shift by 1: X=[t₁..t₁₂], Y=[t₁₃..t₂₄].

**15. Why seq_len=12?**
12 × 5 min = 60 min. One hour captures rush-hour dynamics.

**16. What horizons do you evaluate?**
15 min (step 3), 30 min (step 6), 60 min (step 12).

**17. What is MAE?**
Mean Absolute Error = average |prediction - actual| in mph.

**18. What is RMSE?**
Root Mean Squared Error = √(mean of squared errors). Penalizes large errors more.

**19. What is MAPE?**
Mean Absolute Percentage Error = average |error/actual| × 100%.

**20. What is an adjacency matrix?**
N×N matrix where A[i,j] > 0 means sensors i and j are connected.

**21. What is graph sparsity?**
Percentage of zero entries. Your METR-LA graph is 98.96% sparse.

**22. What is a self-loop?**
A[i,i] = 1. A node connected to itself. Ensures the node uses its own features.

**23. What is early stopping?**
Stop training when validation loss hasn't improved for 15 epochs to prevent overfitting.

**24. What optimizer do you use?**
Adam with lr=0.001 and weight_decay=0.0001.

**25. What is learning rate scheduling?**
ReduceLROnPlateau: if val_loss stalls for 5 epochs, multiply lr by 0.5.

**26. What is gradient clipping?**
If gradient norm exceeds 5.0, rescale it. Prevents exploding gradients.

**27. What is batch size?**
128 samples processed simultaneously per training step.

**28. What GPU did you use?**
NVIDIA RTX 4090 (training), RTX 4060 (validation).

**29. What is AMP?**
Automatic Mixed Precision — uses fp16 for speed, fp32 for precision-critical ops.

**30. Which model has the best accuracy?**
DCRNN (MAE 3.548 on METR-LA, 1.905 on PEMS-BAY).

**31. Which model is most robust?**
STGCN under random missing (27.0% degradation vs DCRNN's 34.5%).

**32. What is robustness in your context?**
How much a model's accuracy degrades when input data is corrupted.

**33. What are your two corruption types?**
Random missing (scattered zeros) and sensor failure (entire sensors go dead).

**34. Why 5 corruption seeds?**
Different random corruptions give different results. 5 seeds → mean ± std for reliability.

**35. What is the degree matrix?**
Diagonal matrix where D[i,i] = sum of row i in adjacency matrix.

**36. What is the graph Laplacian?**
L = D - A. Measures how a node's value differs from its neighbors.

**37. What is an LSTM?**
Long Short-Term Memory — RNN with gates (forget, input, output) that learns long-term dependencies.

**38. Why LSTM over vanilla RNN?**
Vanilla RNNs suffer vanishing gradients. LSTM gates allow gradients to flow through the cell state.

**39. What is ARIMA?**
AutoRegressive Integrated Moving Average — classical time-series model.

**40. What does ARIMA(2,1,2) mean?**
p=2 (2 AR terms), d=1 (difference once), q=2 (2 MA terms).

**41. Why d=1 in your ARIMA?**
Traffic speed is non-stationary. Differencing removes the trend component.

**42. What is Random Forest?**
Ensemble of 100 decision trees, each trained on random subsets of data and features.

**43. What is a GNN?**
Graph Neural Network — processes graph-structured data by aggregating neighbor information.

**44. What is message passing?**
Each node collects features from its neighbors, aggregates them, and updates its own state.

**45. What is STGCN?**
Spatio-Temporal Graph Convolutional Network. Uses Chebyshev graph conv + gated temporal CNN.

**46. What is DCRNN?**
Diffusion Convolutional Recurrent Neural Network. Uses diffusion conv + GRU encoder-decoder.

**47. How many parameters does each GNN have?**
STGCN: ~80K (METR-LA). DCRNN: ~371K (4.7× more).

**48. What is teacher forcing?**
During decoder training, sometimes feed true target instead of model prediction to stabilize learning.

**49. What is an ablation study?**
Remove/change one component to measure its contribution.

**50. What does your identity ablation test?**
Whether GNNs need spatial information or just temporal processing is sufficient.

---

## INTERMEDIATE (50 Questions)

**51. Explain your graph construction pipeline.**
Pearson correlation → Gaussian kernel (σ=0.1) → threshold (ε=0.3) → self-loops → symmetric normalize (STGCN) / random walk normalize (DCRNN).

**52. Why Gaussian kernel instead of raw correlation?**
The kernel applies a sharp cutoff — only highly correlated sensors (>0.9) get meaningful weights. Raw correlation would create a dense, noisy graph.

**53. What is symmetric normalization?**
Â = D^(-½)AD^(-½). Normalizes by degree so high-degree nodes don't dominate aggregation.

**54. What is random walk normalization?**
P = D^(-1)A. Row-stochastic matrix — each row sums to 1. Represents transition probabilities.

**55. Why does STGCN use symmetric and DCRNN uses random walk?**
STGCN is spectral (needs symmetric Laplacian for eigendecomposition). DCRNN models diffusion (needs transition probabilities).

**56. What are Chebyshev polynomials?**
Orthogonal polynomials defined by T₀=1, T₁=x, Tₖ=2xTₖ₋₁-Tₖ₋₂. They approximate spectral graph filters without eigendecomposition.

**57. Why K=3 for Chebyshev?**
K=3 means 3-hop neighborhood. With 2.2 avg connections, this captures local spatial patterns without over-smoothing.

**58. What is the scaled Laplacian?**
L̃ = (2/λ_max)L - I. Scales eigenvalues to [-1,1] so Chebyshev polynomials are valid.

**59. What is diffusion convolution?**
Aggregating neighbor features using random walk matrices: h = Σₖ θₖPᵏX. P^k captures k-hop information.

**60. Why bidirectional diffusion?**
Traffic is directional. Forward (D⁻¹W) models downstream propagation; backward (D⁻¹Wᵀ) models upstream effects.

**61. What is a DCGRU cell?**
Standard GRU with diffusion convolution replacing FC layers. Gates (reset, update) use graph-aware operations.

**62. How does the DCRNN encoder work?**
Processes input sequence step-by-step through 2-layer DCGRU stack. Final hidden states encode the input.

**63. How does the DCRNN decoder work?**
Autoregressively generates predictions: each output feeds as input to produce the next, through 2-layer DCGRU + linear projection.

**64. What is the GLU gating in STGCN?**
Output = P ⊙ σ(Q). The sigmoid gate σ(Q) learns which temporal features to pass through.

**65. Why does STGCN use Conv2d for temporal convolution?**
Data is arranged as (B, C, N, T). Conv2d with kernel (1, k) convolves along T (time) independently for each sensor N.

**66. Explain boundary leakage.**
If sequences span the train/val boundary, a training sample could use validation timesteps as input. We prevent this by creating sequences per-chunk separately.

**67. Why is DCRNN slower than STGCN?**
DCRNN is sequential (GRU processes one timestep at a time). STGCN is parallel (CNN processes all timesteps at once). DCRNN also does 24 graph convolutions per forward pass vs STGCN's 2.

**68. Why does DCRNN training loss increase over epochs?**
Teacher forcing decay. Early epochs use ground truth decoder input (easy). Later epochs use model predictions (harder). The model improves but the task gets harder.

**69. What is the output convolution in STGCN?**
Conv2d(64, 12, kernel=(1,12)) collapses temporal dimension from 12→1 and produces 12 horizon predictions.

**70. Why LayerNorm instead of BatchNorm in STGCN?**
LayerNorm normalizes per-sample (stable across batch sizes). BatchNorm depends on batch statistics which vary, especially with small batches.

**71. How do you prevent graph data leakage?**
Graph is built from train_raw only: `build_graph(data['train_raw'])`. No test correlations are used.

**72. What does 2.2 avg connections mean physically?**
Each METR-LA sensor is strongly correlated with only ~2 other sensors — typically the same highway segment.

**73. Why does STGCN fail on PEMS-BAY?**
PEMS-BAY has 7.6 conn/node. With K=3, each node reaches ~7.6³ ≈ 439 potential nodes — causing over-smoothing where all representations converge.

**74. What is over-smoothing?**
After many rounds of neighborhood aggregation, all node features become similar, losing discriminative local information.

**75. Why is STGCN's RMSE lower than DCRNN's on METR-LA?**
Spectral smoothing prevents wild predictions. DCRNN's autoregressive decoder can occasionally produce large errors.

**76. Explain the train/val/test DataLoader differences.**
Train: shuffle=True (SGD needs random order). Val/Test: shuffle=False (evaluation must be deterministic).

**77. What is pin_memory in DataLoader?**
Allocates CPU tensors in pinned memory for zero-copy GPU transfer, reducing data loading latency.

**78. What is persistent_workers?**
Keeps worker processes alive between epochs, avoiding the overhead of spawning/killing processes each epoch.

**79. What does weight_decay=0.0001 do?**
L2 regularization: adds 0.0001 × ||weights||² to the loss, preventing weights from growing too large.

**80. How does ReduceLROnPlateau work?**
Monitors val_loss. If no improvement for patience=5 epochs, multiplies lr by factor=0.5.

**81. What would happen without gradient clipping?**
DCRNN's sequential decoder can produce exploding gradients, causing NaN losses and training failure.

**82. Why evaluate on de-normalized predictions?**
Normalized MAE is meaningless across datasets. De-normalized MAE is in mph — directly interpretable and comparable to other papers.

**83. What is the Fiedler value?**
λ₂ of the graph Laplacian. Measures algebraic connectivity. λ₂ > 0 means connected graph.

**84. Why does sensor failure hurt GNNs more than random missing?**
Sensor failure is spatially correlated (entire sensors die). GNNs aggregate from neighbors — if neighbors are dead, they inject zeros through message passing.

**85. Why is LSTM most robust under sensor failure?**
LSTM has NO spatial connections. A dead sensor affects only its own prediction, not others.

**86. What is the spectral gap?**
λ₂ - λ₁ = λ₂. Larger gap means faster information diffusion across the graph.

**87. How does your sparsity analysis work?**
Varies ε threshold, measures graph density, connectivity (λ₂), and model performance at each sparsity level.

**88. What is the einsum operation in ChebGraphConv?**
`torch.einsum('mn,bnc->bmc', Tk, x)` is batched matrix multiplication: (N,N) × (B,N,C) → (B,N,C).

**89. Why Xavier initialization for graph conv weights?**
Xavier keeps variance stable across layers, preventing vanishing/exploding activations in deep networks.

**90. What happens if σ is too large in graph construction?**
More sensors become connected → denser graph → potential over-smoothing. σ=0.1 is conservative.

**91. What happens if ε is too high?**
Graph becomes too sparse → sensors become isolated → GNNs lose spatial information (approaches identity graph).

**92. What is the time complexity of Chebyshev graph conv?**
O(K × |E| × C) where K=polynomial order, |E|=edges, C=channels. Linear in edges, not quadratic in nodes.

**93. What is the time complexity of diffusion conv?**
O(K × |E| × C) per direction. With 2 directions and K steps: O(2K × |E| × C).

**94. Why save models as .pt for neural networks and .pkl for sklearn?**
PyTorch models use `torch.save()` → `.pt`. Scikit-learn models use `pickle` → `.pkl`. Different serialization formats.

**95. What is cuDNN benchmark mode?**
Auto-selects the fastest convolution algorithm for the input size. Gives ~10% speedup but sacrifices bit-for-bit reproducibility.

**96. How many diffusion support matrices does DCRNN use?**
4: P_f¹, P_b¹, P_f², P_b² (forward/backward × 2 hops). Plus identity = 5 total in DiffusionConv.

**97. What is the teacher forcing schedule?**
Linear decay: tf_ratio = max(0, 1 - epoch/50). First 50% of training gets help, second half is independent.

**98. Why does the decoder start with zeros?**
No prior information about the future. The decoder must generate the first prediction purely from the encoder's hidden state.

**99. What is the projection layer in DCRNN decoder?**
Linear(hidden_dim=64, out=1). Maps the 64-dim hidden state to 1-dim speed prediction per sensor.

**100. How do you verify your pipeline is correct?**
validate.py loads all checkpoints, re-evaluates metrics, checks MAE < Persistence, verifies horizon ordering (15min < 30min < 60min MAE).

---

## ADVANCED / RESEARCH (50 Questions)

**101. Is 1-2% graph benefit statistically significant?**
With single-run experiments, no. Need multiple training seeds + paired t-tests. The difference is within typical training variance.

**102. Why might random graphs perform as well as learned graphs?**
At ε=0.3, the learned graph has only 2.2 conn/node — barely different from random. The temporal components (CNN/GRU) dominate. Any mixing provides sufficient regularization.

**103. Could attention replace fixed graph topology?**
Yes. Graph Attention Networks (GAT) learn edge weights dynamically. This could adapt to sensor failures by downweighting dead sensors.

**104. How would you make STGCN robust to sensor failure?**
(a) Corruption-aware training (augment with dropout). (b) Attention-based graph conv. (c) Input masking layer that detects and handles zeros.

**105. What is the accuracy-robustness tradeoff?**
DCRNN achieves best accuracy but worst robustness. More expressive models (autoregressive decoder) are more sensitive to input perturbations. This parallels findings in adversarial robustness literature.

**106. Why is degradation higher on PEMS-BAY than METR-LA?**
PEMS-BAY has 7.6 conn/node vs 2.2. More connections = more channels for noise propagation through message passing.

**107. Could you use graph learning instead of correlation-based construction?**
Yes. Methods like Graph WaveNet learn the adjacency matrix end-to-end via learnable node embeddings: A = softmax(E₁ · E₂ᵀ).

**108. What is the relationship between Chebyshev order K and receptive field?**
K=3 means each node aggregates from 3-hop neighbors. Receptive field = {nodes within 3 edges}. On METR-LA (2.2 conn/node), this is ~10 nodes.

**109. How does over-smoothing relate to your PEMS-BAY STGCN result?**
PEMS-BAY's denser graph (7.6 conn) with K=3 causes 3-hop aggregation to reach ~439 nodes. Features become homogeneous, hurting prediction specificity.

**110. What would a Graph Transformer look like for this task?**
Self-attention over all sensor pairs (O(N²)), with positional encoding from graph distance. No fixed topology — learned attention weights serve as dynamic edges.

**111. How does your work compare to Graph WaveNet?**
Graph WaveNet uses adaptive adjacency + dilated causal convolutions. It typically achieves MAE ~2.69 on METR-LA vs your DCRNN's 3.55. But our focus is robustness, not SOTA accuracy.

**112. What are the limitations of your evaluation methodology?**
(a) Single training run (no confidence intervals). (b) Only 2 corruption types. (c) No adversarial attacks. (d) Fixed graph under corruption. (e) Models not retrained on corrupted data.

**113. What is the difference between transductive and inductive GNN settings?**
Transductive: graph fixed at test time (your setting). Inductive: new nodes/graphs at test time. Your models cannot handle new sensors without retraining.

**114. How would you extend this to multi-step multi-feature forecasting?**
Add more input features (flow, occupancy) by changing input channels from 1 to 3. Architecture change: STGCN channels[0]=3, DCRNN encoder in_channels=3.

**115. Why not use temporal attention instead of CNN/GRU?**
Transformers with temporal attention are state-of-the-art (e.g., PDFormer, STAEformer). They capture long-range dependencies better but require more data and compute.

**116. What is curriculum learning and how does teacher forcing relate?**
Curriculum learning trains on easy examples first, then hard ones. Teacher forcing is a form: early training gets "easy" (true input), later training gets "hard" (own predictions).

**117. Could you apply your robustness framework to other GNN tasks?**
Yes. The corruption injection + multi-seed evaluation methodology generalizes to any GNN task (molecular property prediction, social networks, etc.).

**118. What is the difference between node-level and graph-level prediction?**
Node-level: predict a value per node (your task — speed per sensor). Graph-level: predict a single value for the whole graph (e.g., total network congestion).

**119. How does your graph construction relate to distance-based methods?**
Li et al. (2018) used road network distance: A[i,j] = exp(-d²/σ²). You use correlation distance: d = 1 - corr. Both use Gaussian kernels but with different distance metrics.

**120. What is the computational complexity of your full pipeline?**
Data loading: O(T×N). Graph construction: O(N²×T) for correlation. Chebyshev: O(K×N²). Training per epoch: O(B×(K×|E|×C)×T) where B=batches.

**121-130. How Google/Uber use traffic forecasting?**
Google Maps: DeepMind's GNN predicts ETA with graph of road segments. Uber: uses similar GNNs for surge pricing and ETA. Both use real-time streaming data, much larger graphs (millions of segments), and ensemble multiple models.

**131-140. Open research problems:**
- Scalability to city-scale graphs (millions of nodes)
- Online/continual learning (model updates in real-time)
- Multi-modal fusion (traffic + weather + events)
- Fairness (equal prediction quality across neighborhoods)
- Explainability (why does the model predict congestion?)
- Transfer learning across cities
- Handling non-recurrent events (accidents, construction)
- Privacy-preserving federated traffic learning
- 3D traffic (multi-level highways, tunnels)
- Integration with traffic control (prediction → action)

**141-150. Future directions for your work:**
- Add Graph Transformer comparison
- Test adversarial robustness (worst-case corruptions)
- Adaptive graph learning during inference
- Multi-task: predict speed + detect sensor failures
- Cross-city transfer experiments
- Uncertainty quantification (prediction confidence intervals)
- Real-time streaming evaluation
- Larger datasets (PeMS-D7, PeMS-D8)
- Combine STGCN robustness with DCRNN accuracy (ensemble)
- Corruption-aware training (data augmentation with sensor dropout)
