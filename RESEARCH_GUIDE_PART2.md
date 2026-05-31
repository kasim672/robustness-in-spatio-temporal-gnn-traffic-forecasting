












# COMPREHENSIVE RESEARCH GUIDE — PART 2
# Graph Construction → STGCN → DCRNN → Chebyshev → Diffusion

---

# PART 8 — GRAPH CONSTRUCTION IN YOUR PROJECT

## Step-by-Step Process

```python
# graph_builder.py: build_graph() — called with TRAINING data only
adj = compute_correlation_adj(train_raw, sigma=0.1, epsilon=0.3)
```

### Step 1: Pearson Correlation

```
corr(i,j) = Σ(xᵢ - μᵢ)(xⱼ - μⱼ) / √(Σ(xᵢ - μᵢ)² · Σ(xⱼ - μⱼ)²)

Range: [-1, +1]
+1 = perfectly correlated (same highway segment)
 0 = uncorrelated (distant sensors)
-1 = inversely correlated (rare in traffic)
```

### Step 2: Gaussian Kernel

```
distance(i,j) = 1 - corr(i,j)     # Range [0, 2]
A[i,j] = exp(-distance² / (2σ²))  # σ = 0.1

Example: corr = 0.95 → dist = 0.05 → A = exp(-0.0025/0.02) = exp(-0.125) = 0.88
Example: corr = 0.50 → dist = 0.50 → A = exp(-0.25/0.02) = exp(-12.5) ≈ 0.00
```

**Why Gaussian kernel?** It converts correlation to similarity with a sharp cutoff. σ=0.1 means only sensors with corr > ~0.9 get meaningful edge weights.

### Step 3: Threshold (ε = 0.3)

```python
adj[adj < 0.3] = 0.0  # Remove weak connections
np.fill_diagonal(adj, 1.0)  # Add self-loops
```

### Step 4: Normalization

**For STGCN:** Symmetric normalization → Â = D^(-½)AD^(-½)
**For DCRNN:** Random walk normalization → P = D^(-1)A

---

# PART 9 — STGCN (Spatio-Temporal Graph Convolutional Network)

## Paper: Yu et al., "Spatio-Temporal Graph Convolutional Networks," IJCAI 2018

## Architecture Overview

```
Input (batch, 12, 207)
  │
  ▼ reshape to (batch, 1, 207, 12)  — (B, C_in, N, T)
  │
  ▼ ┌──────── ST-Conv Block 1 ────────┐
    │ Temporal Conv (1→16 channels)    │  GLU gating
    │ Graph Conv (Chebyshev K=3)       │  Spectral filtering
    │ Temporal Conv (16→32 channels)   │  GLU gating
    │ LayerNorm + Dropout              │
    └──────────────────────────────────┘
  │  output: (batch, 32, 207, 12)
  ▼
    ┌──────── ST-Conv Block 2 ────────┐
    │ Temporal Conv (32→32 channels)   │
    │ Graph Conv (Chebyshev K=3)       │
    │ Temporal Conv (32→64 channels)   │
    │ LayerNorm + Dropout              │
    └──────────────────────────────────┘
  │  output: (batch, 64, 207, 12)
  ▼
  Output Conv (64→12, kernel=12)  — collapses temporal dim
  │  output: (batch, 12, 207, 1)
  ▼
  Linear (207→207) — per-sensor projection
  │
  ▼
Output (batch, 12, 207)  — predictions for all 12 horizons, all 207 sensors
```

## Temporal Convolution (Gated Linear Unit)

```python
# stgcn.py: TemporalConv (lines 53-74)
# Input: (batch, C_in, N, T)
# Conv2d produces 2×C_out channels
# Split into P and Q
# Output = P ⊙ σ(Q)   ← GLU (Gated Linear Unit)
```

**GLU intuition:** The gate σ(Q) learns WHICH temporal features to pass through. It's like an attention mechanism over time — the network learns to focus on relevant time periods.

**Formula:**
```
h = Conv1D(x)  →  split into [P, Q]
output = P ⊙ sigmoid(Q)

sigmoid(Q) ∈ (0,1) acts as a soft gate:
  ≈ 1 → let this temporal feature through
  ≈ 0 → block this temporal feature
```

## Chebyshev Graph Convolution

**The key insight:** Computing exact spectral graph convolution requires eigendecomposition of L (O(N³) cost). Chebyshev polynomials approximate this with O(K·|E|) cost — polynomial in edges, not cubic in nodes.

**Chebyshev recurrence:**
```
T₀(x) = 1        (identity)
T₁(x) = x        (the input itself)
T₂(x) = 2x·T₁(x) - T₀(x)
Tₖ(x) = 2x·Tₖ₋₁(x) - Tₖ₋₂(x)

Applied to graph Laplacian L̃ (scaled to [-1,1]):
T₀(L̃) = I            (each node sees itself)
T₁(L̃) = L̃            (each node sees 1-hop neighbors)
T₂(L̃) = 2L̃² - I      (each node sees 2-hop neighbors)
T₃(L̃) = 2L̃·T₂ - T₁   (each node sees 3-hop neighbors)
```

**K=3 in your project means:** Each sensor aggregates information from neighbors up to 3 hops away.

```python
# graph_builder.py: compute_chebyshev_polynomials (lines 93-130)
L = I - adj_normalized           # Graph Laplacian
L_tilde = (2/λ_max) * L - I     # Scale to [-1, 1]

cheb_polys = [I, L_tilde]       # T₀, T₁
for k in range(2, K+1):
    cheb_polys.append(2 * L_tilde @ cheb_polys[-1] - cheb_polys[-2])
```

```python
# stgcn.py: ChebGraphConv.forward (lines 29-50)
# For each polynomial order k:
#   Tₖ_x = Tₖ(L̃) @ x          # Apply k-th polynomial to input
# Stack all orders, multiply by learned weights, sum:
#   output = Σₖ Tₖ_x @ Wₖ + bias
```

**Numerical example (4 sensors, K=2):**
```
Input signal f = [60, 55, 45, 30]  (speeds at 4 sensors)

T₀(L̃)·f = I·f = [60, 55, 45, 30]           # Self only
T₁(L̃)·f = L̃·f = [-5.2, 2.1, 8.3, -5.2]    # 1-hop neighbor diff
T₂(L̃)·f = (2L̃²-I)·f = [3.1, -1.5, -4.2, 2.6]  # 2-hop info

Output = W₀·T₀f + W₁·T₁f + W₂·T₂f + bias
```

## Shape Trace Through STGCN

```
Input:          (128, 12, 207)          # batch=128, time=12, sensors=207
Reshape:        (128, 1, 207, 12)       # add channel dim, swap N and T

ST-Block 1:
  TempConv1:    (128, 16, 207, 12)      # 1→16 channels
  GraphConv:    reshape to (128×12, 207, 16) → Chebyshev → reshape back
                (128, 16, 207, 12)      # spatial mixing
  TempConv2:    (128, 32, 207, 12)      # 16→32 channels

ST-Block 2:
  TempConv1:    (128, 32, 207, 12)      # 32→32 channels
  GraphConv:    (128, 32, 207, 12)      # spatial mixing
  TempConv2:    (128, 64, 207, 12)      # 32→64 channels

Output Conv:    (128, 12, 207, 1)       # 64 channels → 12 horizons, T collapses
Squeeze:        (128, 12, 207)          # remove last dim
Linear:         (128, 12, 207)          # per-sensor projection

Parameters: 79,532 (METR-LA) / 142,426 (PEMS-BAY)
```

---

# PART 10 — DCRNN (Diffusion Convolutional Recurrent Neural Network)

## Paper: Li et al., "Diffusion Convolutional Recurrent Neural Network," ICLR 2018

## Core Idea: Traffic as Diffusion

Traffic flow is like a random walk on the graph. Congestion at node i "diffuses" to neighboring nodes over time — like ink spreading in water, but constrained to the road network.

**Forward diffusion:** Information flows in the direction of traffic (downstream)
**Backward diffusion:** Information flows against traffic (upstream effects)

```
Traffic direction →

Sensor A ──→ Sensor B ──→ Sensor C
  (jam)        (will jam)    (will jam later)

Forward:  A's jam propagates to B, then C
Backward: C can "see" that A is jammed (upstream awareness)
```

## Diffusion Convolution Mathematics

**Random walk matrix (forward):**
```
P_f = D_out^(-1) · W    (row-normalized adjacency)

P_f[i,j] = probability of walking from node i to node j in one step
```

**Multi-hop diffusion:**
```
P_f^1 = P_f          (1-hop: direct neighbors)
P_f^2 = P_f · P_f    (2-hop: neighbors of neighbors)
P_f^K = P_f^K         (K-hop: K steps away)
```

**Bidirectional diffusion convolution:**
```
h = Σₖ₌₀ᴷ (θₖ₁ · P_f^k + θₖ₂ · P_b^k) · X

Where:
  P_f^k = forward random walk, k steps
  P_b^k = backward random walk, k steps
  θ = learned weights
  X = input signal
```

```python
# graph_builder.py: compute_diffusion_matrices (lines 133-166)
P_forward = D_inv @ adj           # Forward: D^(-1)W
P_backward = D_inv @ adj.T        # Backward: D^(-1)W^T

# K=2 produces 4 support matrices:
supports = [P_f^1, P_b^1, P_f^2, P_b^2]
```

## DCGRU Cell (Diffusion Convolutional GRU)

Standard GRU replaces matrix multiplications with diffusion convolutions:

```
Standard GRU:                    DCGRU:
r = σ(W_r · [h, x])      →     r = σ(DiffConv([h, x]))
u = σ(W_u · [h, x])      →     u = σ(DiffConv([h, x]))
c = tanh(W_c · [x, r⊙h]) →     c = tanh(DiffConv([x, r⊙h]))
h_new = u⊙h + (1-u)⊙c          h_new = u⊙h + (1-u)⊙c
```

```python
# dcrnn.py: DCGRUCell.forward (lines 77-97)
combined = torch.cat([x, h], dim=-1)           # Concatenate input and hidden
gates = sigmoid(self.gate_conv(combined, supports))  # Diffusion conv for gates
r, u = gates.split(hidden_dim, dim=-1)         # Reset and update gates
combined_r = torch.cat([x, r * h], dim=-1)     # Apply reset gate
candidate = tanh(self.candidate_conv(combined_r, supports))
h_new = u * h + (1 - u) * candidate            # New hidden state
```

## Encoder-Decoder Architecture

```
ENCODER (processes input sequence left-to-right):

t=0  t=1  t=2  ...  t=11
 │    │    │          │
 ▼    ▼    ▼          ▼
[DCGRU] → [DCGRU] → [DCGRU] → ... → [DCGRU] → h_final
         Layer 1 (hidden=64)
[DCGRU] → [DCGRU] → [DCGRU] → ... → [DCGRU] → h_final
         Layer 2 (hidden=64)

DECODER (generates predictions one step at a time):

h_enc → [DCGRU] → proj → ŷ₁
              ↓
         [DCGRU] → proj → ŷ₂    (feeds ŷ₁ as input, or true y₁ with teacher forcing)
              ↓
         [DCGRU] → proj → ŷ₃
              ↓
          ... × 12 steps total
```

## Teacher Forcing

**Problem:** During training, the decoder feeds its OWN predictions as input to the next step. If early predictions are wrong, errors compound — "exposure bias."

**Solution:** With probability `tf_ratio`, feed the TRUE target value instead of the prediction.

```python
# dcrnn.py: DCRNNDecoder.forward (lines 169-183)
if torch.rand(1).item() < tf_ratio:
    decoder_input = teacher_forcing[:, t]  # Use ground truth
else:
    decoder_input = output                  # Use model's own prediction
```

**Schedule:** tf_ratio decays linearly from ~0.5 to 0 over training. Early epochs get more help; later epochs must predict independently.

## Shape Trace Through DCRNN

```
Input:          (128, 12, 207)              # batch=128, time=12, sensors=207
Add feature:    (128, 12, 207, 1)           # add channel dim

ENCODER:
  t=0: x=(128, 207, 1) → DCGRU_L1 → h1=(128, 207, 64)
                        → DCGRU_L2 → h2=(128, 207, 64)
  t=1: x=(128, 207, 1) → DCGRU_L1(h1) → h1_new
                        → DCGRU_L2(h2) → h2_new
  ...
  t=11: → final h1, h2

DECODER:
  step 0: input=zeros → DCGRU → project(h2) → ŷ₁ (128, 207, 1)
  step 1: input=ŷ₁   → DCGRU → project(h2) → ŷ₂
  ...
  step 11: → ŷ₁₂

Stack:         (128, 12, 207, 1)
Squeeze:       (128, 12, 207)               # final output

Parameters: 371,393
```

## Why DCRNN is Slower Than STGCN

| Property | STGCN | DCRNN |
|---|---|---|
| Temporal processing | Parallel (Conv1D) | Sequential (GRU step-by-step) |
| Parameters | 79K | 371K (4.7× more) |
| Graph conv per forward | 2 blocks × 1 = 2 | 12 encoder + 12 decoder = 24 calls |
| Parallelizable | Yes (all timesteps at once) | No (must process t before t+1) |
| Time per epoch (METR-LA) | ~2.6s | ~42s (16× slower) |

---

# PART 11 — TRAINING PIPELINE

## Loss Function: MSE on Normalized Data

```python
loss = MSE(predictions_normalized, targets_normalized)
     = (1/n) Σ (ŷᵢ - yᵢ)²
```

**Why MSE not MAE for training?** MSE is differentiable everywhere (MAE has a non-differentiable point at 0). MSE also penalizes large errors more, which is desirable during training.

**Why evaluate with MAE?** MAE is more interpretable — "average error in mph." RMSE penalizes outliers more. MAPE gives percentage error. We report all three.

## Adam Optimizer

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t         # Momentum (exponential moving average of gradients)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²        # Velocity (EMA of squared gradients)
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)  # Update with adaptive learning rate

Your config: lr=0.001, weight_decay=0.0001 (L2 regularization)
```

## Learning Rate Scheduling

```python
# ReduceLROnPlateau: if val_loss doesn't improve for 5 epochs, lr *= 0.5
# Your training logs show:
# Epoch 20: LR 1.00e-03
# Epoch 25: LR 5.00e-04  ← reduced after plateau
# Epoch 35: LR 2.50e-04  ← reduced again
```

## Early Stopping

```python
# If val_loss doesn't improve for 15 epochs, stop training
# STGCN METR-LA: stopped at epoch 45 (best at epoch 30)
# DCRNN METR-LA: ran all 100 epochs (kept improving)
```

## Mixed Precision Training (AMP)

```python
# train.py uses torch.cuda.amp
with autocast(enabled=True):        # Forward pass in fp16
    predictions = model(X)
    loss = criterion(predictions, Y)

scaler.scale(loss).backward()       # Backward pass with loss scaling
scaler.step(optimizer)              # Update weights in fp32
```

**Why AMP?** RTX 4060/4090 have Tensor Cores that compute fp16 matrix multiplications 2× faster than fp32. AMP automatically uses fp16 where safe and fp32 where precision matters.

---

*Continue to PART 3 → RESEARCH_GUIDE_PART3.md for Robustness, Ablation, Results Analysis, Viva Questions, and Paper Writing*
