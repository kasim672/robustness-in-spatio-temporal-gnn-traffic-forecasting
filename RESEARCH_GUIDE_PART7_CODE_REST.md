# RESEARCH GUIDE — PART 7B: LINE-BY-LINE CODE WALKTHROUGH
# graph_builder.py + data_loader.py + train.py

Notation:
  T  = total timesteps
  N  = number of sensors
  B  = batch size (128)
  Tr = training timesteps (~23,990 for METR-LA)

---

# FILE: src/graph_builder.py

---

## FUNCTION: compute_correlation_adj

```python
def compute_correlation_adj(data, sigma=0.1, epsilon=0.3):
    # data: (Tr, N) = (23990, 207)  — TRAINING data only
    
    corr_matrix = np.corrcoef(data.T)      # (N, N)
```
data.T: (N, Tr) — one row per sensor, transpose for corrcoef.
np.corrcoef: computes pairwise Pearson correlations.
MATH: C[i,j] = cov(x_i, x_j) / (σ_i · σ_j)
     = Σ_t (x_i(t) - μ_i)(x_j(t) - μ_j) / √(var_i · var_j)
Range: C[i,j] ∈ [-1, +1]
SHAPE: corr_matrix → (207, 207)

```python
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
```
Some sensors may be constant (zero variance → 0/0 = NaN).
Replace NaN with 0.0 → no connection for constant sensors.

```python
    dist = 1.0 - corr_matrix              # (N, N)
```
MATH: distance(i,j) = 1 - C[i,j]
dist = 0.0 when perfectly correlated (same behavior = no distance)
dist = 2.0 when perfectly anti-correlated
dist = 1.0 when uncorrelated

```python
    adj = np.exp(-(dist ** 2) / (2 * sigma ** 2))   # (N, N)
```
MATH: A[i,j] = exp(-d(i,j)² / 2σ²)  — Gaussian kernel
With σ=0.1: exp(-d²/0.02)
At d=0.05 (corr=0.95): exp(-0.0025/0.02) = exp(-0.125) = 0.882
At d=0.50 (corr=0.50): exp(-0.25/0.02) = exp(-12.5) = 0.0000037
The kernel is VERY sharp — only very high correlations survive.
SHAPE: adj → (207, 207)

```python
    adj[adj < epsilon] = 0.0              # threshold
    np.fill_diagonal(adj, 1.0)            # self-loops
```
Sparsification: zero out weak connections (adj < 0.3).
Self-loops: each node always aggregates its own features.
Result: 98.96% sparse for METR-LA.

---

## FUNCTION: symmetric_normalize

```python
def symmetric_normalize(adj):
    # adj: (N, N)
    degree = np.array(adj.sum(axis=1)).flatten()   # (N,)
```
degree[i] = Σ_j adj[i,j] = sum of all edge weights from node i.
MATH: d_i = Σ_j A[i,j]

```python
    d_inv_sqrt = np.power(degree, -0.5)    # (N,)  = 1/√d_i
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)       # (N, N) diagonal matrix
    return D_inv_sqrt @ adj @ D_inv_sqrt   # (N, N)
```
MATH: Â = D^{-1/2} · A · D^{-1/2}
Â[i,j] = A[i,j] / √(d_i · d_j)
Purpose: normalizes so that the influence of a neighbor is inversely
proportional to the square root of both node degrees.
High-degree nodes don't dominate (their contributions are scaled down).
isinf check: if d_i=0 (isolated node), d^{-0.5}=inf → set to 0.

---

## FUNCTION: compute_chebyshev_polynomials

```python
def compute_chebyshev_polynomials(adj_normalized, K):
    # adj_normalized: (N, N) = symmetrically normalized adjacency
    N = adj_normalized.shape[0]
    I = np.eye(N)                          # (N, N) identity

    L = I - adj_normalized                 # (N, N) Graph Laplacian
```
MATH: L = I - D^{-1/2}AD^{-1/2}  (normalized Laplacian)
Eigenvalues of L ∈ [0, 2].
L[i,i] = 1,  L[i,j] = -A[i,j]/√(d_i·d_j) for i≠j.
(Lf)_i = f_i - Σ_j A[i,j]/√(d_i·d_j) · f_j
= how much node i's feature differs from its weighted neighbor average.

```python
    lambda_max = eigsh(L_sparse, k=1, which='LM')[0]   # largest eigenvalue
    L_tilde = (2.0 / lambda_max) * L - I               # (N, N) scaled Laplacian
```
MATH: L̃ = (2/λ_max)·L - I
Purpose: scale eigenvalues from [0, λ_max] to [-1, 1].
Required for Chebyshev polynomials to be valid (domain is [-1,1]).

```python
    cheb_polys = [I, L_tilde]
    for k in range(2, K+1):
        cheb_polys.append(2 * L_tilde @ cheb_polys[-1] - cheb_polys[-2])
    return cheb_polys[:K+1]
```
MATH: Chebyshev recurrence:
  T_0(L̃) = I
  T_1(L̃) = L̃
  T_k(L̃) = 2·L̃·T_{k-1}(L̃) - T_{k-2}(L̃)
Each T_k is an (N,N) matrix.
T_k(L̃) applied to signal x gives k-hop neighborhood aggregation.
Returns list of K+1=4 matrices: [T_0, T_1, T_2, T_3]

---

## FUNCTION: compute_diffusion_matrices

```python
def compute_diffusion_matrices(adj, K=2):
    P_forward = random_walk_normalize(adj)    # D_out^{-1} W  (N, N)
    P_backward = random_walk_normalize(adj.T) # D_in^{-1} W^T (N, N)
```
MATH: P_f[i,j] = A[i,j] / d_i  — probability of walking from i to j.
Rows sum to 1 (stochastic matrix).
P_forward: models how info flows WITH traffic direction.
P_backward: models how info flows AGAINST traffic direction.

```python
    supports = []
    Pf_k = P_forward.copy()
    Pb_k = P_backward.copy()
    for k in range(K):               # K=2 → k = 0, 1
        if k == 0:
            supports.append(P_forward)
            supports.append(P_backward)
        else:
            Pf_k = Pf_k @ P_forward      # P_f^2 = P_f · P_f
            Pb_k = Pb_k @ P_backward     # P_b^2 = P_b · P_b
            supports.append(Pf_k)
            supports.append(Pb_k)
    return supports   # [P_f^1, P_b^1, P_f^2, P_b^2]  — 4 matrices
```
4 supports for K=2 diffusion steps in both directions.
P^k: probability of reaching a node in EXACTLY k random walk steps.
Captures 1-hop and 2-hop neighborhood information bidirectionally.

---

# FILE: src/data_loader.py

---

## FUNCTION: load_csv

```python
def load_csv(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = df.values.astype(np.float32)
```
index_col=0: first column is the timestamp (becomes DataFrame index).
parse_dates=True: parse index as datetime objects.
.values: convert DataFrame to raw numpy array.
SHAPE: data → (34272, 207) for METR-LA
dtype float32: use 32-bit floats (GPU-friendly, saves memory vs float64).

---

## FUNCTION: handle_missing_values

```python
    df = pd.DataFrame(data)
    df = df.ffill().bfill()          # forward-fill then backward-fill
    df = df.fillna(df.mean())        # remaining NaN → column mean
    df = df.fillna(0)                # completely empty columns → 0
    cleaned = df.values.astype(np.float32)
```
Forward-fill: [65, 60, NaN, NaN, 55] → [65, 60, 60, 60, 55]
Backward-fill: [NaN, NaN, 65, 60] → [65, 65, 65, 60]
Column mean: if entire column is NaN, use 0 (last resort).

```python
    for col in range(cleaned.shape[1]):
        col_data = cleaned[:, col]
        mask = col_data == 0
        if mask.sum() > 0 and mask.sum() < len(col_data):
            col_mean = col_data[~mask].mean()
            cleaned[mask, col] = col_mean
```
Replace remaining zeros with non-zero column mean.
mask.sum() < len: don't process columns where ALL values are zero.
SHAPE: cleaned → (34272, 207) same as input.

---

## FUNCTION: normalize_data

```python
def normalize_data(data, mean=None, std=None):
    if mean is None:
        mean = data.mean(axis=0)       # (N,) — one mean per sensor
    if std is None:
        std = data.std(axis=0)         # (N,) — one std per sensor
        std[std < 1e-5] = 1.0          # avoid division by zero
    normalized = (data - mean) / std   # broadcasts: (T,N) - (N,) / (N,)
    return normalized, mean, std
```
MATH: z[t,i] = (x[t,i] - μ_i) / σ_i
axis=0: compute statistics across time dimension (per sensor).
std < 1e-5 check: if sensor has near-zero variance, set std=1 (no scaling).
Broadcasting: mean/std shape (N,) broadcasts over T dimension automatically.
SHAPE: normalized → (T, N) same as input.

---

## FUNCTION: create_sequences

```python
def create_sequences(data, seq_len=12, pred_len=12):
    T, N = data.shape
    num_samples = T - seq_len - pred_len + 1
    X = np.zeros((num_samples, seq_len, N), dtype=np.float32)
    Y = np.zeros((num_samples, pred_len, N), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = data[i : i + seq_len]           # (12, N)
        Y[i] = data[i + seq_len : i + seq_len + pred_len]  # (12, N)
    
    return X, Y
```
Sliding window with stride 1.
Sample i: input = timesteps [i, i+11], target = timesteps [i+12, i+23].
num_samples = 23990 - 12 - 12 + 1 = 23967 for training.
SHAPE X: (23967, 12, 207),  SHAPE Y: (23967, 12, 207)

---

## FUNCTION: split_data

```python
def split_data(data, train_ratio=0.7, val_ratio=0.1):
    T = len(data)                              # 34272
    train_end = int(T * 0.7)                   # 23990
    val_end = int(T * (0.7 + 0.1))            # 27417
    
    train_data = data[:23990]                  # (23990, 207)
    val_data   = data[23990:27417]             # (3427, 207)
    test_data  = data[27417:]                  # (6855, 207)
    return train_data, val_data, test_data
```
Pure index slicing — no shuffling, no overlap.
Critical: split BEFORE normalization to avoid statistical leakage.
The 3 chunks never share timesteps.

---

## FUNCTION: prepare_dataset (the master pipeline)

```python
raw_data, sensor_ids, timestamps = load_csv(filepath)
# SHAPE: (34272, 207)

raw_data = handle_missing_values(raw_data)
# SHAPE: (34272, 207) — zeros replaced

train_raw, val_raw, test_raw = split_data(raw_data)
# train_raw: (23990, 207), val_raw: (3427, 207), test_raw: (6855, 207)

mean = train_raw.mean(axis=0)   # (207,) — per-sensor mean from TRAIN only
std = train_raw.std(axis=0)     # (207,) — per-sensor std from TRAIN only

train_norm = (train_raw - mean) / std   # (23990, 207)
val_norm   = (val_raw - mean) / std     # (3427, 207)
test_norm  = (test_raw - mean) / std    # (6855, 207)

# Sequences created SEPARATELY per chunk (no boundary leakage)
X_train, Y_train = create_sequences(train_norm)  # (23967, 12, 207)
X_val,   Y_val   = create_sequences(val_norm)    # (3404, 12, 207)
X_test,  Y_test  = create_sequences(test_norm)   # (6832, 12, 207)
```
This order prevents ALL THREE leakage types:
  Temporal leakage → stats from train only
  Boundary leakage → sequences per chunk only
  Spatial leakage  → graph built from train_raw only (in run_gnn.py)

---

# FILE: src/train.py

---

## FUNCTION: train_model

```python
def train_model(model, loaders, graph, config, save_path, ...):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
```
Adam optimizer with L2 regularization (weight_decay).
MATH: θ ← θ - lr · m̂/(√v̂ + ε) - weight_decay · θ
weight_decay adds gradient of ||θ||² → keeps weights small.

```python
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
```
GradScaler: scales loss upward before backward pass to prevent fp16 underflow.
After backward pass, unscales gradients before optimizer step.

```python
    for epoch in range(config.EPOCHS):
        model.train()
        for X_batch, Y_batch in loaders['train']:
            X_batch = X_batch.to(device)   # (B, 12, N)
            Y_batch = Y_batch.to(device)   # (B, 12, N)
            
            optimizer.zero_grad()
```
Zero gradients: PyTorch accumulates gradients by default.
Must zero before each batch to prevent accumulation.

```python
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                if is_stgcn:
                    pred = model(X_batch, cheb_polys_gpu)
                else:
                    pred = model(X_batch, supports_gpu,
                                 targets=Y_batch, tf_ratio=tf_ratio)
```
autocast context: automatically uses fp16 for eligible operations.
STGCN: pass Chebyshev polynomials as graph representation.
DCRNN: pass diffusion supports + targets for teacher forcing.
Both return shape: (B, 12, N) = (128, 12, 207)

```python
                loss = F.mse_loss(pred, Y_batch)
```
MATH: MSE = (1/BxTxN) Σ_{b,t,n} (pred[b,t,n] - Y[b,t,n])²
Loss is computed on NORMALIZED values (both pred and Y_batch are normalized).
This is correct — denormalization only happens during evaluation.

```python
            scaler.scale(loss).backward()
```
Scales loss by scaler factor (e.g., 65536).
Computes gradients in fp32 (despite fp16 forward pass).
Scaled gradients prevent fp16 underflow.

```python
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
```
unscale_: divides gradients by scaler factor (back to true scale).
clip_grad_norm_: if ||grad||_2 > 5.0, rescale all gradients.
MATH: if ||g|| > max_norm: g ← g × (max_norm / ||g||)
Prevents exploding gradients, especially important for DCRNN's RNN.

```python
            scaler.step(optimizer)     # update weights in fp32
            scaler.update()            # adjust scale factor for next iteration
```
scaler.step: optimizer updates weights using unscaled fp32 gradients.
scaler.update: if gradients had inf/NaN, reduce scale. Otherwise increase.

```python
        # Validation
        val_loss = evaluate_epoch(model, loaders['val'], ...)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break     # early stopping
```
Save ONLY the best model (by validation loss).
state_dict(): ordered dict of all learnable parameter tensors.
Early stopping: stop when val_loss hasn't improved for 15 epochs.

---

## Summary: Data Flow Through Full Pipeline

```
CSV file
  → load_csv():          (34272, 207)
  → handle_missing():    (34272, 207)  zeros replaced
  → split_data():        train(23990,207) | val(3427,207) | test(6855,207)
  → normalize():         train_norm(23990,207) | val_norm | test_norm
                         mean(207,) and std(207,) from train only
  → create_sequences():  X_train(23967,12,207) Y_train(23967,12,207)
  → DataLoader:          batches of (128,12,207)
  → model.forward():     (128,12,207) → (128,12,207) predictions (normalized)
  → evaluate():          denormalize → MAE/RMSE/MAPE in mph
  → save metrics:        results/metrics/METR-LA_stgcn.json
```

Every step is causally ordered — no future information ever reaches the model.
