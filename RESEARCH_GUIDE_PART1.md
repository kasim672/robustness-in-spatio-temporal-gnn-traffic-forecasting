# COMPREHENSIVE RESEARCH GUIDE — PART 1
# Foundations: Problem → Data → Baselines → Graph Theory

---

# PART 1 — PROBLEM UNDERSTANDING

## What is Traffic Forecasting?

**Intuitive explanation:**
Imagine 207 speed sensors placed across Los Angeles highways. Each sensor records the average vehicle speed every 5 minutes. Traffic forecasting means: given the last 60 minutes of speed readings from ALL sensors, predict what the speeds will be for the NEXT 60 minutes at ALL sensors simultaneously.

```
TIME →
Sensor 1:  [65, 63, 58, 52, 48, 45, 42, 40, 38, 35, 33, 30] → predict next 12 values
Sensor 2:  [70, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40, 38] → predict next 12 values
...
Sensor 207:[55, 55, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46] → predict next 12 values
```

**Formal definition:**
Given historical observations X(t-T+1), ..., X(t) where X(t) ∈ ℝ^N (N sensors), predict future values X(t+1), ..., X(t+H) where H is the prediction horizon.

In your project: T = 12 (60 min input), H = 12 (60 min output), N = 207 (METR-LA) or 325 (PEMS-BAY).

## Why Traffic Forecasting Matters

1. **Navigation:** Google Maps, Waze predict travel times
2. **Signal control:** Adaptive traffic lights adjust timing
3. **Urban planning:** Cities design roads based on predicted congestion
4. **Emergency services:** Ambulances need fastest routes in real-time
5. **Autonomous vehicles:** Self-driving cars need to predict surrounding traffic
6. **Economic impact:** Traffic congestion costs the US ~$87 billion/year

## The Two Types of Dependencies

### Temporal Dependencies
Traffic at sensor i at time t depends on its OWN past values:

```
Morning rush:  65 → 60 → 55 → 50 → 45 → 40  (speed drops gradually)
                                                 ↓
Recovery:      40 → 42 → 45 → 50 → 55 → 60  (speed recovers after rush)
```

**Key patterns:**
- **Trend:** Speed gradually changes (rush hour onset)
- **Periodicity:** Similar patterns repeat daily (8am rush, 5pm rush)
- **Seasonality:** Weekly patterns (weekday vs weekend)

### Spatial Dependencies
Traffic at sensor i depends on NEIGHBORING sensors:

```
Highway flow direction →

Sensor A -------- Sensor B -------- Sensor C
(congested)       (slowing)        (still fast)

If A is jammed, B will jam in ~5 min, C in ~10 min
```

**This is WHY we need graphs.** Traditional models (ARIMA, LSTM) only see temporal patterns. Graph Neural Networks see BOTH temporal AND spatial patterns.

## Challenges in Traffic Forecasting

| Challenge | Description | Example |
|---|---|---|
| Non-linearity | Speed doesn't change linearly | Sudden accidents cause instant drops |
| Non-stationarity | Statistical properties change over time | Rush hour vs midnight |
| Spatial heterogeneity | Different sensors behave differently | Highway vs residential |
| Missing data | Sensors fail or malfunction | 8.11% zeros in METR-LA |
| Long-range dependencies | Events propagate across the network | A crash 10km away affects you in 20 min |
| Multiple horizons | Short-term is easier than long-term | 15 min prediction is easier than 60 min |

## Short-term vs Long-term Forecasting

| Horizon | Steps | Time | Difficulty | Use case |
|---|---|---|---|---|
| Short | 3 | 15 min | Easy | Navigation, signal control |
| Medium | 6 | 30 min | Medium | Route planning |
| Long | 12 | 60 min | Hard | Urban planning, logistics |

**Why long-term is harder:** Errors compound. At 15 min, you're essentially extrapolating recent trends. At 60 min, you need to understand cyclical patterns, spatial propagation, and regime changes.

### Key Takeaways
- Traffic forecasting is a multivariate spatio-temporal prediction problem
- It has BOTH temporal (time) and spatial (location) dependencies
- Graph Neural Networks are designed to capture both simultaneously
- Longer horizons are exponentially harder

### Common Viva Questions
1. **Q:** Why can't we just use ARIMA for traffic forecasting?
   **A:** ARIMA only captures temporal patterns for individual sensors. It cannot model spatial dependencies (how congestion at one sensor affects neighboring sensors). It also assumes linear relationships and stationarity, which traffic data violates.

2. **Q:** What makes traffic forecasting different from weather forecasting?
   **A:** Traffic is constrained to a road network (graph structure), has sharper transitions (accidents cause sudden changes), and is heavily influenced by human behavior (rush hours). Weather is on a continuous spatial field.

3. **Q:** Why 5-minute intervals?
   **A:** This is the native sampling rate of Caltrans PeMS loop detectors. It provides sufficient temporal resolution to capture rush-hour dynamics while keeping data volume manageable.

---

# PART 2 — DATASETS

## METR-LA (Metropolitan Traffic — Los Angeles)

**History:** Collected by the Caltrans Performance Measurement System (PeMS). The dataset was curated by Li et al. (2018) in the DCRNN paper. It has become the standard benchmark for spatio-temporal traffic forecasting.

**Physical setup:** 207 inductive loop detectors embedded in the pavement of Los Angeles County highways. Each detector measures the average speed of vehicles passing over it in 5-minute windows.

```
Properties:
- Sensors:       207
- Timesteps:     34,272
- Duration:      Mar 1, 2012 → Jun 27, 2012 (≈4 months)
- Interval:      5 minutes
- Feature:       Speed in miles per hour (mph)
- Missing/Zero:  8.11% (sensor failures → recorded as 0)
```

**What "speed" means physically:** The loop detector counts vehicles and measures the time each vehicle occupies the detector zone. From this, it computes average speed. A reading of 65 mph means vehicles are flowing freely. A reading of 15 mph means heavy congestion. A reading of 0 mph means either no vehicles OR sensor failure (ambiguous — this is why we clean zeros).

## PEMS-BAY (PeMS — San Francisco Bay Area)

```
Properties:
- Sensors:       325
- Timesteps:     52,116
- Duration:      Jan 1, 2017 → Jun 30, 2017 (6 months)
- Interval:      5 minutes
- Feature:       Speed (mph)
- Missing/Zero:  0.003% (much cleaner)
```

## Dataset Comparison

| Property | METR-LA | PEMS-BAY | Implication |
|---|---|---|---|
| Sensors | 207 | 325 | PEMS-BAY has 57% more sensors → larger graph |
| Timesteps | 34,272 | 52,116 | PEMS-BAY has 52% more data → better training |
| Zero values | 8.11% | 0.003% | METR-LA needs more cleaning |
| Graph density | 2.2 conn/node | 7.6 conn/node | PEMS-BAY is 3.5× denser |
| Sparsity | 98.96% | 97.67% | Both are very sparse |

**Why researchers use both:** Testing on two datasets shows whether findings generalize. If a method works on both LA and Bay Area traffic, it's more likely to work elsewhere.

### Graph Statistics from Your Project

```
METR-LA Graph:
- 207 × 207 adjacency matrix
- 447 non-zero entries out of 42,849 possible
- 98.96% sparse (most sensors are NOT connected)
- Average 2.2 connections per sensor

PEMS-BAY Graph:
- 325 × 325 adjacency matrix
- 2,457 non-zero entries out of 105,625 possible
- 97.67% sparse
- Average 7.6 connections per sensor
```

**What this means:** In METR-LA, each sensor "talks to" only ~2 other sensors on average. The graph is extremely sparse. In PEMS-BAY, each sensor talks to ~7.6 others — the Bay Area highway network is more interconnected.

### Common Viva Questions
1. **Q:** Why not use more features like flow and occupancy?
   **A:** The benchmark uses speed-only to enable fair comparison across papers. Speed is also the most directly useful metric for navigation applications.

2. **Q:** How do you handle the 8.11% zero values in METR-LA?
   **A:** We use forward-fill → backward-fill → column mean replacement. Zeros in traffic data typically indicate sensor failures (not actual zero speed), so we impute them with temporally adjacent or average values.

---

# PART 3 — DATA PIPELINE

## Complete Pipeline Visualization

```
Raw CSV
  │
  ▼
[1. Load CSV] ──────────────── pd.read_csv() → numpy array (T, N)
  │                            T=34272, N=207
  ▼
[2. Clean] ─────────────────── Forward-fill → Backward-fill → Column mean
  │                            Replace zeros with column mean
  ▼
[3. Chronological Split] ──── Train (70%) | Val (10%) | Test (20%)
  │                            23,990    |   3,427   |  6,855
  ▼
[4. Normalize] ────────────── Z-score using TRAIN stats ONLY
  │                            x_norm = (x - μ_train) / σ_train
  ▼
[5. Create Sequences] ─────── Sliding windows PER CHUNK
  │                            X: (N, 12, 207)  Y: (N, 12, 207)
  ▼
[6. DataLoader] ───────────── Batched, shuffled (train only), pinned memory
```

### Stage 1: Loading

```python
# In data_loader.py, line 31-56
raw_data, sensor_ids, timestamps = load_csv(filepath)
# raw_data shape: (34272, 207) — 34K timesteps × 207 sensors
```

### Stage 2: Cleaning

**Why:** Traffic sensors sometimes fail and record 0 mph. This doesn't mean traffic stopped — it means the sensor broke. We must replace these with reasonable values.

```python
# Strategy: forward-fill → backward-fill → column mean
# In data_loader.py, lines 59-94

# Step 1: Forward-fill (use previous value)
# If sensor reads: [65, 60, 0, 0, 55], becomes [65, 60, 60, 60, 55]

# Step 2: Backward-fill (for leading zeros)
# If [0, 0, 65, 60], becomes [65, 65, 65, 60]

# Step 3: Column mean (if entire column was zero)
# Replace remaining zeros with that sensor's average speed
```

### Stage 3: Chronological Splitting

```python
# In data_loader.py, lines 167-188
T = len(data)           # 34,272
train_end = int(T * 0.7)  # 23,990
val_end = int(T * 0.8)    # 27,417

train_data = data[:23990]      # Mar 1 → May 24
val_data   = data[23990:27417] # May 24 → Jun 3
test_data  = data[27417:]      # Jun 3 → Jun 27
```

**CRITICAL:** We split BEFORE normalization and BEFORE sequence creation. This prevents two types of leakage.

### Stage 4: Normalization (Z-Score)

**Formula:**
```
x_normalized = (x - μ) / σ

where μ = mean of TRAINING data only
      σ = std of TRAINING data only
```

```python
# In data_loader.py, lines 262-269
mean = train_raw.mean(axis=0)  # shape (207,) — one mean per sensor
std = train_raw.std(axis=0)    # shape (207,) — one std per sensor

train_norm = (train_raw - mean) / std   # Apply to train
val_norm   = (val_raw - mean) / std     # Apply SAME stats to val
test_norm  = (test_raw - mean) / std    # Apply SAME stats to test
```

**Why train-only stats?** If we computed mean/std over ALL data, the model would implicitly "know" about future test data values through the normalization. This is information leakage.

### Stage 5: Sequence Creation

```python
# In data_loader.py, lines 139-164
# For each position i:
#   X[i] = data[i : i+12]           (input: 12 timesteps)
#   Y[i] = data[i+12 : i+12+12]    (target: next 12 timesteps)
```

**Visual example (sensor readings in time):**
```
Time:    t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 ... t23
         |←────── Input X (12 steps) ──────→|←────── Target Y (12 steps) ──────→|

Sample 0: X = [t0..t11],  Y = [t12..t23]
Sample 1: X = [t1..t12],  Y = [t13..t24]
Sample 2: X = [t2..t13],  Y = [t14..t25]
```

**Boundary leakage prevention:** Sequences are created SEPARATELY for train, val, and test chunks. A sequence from the training set can NEVER use values from the validation or test set.

### Stage 6: DataLoader

```python
# In data_loader.py, lines 191-222
DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,          # Only for training!
    num_workers=4,         # 4 CPU workers prepare batches
    pin_memory=True,       # Zero-copy CPU→GPU transfer
    persistent_workers=True, # Keep workers alive between epochs
    prefetch_factor=2,     # Pre-load 2 batches per worker
)
```

### Data Leakage — The Three Types You Must Prevent

| Type | What Goes Wrong | How We Prevent It |
|---|---|---|
| **Temporal leakage** | Normalization uses future data | Train-only mean/std |
| **Boundary leakage** | Sliding windows cross train/test boundary | Per-chunk sequence creation |
| **Spatial leakage** | Graph uses test sensor correlations | Graph built from train_raw only |

### Common Viva Questions
1. **Q:** What is data leakage and why is it dangerous?
   **A:** Data leakage occurs when information from the test set influences model training, either directly (seeing test labels) or indirectly (normalization stats, graph structure). It inflates reported performance and the model fails on truly unseen data.

2. **Q:** Why not shuffle the data randomly?
   **A:** Time-series data has temporal ordering. Random shuffling would put future data in the training set and past data in the test set, violating the causal structure. The model would learn to "predict" the past, which is useless.

3. **Q:** Why 70/10/20 split and not 80/10/10?
   **A:** 20% test gives a larger evaluation set (6,832 sequences) for statistically reliable metrics. The 10% validation is sufficient for early stopping and hyperparameter tuning. This split ratio follows Li et al. (2018).

---

# PART 4 — NORMALIZATION (Deep Dive)

## Z-Score Normalization

**Intuition:** Different sensors have different speed ranges. A highway sensor might average 65 mph while a residential sensor averages 30 mph. Without normalization, the model would focus on high-speed sensors simply because their values are larger.

**Formula:**
```
z = (x - μ) / σ

where:
  x = raw speed value
  μ = mean speed (per sensor, computed from training data)
  σ = standard deviation (per sensor, computed from training data)
```

**Example for one sensor:**
```
Raw speeds:     [60, 65, 70, 55, 50]
μ = 60,  σ = 7.07

Normalized:     [0.00, 0.71, 1.41, -0.71, -1.41]

Interpretation:
  0.00 → exactly average
  1.41 → 1.41 standard deviations above average (fast)
 -1.41 → 1.41 standard deviations below average (slow)
```

## Inverse Normalization (De-normalization)

**Why needed:** The model predicts normalized values. To compute MAE in real mph, we must convert back:

```
x_original = z × σ + μ

# In data_loader.py, lines 119-136
def denormalize_data(data, mean, std):
    return data * std + mean
```

**This is critical for evaluation.** If you compute MAE on normalized values, the number is meaningless — you need MAE in mph to compare with other papers.

---

# PART 5 — SEQUENCE CREATION (Deep Dive)

## Input/Output Shapes

```
Input X:  (batch_size, seq_len, num_sensors) = (128, 12, 207)
Output Y: (batch_size, pred_len, num_sensors) = (128, 12, 207)
```

**What each dimension means:**
- `batch_size = 128`: Process 128 samples simultaneously (GPU parallelism)
- `seq_len = 12`: Look back 12 timesteps = 60 minutes
- `num_sensors = 207`: All sensors at once
- `pred_len = 12`: Predict 12 timesteps = 60 minutes ahead

## Why seq_len = 12?

12 timesteps × 5 minutes = 60 minutes. One hour of history captures:
- Recent trends (last 15 min)
- Medium-term patterns (last 30 min)
- A full cycle of short-term dynamics

## Forecast Horizons

```
Prediction Y = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12]
                 ↑         ↑                   ↑
              5 min     15 min              60 min

Horizon mapping:
  15 min → Y[:, 2, :]   (step index 3, 0-indexed = 2)  → EVAL_HORIZONS["15min"] = 3
  30 min → Y[:, 5, :]   (step index 6, 0-indexed = 5)  → EVAL_HORIZONS["30min"] = 6
  60 min → Y[:, 11, :]  (step index 12, 0-indexed = 11) → EVAL_HORIZONS["60min"] = 12
```

---

# PART 6 — BASELINE MODELS

## 1. Persistence (Copy-Last-Value)

**Intuition:** "Tomorrow's weather will be the same as today." The simplest possible forecast — predict the last observed value for all future timesteps.

**Formula:** ŷ(t+h) = x(t)  for all horizons h

**Strengths:** Zero parameters, instant computation, surprisingly good at 15 min
**Weaknesses:** Cannot capture any dynamics, gets worse at longer horizons

**Your results:** MAE 3.856 (METR-LA) — this is the bar every model must beat.

## 2. Historical Average

**Intuition:** "Traffic at 8am on Tuesday will be similar to average 8am traffic across all Tuesdays."

**Formula:** ŷ(t+h) = (1/D) Σ x(t+h on same time-of-day in training set)

**Interesting property:** Predictions are IDENTICAL for all horizons (5.120 for all). This is because the time-of-day average doesn't change whether you're predicting 15 or 60 min ahead.

**Your results:** MAE 5.120 (METR-LA) — worst model. Proves that temporal dynamics matter.

## 3. ARIMA (AutoRegressive Integrated Moving Average)

**Intuition:** Combine three ideas:
- **AR (AutoRegressive):** Current value depends on past values
- **I (Integrated):** Difference the series to make it stationary  
- **MA (Moving Average):** Current value depends on past errors

**Your order:** ARIMA(2,1,2)
- p=2: Use last 2 values for autoregression
- d=1: Difference once (removes trend)
- q=2: Use last 2 errors for moving average

**Formula:**
```
After differencing: y'(t) = y(t) - y(t-1)

y'(t) = φ₁y'(t-1) + φ₂y'(t-2) + θ₁ε(t-1) + θ₂ε(t-2) + ε(t)

where φ = AR coefficients, θ = MA coefficients, ε = white noise
```

**Why d=1 was critical:** With d=0, ARIMA assumed the series was already stationary. Traffic speed is NOT stationary (rush hour changes the mean). Differencing (d=1) removes this non-stationarity. Without it, forecasts exploded.

**Limitation:** Fitted per-sensor (30 sensors for speed). Cannot model spatial correlations.

## 4. Random Forest

**Intuition:** An ensemble of decision trees. Each tree sees a random subset of features and training samples, then votes on the prediction.

**Your config:** 100 trees, max depth 15, flattened input

**Input transformation:** The 12×207 input matrix is flattened to a 2,484-dimensional vector. Each of the 12 RF models (one per horizon) independently predicts the next timestep.

**Strengths:** Robust, handles non-linearity, no normalization needed (though we still normalize for consistency)
**Weaknesses:** Flattening destroys temporal structure, no spatial modeling

## 5. LSTM (Long Short-Term Memory)

**Intuition:** A recurrent neural network with "memory cells" that can learn to remember or forget information over long sequences.

**Architecture in your code:**
```
Input (batch, 12, 207) → LSTM Layer 1 (hidden=64) → LSTM Layer 2 (hidden=64) → Linear → Output (batch, 12, 207)
```

**The LSTM cell equations:**
```
Forget gate:  f(t) = σ(W_f · [h(t-1), x(t)] + b_f)
Input gate:   i(t) = σ(W_i · [h(t-1), x(t)] + b_i)
Candidate:    c̃(t) = tanh(W_c · [h(t-1), x(t)] + b_c)
Cell state:   c(t) = f(t) ⊙ c(t-1) + i(t) ⊙ c̃(t)
Output gate:  o(t) = σ(W_o · [h(t-1), x(t)] + b_o)
Hidden state: h(t) = o(t) ⊙ tanh(c(t))
```

**Why LSTM over vanilla RNN:** Vanilla RNNs suffer from vanishing gradients — they can't learn long-term dependencies. LSTM's gating mechanism (forget, input, output gates) allows gradients to flow unchanged through the cell state, enabling learning over 12+ timesteps.

**Parameters:** 55,372 — much smaller than DCRNN (371K)

### Common Viva Questions
1. **Q:** Why is Persistence a useful baseline?
   **A:** It sets the minimum bar. Any model that can't beat "just copy the last value" has failed. Persistence MAE tells us how much the signal changes over the prediction horizon.

2. **Q:** Why does LSTM beat Random Forest?
   **A:** LSTM preserves temporal ordering through its sequential processing, while RF flattens the input and loses the time dimension. LSTM's gating mechanism can learn that "speed 5 minutes ago matters more than speed 60 minutes ago."

3. **Q:** Why not use a Transformer instead of LSTM?
   **A:** Transformers could work (and are used in recent papers like Traffic Transformer). LSTM was chosen for a fair comparison since STGCN/DCRNN papers used LSTM as their baseline. Also, with only 12 timesteps, LSTM's sequential nature is not a bottleneck.

---

# PART 7 — GRAPH THEORY FOUNDATIONS

## Nodes and Edges

**In your project:** Each sensor is a node. An edge between two sensors means their traffic patterns are correlated.

```
Graph G = (V, E)

V = {sensor_1, sensor_2, ..., sensor_207}  (vertices/nodes)
E = {(i, j) | sensors i and j are correlated}  (edges)
```

## Adjacency Matrix A

The adjacency matrix is an N×N matrix where A[i,j] > 0 if sensors i and j are connected.

```
     s1   s2   s3   s4
s1 [ 1.0  0.8  0.0  0.0 ]
s2 [ 0.8  1.0  0.7  0.0 ]    Edge weight = correlation strength
s3 [ 0.0  0.7  1.0  0.9 ]    0.0 = no connection
s4 [ 0.0  0.0  0.9  1.0 ]    1.0 = self-loop (diagonal)
```

**Your METR-LA adjacency:** 207×207 matrix, 447 non-zero entries, 98.96% sparse.

## Degree Matrix D

The degree of node i = sum of its edge weights = how "connected" it is.

```
D = diag(d₁, d₂, ..., d_N)    where d_i = Σⱼ A[i,j]

Example: d₁ = 1.0 + 0.8 = 1.8  (self-loop + one neighbor)
```

## Graph Laplacian L

```
L = D - A

Intuition: The Laplacian measures how different a node's value is from its neighbors.
If we have signal f on the graph:  (Lf)_i = d_i·f_i - Σⱼ A[i,j]·f_j
```

**This is the graph equivalent of the second derivative** (∇²f in continuous space). Just as ∇²f measures how a function deviates from its local average, Lf measures how a node's value deviates from its neighbors' weighted average.

## Normalized Laplacian

```
L_sym = D^(-1/2) · L · D^(-1/2) = I - D^(-1/2) · A · D^(-1/2)

# In graph_builder.py, lines 57-72
def symmetric_normalize(adj):
    degree = adj.sum(axis=1)
    d_inv_sqrt = degree ** (-0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt
```

**Why normalize?** Without normalization, high-degree nodes dominate. Normalization makes the operation independent of node degree.

## Eigenvalues and Spectral Theory

The Laplacian L has eigenvalues 0 = λ₁ ≤ λ₂ ≤ ... ≤ λ_N.

**Key properties:**
- λ₁ = 0 always (constant signal has zero Laplacian)
- λ₂ = **Fiedler value** = algebraic connectivity
  - λ₂ > 0 means the graph is connected
  - Larger λ₂ = more connected graph
  - λ₂ = 0 means the graph has disconnected components

**Why this matters for your project:** Your sparsity analysis examines how λ₂ changes with different ε thresholds. A very sparse graph (high ε) may disconnect the graph (λ₂ → 0), losing all spatial information.

## How Graph Theory Connects to Your Code

```
Graph Theory Concept    →    Code Location           →    Used By
─────────────────────────────────────────────────────────────────────
Adjacency Matrix A      →    graph_builder.py:12-54   →    Both models
Symmetric Norm D⁻½AD⁻½  →    graph_builder.py:57-72   →    STGCN
Random Walk Norm D⁻¹A   →    graph_builder.py:75-90   →    DCRNN
Laplacian L = I - Â     →    graph_builder.py:112     →    STGCN (Chebyshev)
Chebyshev Polynomials   →    graph_builder.py:93-130  →    STGCN
Diffusion Matrices      →    graph_builder.py:133-166 →    DCRNN
```

### Common Viva Questions
1. **Q:** What is the physical meaning of the adjacency matrix in your project?
   **A:** A[i,j] represents how correlated the traffic speed patterns of sensors i and j are. It's computed as a Gaussian kernel of the Pearson correlation: A[i,j] = exp(-(1-corr)²/2σ²). High correlation means sensors on the same highway segment.

2. **Q:** What does it mean that your graph is 98.96% sparse?
   **A:** Out of 42,849 possible sensor pairs, only 447 have significant correlation. Most sensors are far apart and have independent traffic patterns. Each sensor is connected to only ~2.2 others on average.

3. **Q:** What is the spectral gap and why does it matter?
   **A:** The spectral gap is the difference between the two smallest eigenvalues (λ₂ - λ₁ = λ₂ since λ₁=0). A larger spectral gap means information diffuses faster across the graph. For traffic, this means spatial patterns are learned more efficiently.

---

*Continue to PART 2 → RESEARCH_GUIDE_PART2.md for STGCN, DCRNN, Training, Robustness, and Paper Writing*
