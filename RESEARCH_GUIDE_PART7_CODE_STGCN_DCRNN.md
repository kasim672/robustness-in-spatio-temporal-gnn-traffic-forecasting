# RESEARCH GUIDE — PART 7A: LINE-BY-LINE CODE WALKTHROUGH
# stgcn.py + dcrnn.py

Notation used throughout:
  B  = batch size (128)
  T  = sequence length (12)
  N  = number of sensors (207 METR-LA / 325 PEMS-BAY)
  C  = number of channels (varies per layer)
  K  = Chebyshev order (3)

---

# FILE: src/models/stgcn.py

---

## CLASS: ChebGraphConv

```python
class ChebGraphConv(nn.Module):
```
Implements the Chebyshev spectral graph convolution from Defferrard et al. (2016).
Mathematical goal: y = Σ_{k=0}^{K} θ_k · T_k(L̃) · x
where T_k are Chebyshev polynomials of the scaled graph Laplacian L̃.

---

```python
def __init__(self, K, in_channels, out_channels):
    self.K = K                       # polynomial order = 3
    self.weight = nn.Parameter(
        torch.FloatTensor(K+1, in_channels, out_channels)
    )
```
SHAPE: weight → (4, C_in, C_out)
One weight matrix per polynomial order (K+1 = 4 matrices).
Each matrix mixes C_in input features into C_out output features.
MATH: θ_k ∈ ℝ^{C_in × C_out}  for k = 0,1,2,3

```python
    self.bias = nn.Parameter(torch.FloatTensor(out_channels))
    nn.init.xavier_uniform_(self.weight)
    nn.init.zeros_(self.bias)
```
SHAPE: bias → (C_out,)
Xavier init: keeps variance = 2/(fan_in + fan_out), prevents vanishing/exploding activations.
Bias initialized to zero — standard practice.

---

```python
def forward(self, x, cheb_polys):
```
INPUT:
  x          → (B, N, C_in)   features at all nodes for one timestep
  cheb_polys → list of K+1 matrices, each (N, N)
               [T_0(L̃), T_1(L̃), T_2(L̃), T_3(L̃)]

```python
    batch, N, C_in = x.shape
    outputs = []
    for k in range(self.K + 1):       # k = 0, 1, 2, 3
        Tk = cheb_polys[k]             # (N, N)
        Tk_x = torch.einsum('mn,bnc->bmc', Tk, x)
```
SHAPE: Tk → (N, N),  x → (B, N, C_in)
einsum 'mn,bnc->bmc':
  For each batch b and channel c:
    output[b, m, c] = Σ_n  Tk[m,n] · x[b,n,c]
This is: T_k(L̃) @ X   — graph filtering at order k
SHAPE after einsum: Tk_x → (B, N, C_in)

```python
        outputs.append(Tk_x)
    x_cheb = torch.stack(outputs, dim=0)   # (K+1, B, N, C_in)
```
SHAPE: x_cheb → (4, B, N, C_in)
Stack along new dim 0: all polynomial orders together.

```python
    out = torch.einsum('kbnc,kco->bno', x_cheb, self.weight)
```
einsum 'kbnc,kco->bno':
  out[b,n,o] = Σ_k Σ_c  x_cheb[k,b,n,c] · weight[k,c,o]
MATH: y = Σ_k T_k(L̃)·X · θ_k     (sum over polynomial orders)
SHAPE after einsum: out → (B, N, C_out)

```python
    return out + self.bias             # (B, N, C_out)
```
Add bias broadcast over batch and node dimensions.
FINAL SHAPE: (B, N, C_out)

---

## CLASS: TemporalConv

```python
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.conv = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size-1)//2),
        )
```
Conv2d operates on (B, C, H, W). Here H=N (nodes), W=T (time).
kernel_size=(1,3): convolve along time axis only (W), not spatial axis (H).
padding=(0,1): same-padding on time axis so T stays the same.
Output channels = 2*out_channels because we split into P and Q for GLU.
MATH: this implements two parallel convolutions simultaneously.

```python
    def forward(self, x):
        # x: (B, C_in, N, T)
        conv_out = self.conv(x)        # (B, 2*C_out, N, T)
        P, Q = conv_out.split(self.out_channels, dim=1)
```
SHAPE: P → (B, C_out, N, T),  Q → (B, C_out, N, T)
Split along channel dimension (dim=1).

```python
        return P * torch.sigmoid(Q)    # (B, C_out, N, T)
```
MATH: GLU gate: output = P ⊙ σ(Q)
σ(Q) ∈ (0,1) acts as a soft gate per channel per node per timestep.
Intuition: Q learns WHICH temporal features should pass through.
           P contains the actual feature values.
SHAPE: (B, C_out, N, T) — same T due to padding.

---

## CLASS: STConvBlock

```python
class STConvBlock(nn.Module):
    def __init__(self, K, in_channels, spatial_channels, out_channels, kernel_size=3):
        self.temporal1 = TemporalConv(in_channels, spatial_channels, kernel_size)
        self.graph_conv = ChebGraphConv(K, spatial_channels, spatial_channels)
        self.temporal2 = TemporalConv(spatial_channels, out_channels, kernel_size)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
```
Architecture: Temporal → Graph → Temporal (sandwich pattern).
LayerNorm normalizes over the channel dimension (last dim after permute).
Dropout 10% for regularization.

```python
    def forward(self, x, cheb_polys):
        # x: (B, C_in, N, T)
        t1 = self.temporal1(x)         # (B, spatial_ch, N, T)
```
First temporal conv extracts time-local features.
SHAPE: (B, 16, N, T) for Block1, (B, 32, N, T) for Block2.

```python
        batch, C, N, T = t1.shape
        t1_reshaped = t1.permute(0, 3, 2, 1).reshape(batch*T, N, C)
```
MATH: reshape to apply graph conv independently per timestep.
permute(0,3,2,1): (B, C, N, T) → (B, T, N, C)
reshape(batch*T, N, C): (B, T, N, C) → (B*T, N, C)
Now each of the B*T items is one timestep's node features.
SHAPE: (B*T, N, C)

```python
        gc = self.graph_conv(t1_reshaped, cheb_polys)  # (B*T, N, C)
        gc = gc.reshape(batch, T, N, C).permute(0, 3, 2, 1)  # (B, C, N, T)
```
Apply Chebyshev graph conv to each timestep independently.
reshape and permute back to (B, C, N, T).
MATH: for each t: h_t = Σ_k θ_k · T_k(L̃) · x_t

```python
        gc = F.relu(gc)                # (B, C, N, T)
        t2 = self.temporal2(gc)        # (B, C_out, N, T)
```
ReLU non-linearity after graph conv.
Second temporal conv extracts higher-level temporal features.

```python
        t2 = t2.permute(0, 2, 3, 1)   # (B, N, T, C_out)
        t2 = self.layer_norm(t2)       # (B, N, T, C_out)
        t2 = self.dropout(t2)
        t2 = t2.permute(0, 3, 1, 2)   # (B, C_out, N, T)
        return t2
```
LayerNorm expects last dim to be the normalized dimension (C_out).
So permute so C_out is last, normalize, permute back.
MATH: LN(x) = (x - μ) / (σ + ε) · γ + β  per sample per token.

---

## CLASS: STGCN

```python
def __init__(self, num_sensors, seq_len=12, pred_len=12, K=3, channels=None):
    channels = [1, 16, 32, 64]        # default channel progression
    self.block1 = STConvBlock(K, channels[0], channels[1], channels[2])
    # K=3, in=1, spatial=16, out=32
    self.block2 = STConvBlock(K, channels[2], channels[2], channels[3])
    # K=3, in=32, spatial=32, out=64
    self.output_conv = nn.Conv2d(channels[3], pred_len, kernel_size=(1, seq_len))
    # collapses T=12 → 1, expands C=64 → 12 (one per horizon)
    self.output_linear = nn.Linear(num_sensors, num_sensors)
    # per-horizon sensor-to-sensor projection
```

```python
def forward(self, x, cheb_polys):
    # x: (B, T, N) = (B, 12, 207)
    x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, T)
```
permute(0,2,1): (B,T,N) → (B,N,T)
unsqueeze(1): add channel dim → (B,1,N,T)
Now format matches Conv2d's expected (B, C_in, H, W) = (B, 1, N, T).

```python
    x = self.block1(x, cheb_polys)    # (B, 32, N, T)
    x = self.block2(x, cheb_polys)    # (B, 64, N, T)
    x = self.output_conv(x)           # (B, 12, N, 1)
```
output_conv: kernel=(1, seq_len=12) collapses T dimension entirely.
Produces 12 channels = 12 prediction horizons.
SHAPE: (B, pred_len=12, N, 1)

```python
    x = x.squeeze(-1)                 # (B, 12, N)
    x = self.output_linear(x)         # (B, 12, N)
    return x
```
squeeze: remove last dim of size 1.
output_linear: Linear(N, N) — independent projection per horizon.
FINAL SHAPE: (B, pred_len, N) = (128, 12, 207) ✓

---
---

# FILE: src/models/dcrnn.py

---

## CLASS: DiffusionConv

```python
class DiffusionConv(nn.Module):
    def __init__(self, num_supports, in_channels, out_channels):
        total_supports = num_supports + 1   # +1 for identity
        self.weight = nn.Parameter(
            torch.FloatTensor(total_supports, in_channels, out_channels)
        )
```
SHAPE: weight → (num_supports+1, C_in, C_out)
num_supports = 4 (P_f^1, P_b^1, P_f^2, P_b^2)
total_supports = 5 (4 diffusion + 1 identity)
MATH: one weight matrix θ_s per support matrix S_s.

```python
    def forward(self, x, supports):
        # x: (B, N, C_in)
        out = torch.einsum('bnc,co->bno', x, self.weight[0])
```
SHAPE: weight[0] → (C_in, C_out), x → (B, N, C_in)
einsum: out[b,n,o] = Σ_c x[b,n,c] · weight[0][c,o]
This is the identity term: I · x · θ_0 (each node uses its own features).
SHAPE after einsum: (B, N, C_out)

```python
        for k, S in enumerate(supports):
            Sx = torch.einsum('mn,bnc->bmc', S, x)   # (B, N, C_in)
```
S → (N, N) diffusion matrix (e.g., P_f^1)
einsum: Sx[b,m,c] = Σ_n S[m,n] · x[b,n,c]
MATH: S·x = diffusion-weighted neighbor aggregation.
Row m of S contains transition probabilities FROM node m TO its neighbors.
So Sx[b,m,:] = weighted average of neighbor features.
SHAPE: (B, N, C_in)

```python
            out = out + torch.einsum('bnc,co->bno', Sx, self.weight[k+1])
```
Add contribution of this support: S·x · θ_{k+1}
MATH: full formula: y = Σ_s (S_s · x) · θ_s + bias
SHAPE: (B, N, C_out)

```python
        return out + self.bias         # (B, N, C_out)
```

---

## CLASS: DCGRUCell

```python
class DCGRUCell(nn.Module):
    def __init__(self, num_supports, in_channels, hidden_dim):
        self.gate_conv = DiffusionConv(
            num_supports, in_channels + hidden_dim, 2 * hidden_dim
        )
        self.candidate_conv = DiffusionConv(
            num_supports, in_channels + hidden_dim, hidden_dim
        )
```
gate_conv: input is [x; h] concat, outputs BOTH reset and update gate simultaneously.
candidate_conv: input is [x; r⊙h], outputs candidate hidden state.
Output 2*hidden_dim then split → avoids two separate convolutions for r, u.

```python
    def forward(self, x, h, supports):
        # x: (B, N, C_in)  — input at current timestep
        # h: (B, N, hidden_dim)  — previous hidden state
        combined = torch.cat([x, h], dim=-1)   # (B, N, C_in + hidden_dim)
```
Concatenate along feature dimension.
SHAPE: (B, N, in_channels + hidden_dim)

```python
        gates = self.gate_conv(combined, supports)  # (B, N, 2*hidden_dim)
        gates = torch.sigmoid(gates)
        r, u = gates.split(self.hidden_dim, dim=-1)
```
MATH: [r; u] = σ(DiffConv([x; h]))
r → reset gate  (B, N, hidden_dim): how much past to forget
u → update gate (B, N, hidden_dim): how much new vs old state to keep
σ squashes to (0,1) — soft binary gates.

```python
        combined_r = torch.cat([x, r * h], dim=-1)  # (B, N, in+hidden)
        candidate = torch.tanh(
            self.candidate_conv(combined_r, supports)
        )                                             # (B, N, hidden_dim)
```
MATH: c̃ = tanh(DiffConv([x; r⊙h]))
r⊙h: reset gate selectively erases old hidden state before computing candidate.
tanh squashes to (-1, +1).

```python
        h_new = u * h + (1 - u) * candidate          # (B, N, hidden_dim)
        return h_new
```
MATH: h_new = u⊙h + (1-u)⊙c̃
u≈1: keep old hidden state h (memory preserved)
u≈0: replace with new candidate c̃ (memory updated)
FINAL SHAPE: (B, N, hidden_dim) = (128, 207, 64)

```python
    def init_hidden(self, batch_size, num_nodes, device):
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
```
Initialize h_0 = 0 for all nodes. Standard initialization for RNNs.

---

## CLASS: DCRNNEncoder

```python
def forward(self, x_seq, supports):
    # x_seq: (B, T, N, C_in) = (128, 12, 207, 1)
    h_list = [cell.init_hidden(batch, N, device) for cell in self.cells]
    # h_list: list of 2 tensors, each (B, N, 64)
    
    for t in range(seq_len):           # t = 0, 1, ..., 11
        x_t = x_seq[:, t]             # (B, N, C_in) = (128, 207, 1)
        for layer_idx, cell in enumerate(self.cells):
            inp = x_t if layer_idx == 0 else h_list[layer_idx - 1]
            h_list[layer_idx] = cell(inp, h_list[layer_idx], supports)
```
Sequential processing: CANNOT be parallelized across time (GRU is recurrent).
Layer 0: input = x_t (sensor speed), h = previous hidden
Layer 1: input = h from layer 0, h = layer 1's previous hidden
After t=11: h_list holds the final encoder hidden states.
SHAPE h_list: [tensor(B,N,64), tensor(B,N,64)]

```python
    return h_list                      # final hidden states after seeing all 12 steps
```
These encode the entire input sequence into fixed-size representations.

---

## CLASS: DCRNNDecoder

```python
def forward(self, h_list, supports, pred_len, teacher_forcing=None, tf_ratio=0.0):
    decoder_input = torch.zeros(batch, N, self.out_channels, device=device)
    # SHAPE: (B, N, 1) = (128, 207, 1)  — start token is zeros
    outputs = []

    for t in range(pred_len):          # t = 0, 1, ..., 11
        for layer_idx, cell in enumerate(self.cells):
            inp = decoder_input if layer_idx == 0 else h_list[layer_idx - 1]
            h_list[layer_idx] = cell(inp, h_list[layer_idx], supports)
        
        output = self.projection(h_list[-1])   # (B, N, 1)
```
projection: Linear(hidden_dim=64, out_channels=1)
MATH: ŷ_t = W · h_L + b  where h_L is the last layer's hidden state.
SHAPE: (B, N, 1)

```python
        outputs.append(output)
        
        if teacher_forcing is not None and torch.rand(1).item() < tf_ratio:
            decoder_input = teacher_forcing[:, t]   # (B, N, 1) — true target
        else:
            decoder_input = output                  # (B, N, 1) — model's own prediction
```
Teacher forcing: with probability tf_ratio, use true target as next input.
As tf_ratio decays toward 0, model learns to be self-reliant.

```python
    return torch.stack(outputs, dim=1)  # (B, pred_len, N, 1)
```
Stack 12 output tensors along new dim 1.
FINAL SHAPE (before squeeze): (B, 12, N, 1)

---

## CLASS: DCRNN (top-level)

```python
def forward(self, x, supports, targets=None, tf_ratio=0.0):
    # x: (B, T, N) = (128, 12, 207)
    x = x.unsqueeze(-1)               # (B, T, N, 1)  — add feature dim
    h_list = self.encoder(x, supports)
    # h_list: [tensor(B,N,64), tensor(B,N,64)]
    
    tf_targets = targets.unsqueeze(-1) if targets is not None else None
    # tf_targets: (B, T, N, 1) if teacher forcing
    
    output = self.decoder(h_list, supports, self.pred_len, tf_targets, tf_ratio)
    # output: (B, pred_len, N, 1)
    
    return output.squeeze(-1)          # (B, pred_len, N) = (128, 12, 207)
```
FINAL SHAPE: (B, pred_len, N) = (128, 12, 207) — matches STGCN output format ✓

---

## Parameter Count Breakdown

STGCN (N=207):
  ChebGraphConv Block1: weight=(4,16,16) + bias=16 → 1,040
  ChebGraphConv Block2: weight=(4,32,32) + bias=32 → 4,128
  TemporalConv layers: Conv2d weights → ~45,000
  Output conv + linear: ~29,000
  TOTAL: ~79,500

DCRNN (N=207):
  Each DiffusionConv: weight=(5, C_in+hidden, 2*hidden) → large
  gate_conv Block1: (5, 1+64, 128) = 5×65×128 = 41,600 + bias
  candidate_conv Block1: (5, 65, 64) = 20,800 + bias
  Layer 2 (hidden→hidden): similar but in_channels=64
  Decoder: same structure × 2 layers + projection
  TOTAL: ~371,000 (4.7× larger than STGCN)
