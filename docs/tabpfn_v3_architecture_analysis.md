# TabPFN v3 Architecture Analysis

Source: [`PriorLabs/TabPFN`](https://github.com/PriorLabs/TabPFN) @ `/tmp/TabPFN/src/tabpfn/architectures/tabpfn_v3.py` (2,395 lines, single-file architecture)

---

## 1. Module Hierarchy

```
TabPFNV3 (Architecture)
├── TorchStandardScaler              # Stage 0: per-feature z-score normalization
├── x_embed: Linear                  # Stage 1: cell embedding (G → E)
├── col_y_encoder                    # Stage 1: target-aware column embedding
│   ├── TrainableOrthogonalEmbedding   (multiclass)
│   └── Linear(1, E)                   (regression)
├── feature_distribution_embedder: FeatureDistributionEmbedder  # Stage 2a
│   └── layers: ModuleList[InducedSelfAttentionBlock × dist_embed_num_blocks]
│       ├── cross_attn_block1: CrossAttentionBlock  (inducing → data)
│       │   ├── attn: CrossAttention
│       │   ├── mlp: MLP
│       │   └── layernorm_q, layernorm_kv, layernorm2: RMSNorm
│       ├── cross_attn_block2: CrossAttentionBlock  (data → inducing hidden)
│       └── inducing_vectors: Parameter(n_ind, E)
├── column_aggregator: ColumnAggregator             # Stage 2b
│   ├── blocks: ModuleList[TransformerBlock × feat_agg_num_blocks]
│   │   ├── attention: Attention (self-attn + RoPE)
│   │   ├── mlp: MLP
│   │   └── layernorm, layernorm_mlp: RMSNorm
│   ├── cls_tokens: Parameter(num_cls_tokens, E)
│   ├── rope: RotaryEmbedding
│   └── out_ln: RMSNorm
├── icl_y_encoder                    # Stage 3: ICL target embedding
│   ├── TrainableOrthogonalEmbedding   (multiclass)
│   └── Linear(1, icl_emsize)          (regression)
├── icl_blocks: ModuleList[ICLTransformerBlock × nlayers]  # Stage 3
│   ├── icl_attention: ICLAttention (GQA, KV cache, split train/test)
│   │   ├── q_projection: Linear(E, H*D, bias=False)
│   │   ├── k_projection: Linear(E, Hkv*D, bias=False)
│   │   ├── v_projection: Linear(E, Hkv*D, bias=False)
│   │   ├── out_projection: Linear(H*D, E, bias=False)
│   │   └── softmax_scaling_layer: SoftmaxScalingMLP
│   ├── mlp: MLP (Sequential: Linear→GELU→Linear)
│   └── layernorm, layernorm_mlp: RMSNorm
├── output_norm: RMSNorm
├── many_class_decoder: ManyClassDecoder  (multiclass)
│   ├── q_projection: Linear(E, attn_size)
│   └── k_projection: Linear(E, attn_size)
└── output_projection: Sequential     (regression)
    ├── Linear(icl_emsize, icl_emsize * ff_factor)
    ├── GELU
    └── Linear(icl_emsize * ff_factor, n_out)
```

---

## 2. Default Configuration (TabPFNV3Config)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 128 | Base embedding dimension |
| `dist_embed_num_blocks` | 3 | Distribution embedder layers |
| `dist_embed_num_heads` | 8 | Heads in distribution embedder |
| `dist_embed_num_inducing_points` | 128 | Inducing points per block |
| `feature_group_size` | 3 | Features per circular-shift group |
| `feat_agg_num_blocks` | 3 | Column aggregator layers |
| `feat_agg_num_heads` | 8 | Heads in column aggregator |
| `feat_agg_num_cls_tokens` | 4 | CLS tokens for aggregation |
| `feat_agg_rope_base` | 100,000 | RoPE base frequency |
| `nlayers` | 24 | ICL transformer blocks |
| `icl_num_heads` | 8 | ICL attention heads |
| `icl_num_kv_heads` | None (=8) | GQA KV heads (None = standard MHA) |
| `icl_num_kv_heads_test` | None | Separate GQA for test rows |
| `decoder_head_dim` | 64 | Many-class decoder head dim |
| `decoder_num_heads` | 6 | Many-class decoder heads |
| `ff_factor` | 2 | MLP expansion factor |
| `use_nan_indicators` | True | Doubles cell embedding input |
| `inference_row_chunk_size` | 2048 | Row chunking for memory |
| `inference_col_chunk_size` | 4 | Column chunking for memory |

**Derived dimensions:**
- `icl_emsize = embed_dim × feat_agg_num_cls_tokens = 128 × 4 = 512`
- `head_dim = embed_dim / num_heads = 128 / 8 = 16` (distribution/aggregator)
- `icl_head_dim = icl_emsize / icl_num_heads = 512 / 8 = 64` (ICL)
- MLP hidden: `emsize × ff_factor` (256 for stages 2a/2b, 1024 for ICL)

---

## 3. Forward Pass: Layer-by-Layer with Tensor Shapes

### Shape Convention
```
B: batch size
R: total rows (N_train + M_test)
Ri: input rows (could be R or M depending on cache)
N: train rows
M: test rows
C: total columns (after feature grouping: ceil(raw_features / group_size))
G: feature_group_size (3) or 2*G (6) if nan_indicators
E: embed_dim (128)
D: icl_emsize (512)
Cl: num_cls_tokens (4)
H: num_heads
Hkv: num_kv_heads
```

### Stage 0: Preprocessing

```python
# Input: x_RiBC of shape (Ri, B, C_raw) — note: rows-first, transposed later
# 1. Standard scaling (z-score per feature, fitted on train rows)
x_scaled = standard_scaler.transform(x_RiBC)  # (Ri, B, C_raw)
# 2. NaN indicator encoding (if use_nan_indicators)
nan_mask = isnan(x_scaled)  # (Ri, B, C_raw)
x_with_nan = cat([nan_to_num(x_scaled), nan_mask], dim=-1)  # (Ri, B, C_raw*2)
# 3. Feature grouping: circular shifts produce groups of size G
#    Groups columns into non-overlapping windows of feature_group_size
x_grouped = reshape_to_groups(x_with_nan)  # (B, Ri, C, G) where G=6 with nan
```

### Stage 1: Cell Embedding + Target-Aware Column Embedding

```python
# Cell embedding: Linear(G, E)
x_BRiCE = x_embed(x_grouped_BRiCG)  # (B, Ri, C, E=128)

# Target-aware column embedding (added to train rows only):
# multiclass: col_y_encoder(y) → (B, N, E)
# regression: col_y_encoder(y.unsqueeze(-1)) → (B, N, E)
y_col_emb = col_y_encoder(y)  # (B, N, E=128)
x_BRiCE[:, :N] += y_col_emb.unsqueeze(2)  # broadcast across C columns
```

### Stage 2a: Feature Distribution Embedding (InducedSelfAttention)

```python
# Per-column induced self-attention (SetTransformer style)
# Reduces O(R²) to O(R × n_inducing)
# Input: (B, Ri, C, E=128)
# Internally reshapes to (B*C, Ri, E) for per-column processing

for block in feature_distribution_embedder.layers:  # × 3 blocks
    # Step 1: Inducing points attend to train data
    ind = inducing_vectors.expand(B*C, n_ind=128, E)  # (B*C, 128, 128)
    hidden = cross_attn_block1(ind, x_flat[:, :N])    # (B*C, 128, 128)
    #   Q=ind, K/V=train_data → inducing hidden states
    
    # Step 2: All rows attend to inducing hidden states
    x_flat = cross_attn_block2(x_flat, hidden)        # (B*C, Ri, 128)
    #   Q=all_rows, K/V=hidden → updated embeddings

# Output: (B, Ri, C, E=128)
```

### Stage 2b: Column Aggregation (ColumnAggregator)

```python
# Cross-feature interaction with learnable CLS tokens
# Input: (B, Ri, C, E=128)

# Prepend CLS tokens along column axis
cls = cls_tokens.expand(B, Ri, Cl=4, E)         # (B, Ri, 4, 128)
x = cat([cls, x_BRiCE], dim=2)                  # (B, Ri, C+4, 128)

# Self-attention over features (with RoPE)
for block in blocks[:-1]:  # × 2 blocks
    x = block(x, rope=rope)                      # (B, Ri, C+4, 128)
    # attention: (B*Ri, C+4, 128) → SDPA with RoPE
    # mlp: (B*Ri*C+4, 128) → (256) → (128)

# Last block: CLS readout (cross-attention)
cls_out = blocks[-1].forward_cross(
    x[..., :Cl, :],  # query = CLS tokens: (B, Ri, 4, 128)
    x,               # key/value = full seq: (B, Ri, C+4, 128)
    rope=rope,
)
cls_out = out_ln(cls_out)                        # (B, Ri, Cl=4, E=128)
```

### Stage 2→3 Transition: Flatten CLS tokens

```python
x_BRiD = cls_out.flatten(-2)  # (B, Ri, Cl*E) = (B, Ri, 512)
```

### Stage 3: ICL Transformer

```python
# Add ICL target embedding to train rows
y_icl_emb = icl_y_encoder(y)                    # (B, N, D=512)
x_BRiD[:, :N] += y_icl_emb

# ICL transformer blocks with train-only K/V (× 24 layers)
for block in icl_blocks:
    # Pre-norm attention
    x_normed = layernorm(x_BRiD)                # (B, R, 512)
    q = q_projection(x_normed).view(B, R, H=8, 64)   # (B, R, 8, 64)
    
    # K/V computed from TRAIN rows only
    k = k_projection(x_normed[:, :N]).view(B, N, Hkv, 64)  # (B, N, Hkv, 64)
    v = v_projection(x_normed[:, :N]).view(B, N, Hkv, 64)  # (B, N, Hkv, 64)
    
    # Optional: test rows use fewer KV heads (GQA for test only)
    if num_kv_heads_test is not None:
        out_train = sdpa(q[:, :N], k, v)          # full KV heads
        out_test = sdpa(q[:, N:], k[:,:,:nh_test], v[:,:,:nh_test])  # fewer KV heads
        attn_out = cat([out_train, out_test], dim=1)
    else:
        attn_out = sdpa(q, k, v)                  # (B, R, 8, 64)
    
    out = out_projection(attn_out.reshape(B, R, 512))
    x_BRiD = x_BRiD + out                        # residual
    
    # Pre-norm MLP
    mlp_out = mlp(layernorm_mlp(x_BRiD))         # (B, R, 512)→(1024)→(512)
    x_BRiD = x_BRiD + mlp_out                    # residual

x_BRiD = output_norm(x_BRiD)                     # (B, R, 512)
```

### Stage 4: Decoder

```python
# Split train/test embeddings
train_emb = x_BRiD[:, :N]  # (B, N, 512)
test_emb = x_BRiD[:, N:]   # (B, M, 512)

# MULTICLASS: Attention-based retrieval decoder
q = q_projection(test_emb).view(B, M, 6, 64)        # (B, M, 6, 64)
k = k_projection(train_emb).view(B, N, 6, 64)       # (B, N, 6, 64)
one_hot_v = F.one_hot(y, num_classes).expand(B, N, 6, T)  # (B, N, 6, T)
# Chunked SDPA with values = one-hot targets
attn_out = chunked_class_attention(q, k, one_hot_v)  # (B, M, 6, T)
logits = log(clamp(attn_out.mean(dim=2), min=1e-5) + 3e-5)  # (M, B, T)

# REGRESSION: MLP decoder
logits = output_projection(test_emb)  # (B, M, n_buckets)
# Bar distribution: logits → bucket probabilities → expected value
```

---

## 4. Key Architectural Differences from nanoTabPFN

| Aspect | nanoTabPFN | TabPFN v3 |
|--------|-----------|-----------|
| **Total layers** | 3 | 24 ICL + 3 dist_embed + 3 col_agg = 30 |
| **Embedding dim** | 96 | 128 (base) → 512 (ICL) |
| **Normalization** | LayerNorm (post-norm) | RMSNorm (pre-norm) |
| **Attention type** | Standard MHA with bias | Separate Q/K/V projections, no bias |
| **Feature attention** | Direct self-attention over columns | Induced self-attention (O(n) via inducing points) |
| **Datapoint attention** | Split-query (train self-attn + test cross-attn) | ICLAttention (K/V restricted to train, single forward) |
| **Positional encoding** | None | RoPE in aggregator, SoftmaxScaling MLPs |
| **GQA** | No | Yes (separate KV heads for test rows) |
| **KV cache** | No | Multi-level: ICL + inducing_hidden + scaler stats |
| **Cell embedding** | Linear(1, E) per feature | Linear(G, E) with feature grouping + NaN indicators |
| **Target encoding** | Scalar linear | Orthogonal embedding (multiclass) or linear |
| **Decoder** | 2-layer MLP on target column | Attention-based retrieval (multiclass) or MLP |
| **Regression** | N/A | Bar distribution (bucket discretization) |
| **MLP structure** | Linear→GELU→Linear with bias | Linear→GELU→Linear, no bias, zero-init output |
| **Memory optimization** | None | Chunked eval, activation checkpointing, OOM fallback |
| **Feature grouping** | 1 feature per cell | Circular-shift groups of 3 |
| **torch.compile** | No | Yes (optional, with dynamic shapes) |
| **Parameters** | ~356K | ~50M+ estimated |

---

## 5. Custom Ops Requiring MLX Equivalents

### PyTorch Operations Used in Forward Pass

| PyTorch Op | Usage | MLX Equivalent | Difficulty |
|------------|-------|----------------|------------|
| `F.scaled_dot_product_attention` | All attention layers | `mx.fast.scaled_dot_product_attention` | Easy (already used in existing MLX backend) |
| `F.rms_norm` | `_DtypeMatchingRMSNorm` | `mx.fast.rms_norm` or manual | Easy |
| `F.gelu` | All MLPs | `nn.gelu` | Trivial |
| `F.one_hot` | Many-class decoder values | Manual scatter or `mx.one_hot` (if available) | Medium |
| `F.pad` | Chunked class attention, MLX backend | `mx.pad` | Easy |
| `torch.linalg.qr` | Orthogonal embedding init | Not needed at inference (init only) | N/A |
| `torch.nan_to_num` | NaN-safe output | `mx.where(mx.isnan(...), 0, ...)` | Easy |
| `torch.clamp` | Decoder log safety | `mx.clip` | Trivial |
| `torch._dynamo.mark_dynamic` | torch.compile hints | No equivalent (skip) | N/A |
| `torch.utils.checkpoint.checkpoint` | Activation checkpointing | Not applicable for inference | N/A |
| `nn.init.trunc_normal_` | Weight init | Not needed at inference | N/A |
| `nn.init.xavier_uniform_` | Weight init | Not needed at inference | N/A |
| `nn.init.zeros_` | Weight init | Not needed at inference | N/A |
| `torch.compile` | JIT compilation | MLX uses lazy eval + `mx.eval()` | Different paradigm |
| `.contiguous()` | Memory layout | No-op in MLX (row-major by default) | Trivial |
| `.view() / .reshape()` | Tensor reshaping | `mx.reshape` | Trivial |
| `.transpose()` | Dimension permutation | `mx.transpose` | Trivial |
| `.expand()` | Broadcasting | `mx.broadcast_to` | Easy |
| `.unsqueeze() / .squeeze()` | Dimension manipulation | `mx.expand_dims` / `mx.squeeze` | Trivial |
| `torch.cat` | Concatenation | `mx.concatenate` | Trivial |
| `torch.stack` | Stacking | `mx.stack` | Trivial |
| `torch.arange` | Sequence generation | `mx.arange` | Trivial |
| `torch.log` | Decoder logits | `mx.log` | Trivial |
| `torch.tanh` | SoftmaxScaling modulation | `mx.tanh` | Trivial |
| `torch.cos / torch.sin` | RoPE | `mx.cos` / `mx.sin` | Trivial |
| `.repeat_interleave` | RoPE interleaved mode | Manual with reshape + broadcast | Medium |
| `.flatten(-2)` | CLS token flattening | `mx.reshape` | Trivial |
| `nn.Embedding` | Class label lookup | `nn.Embedding` (MLX has this) | Easy |
| `nn.Parameter` | Learnable parameters (inducing vectors, CLS tokens, RoPE freqs) | Direct array assignment in `__init__` | Easy |

### Non-Trivial Patterns Requiring Careful Porting

1. **RoPE (Rotary Positional Embeddings)**: The `apply_rope` function uses interleaved or split-half rotation. MLX has `mx.fast.rope` but the interface differs — need to verify it matches the non-interleaved split-half pattern used here.

2. **SoftmaxScalingMLP**: Query-dependent attention scaling — a learned multiplier applied to queries before SDPA. This is a novel architectural choice not present in standard transformers.

3. **Chunked Class Attention**: The many-class decoder folds class dimension chunks into the batch dimension for a single SDPA call. Complex reshaping logic but pure tensor ops.

4. **GQA with asymmetric train/test heads**: ICLAttention uses different numbers of KV heads for train vs test rows within the same forward pass. Requires splitting the computation.

5. **KV Cache multi-level**: Three levels of caching (ICL KV, inducing hidden states, scaler statistics) with device movement and dtype matching.

6. **Induced Self-Attention**: SetTransformer-style O(n) attention via learnable inducing points — two sequential cross-attention operations.

7. **Feature Grouping with Circular Shifts**: Non-obvious preprocessing that groups features into windows of 3, doubling with NaN indicators.

---

## 6. Component Difficulty Assessment

| Component | Difficulty | Rationale |
|-----------|-----------|-----------|
| **MLP (Linear→GELU→Linear)** | Easy | Direct MLX equivalent, no bias |
| **RMSNorm** | Easy | `mx.fast.rms_norm` or `nn.RMSNorm` |
| **Attention (self-attn with RoPE)** | Easy-Medium | Standard pattern, RoPE needs verification |
| **CrossAttention** | Easy | Same as self-attn but separate Q source |
| **ICLAttention (train-only K/V)** | Medium | Split logic + GQA + KV cache |
| **RotaryEmbedding** | Medium | Need to match exact rotation pattern; MLX has `mx.fast.rope` |
| **SoftmaxScalingMLP** | Medium | Novel, but just learned scaling of queries |
| **InducedSelfAttentionBlock** | Medium | Two-stage cross-attention with learnable inducing points |
| **ColumnAggregator** | Medium | CLS token handling + cross-attn readout |
| **FeatureDistributionEmbedder** | Medium | Stack of ISA blocks with caching |
| **KV Cache System** | Medium-Hard | Multi-level, device movement, dtype matching |
| **ManyClassDecoder** | Medium-Hard | Chunked class attention with batch-folding |
| **TabPFNV3 (full model)** | Hard | Orchestrating all stages, chunked inference, OOM handling |
| **Feature Grouping + NaN Encoding** | Easy-Medium | Preprocessing logic, circular shifts |
| **Standard Scaler** | Easy | Mean/std computation and caching |
| **torch.compile integration** | N/A | Not applicable to MLX (different JIT model) |
| **Activation Checkpointing** | N/A | Inference-only port doesn't need this |
| **OOM Fallback Logic** | Low priority | MLX unified memory reduces OOM risk |

---

## 7. Recommended Porting Order

### Phase 1: Foundation (Low Risk, High Reuse)

1. **RMSNorm** — Used everywhere, easy to implement
2. **MLP** — GELU feed-forward, zero-init output weight
3. **RotaryEmbedding** — Implement and test against PyTorch reference
4. **Attention (self-attention)** — Standard MHA with RoPE + SDPA
5. **CrossAttention** — Extends Attention with separate Q/KV inputs

### Phase 2: Distribution Embedding Pipeline

6. **InducedSelfAttentionBlock** — Two cross-attention blocks + inducing vectors
7. **FeatureDistributionEmbedder** — Stack of ISA blocks
8. **CrossAttentionBlock** — Pre-norm cross-attention + MLP wrapper

### Phase 3: Column Aggregation

9. **TransformerBlock** — Self-attention + MLP with RoPE
10. **ColumnAggregator** — CLS tokens + transformer stack + cross-attn readout

### Phase 4: ICL Transformer

11. **SoftmaxScalingMLP** — Query-dependent attention scaling
12. **ICLAttention** — Train-only K/V with optional GQA split
13. **ICLTransformerBlock** — Full block with cache support

### Phase 5: Decoder + Integration

14. **TrainableOrthogonalEmbedding** — Label embedding (load weights only)
15. **ManyClassDecoder** — Chunked class attention retrieval
16. **TorchStandardScaler** — Z-score normalization
17. **Feature grouping + NaN encoding** — Preprocessing
18. **TabPFNV3** — Full model orchestration

### Phase 6: Optimization (Optional)

19. **KV Cache System** — Multi-level caching for efficient inference
20. **Chunked Evaluation** — Row/column chunking for large datasets
21. **GQA test-set optimization** — Asymmetric KV heads

---

## 8. Existing MLX Backend (What's Already Done)

File: `/tmp/TabPFN/src/tabpfn/architectures/shared/mlx_backend.py` (98 lines)

The official repo already includes a **minimal MLX backend** that:
- Routes `scaled_dot_product_attention` to `mx.fast.scaled_dot_product_attention` when on MPS
- Handles bfloat16 via int16 view conversion
- Pads head_dim to 64 or 128 (Flash attention constraint)
- Converts PyTorch tensors to/from MLX arrays via numpy

This is **attention-only** — it does not port any model components. It's used as a performance optimization within the PyTorch model, not as a standalone inference engine.

Key insight: Our port will run the entire model natively in MLX, eliminating the PyTorch↔MLX conversion overhead.

---

## 9. Risk Assessment and Open Questions

### High Risk
- **Numerical equivalence across 24 ICL layers**: Small per-layer drift compounds. nanoTabPFN (3 layers) achieved 3.58e-7 max diff — v3 will be harder.
- **RoPE implementation correctness**: The split-half rotation pattern must exactly match or attention patterns diverge catastrophically.
- **SoftmaxScalingMLP**: Novel learned scaling — any sign error in the query modulation breaks attention completely.

### Medium Risk
- **Induced self-attention**: Two-stage mechanism with detached hidden states — gradient flow doesn't matter for inference but the order of operations is subtle.
- **Many-class decoder chunking**: Complex reshape logic that folds class dim into batch dim — easy to get wrong but purely mechanical.
- **Feature grouping circular shifts**: Preprocessing must exactly match or all downstream computation is wrong.

### Low Risk
- **Standard linear/MLP/norm layers**: Direct equivalents exist in MLX.
- **KV cache**: Conceptually simple (store tensors, pass them later), just requires careful API design.

### Open Questions
1. What pretrained checkpoint format does the official model use? (likely safetensors based on HuggingFace hosting)
2. Does the model use different configs for different dataset sizes? (inference_row_chunk_size suggests adaptive)
3. Is `max_num_classes` a fixed architectural parameter or runtime? (affects decoder construction)
4. How are the regression borders (bar distribution) defined? (spline-based, stored as buffer)

---

## 10. Estimated Effort

| Phase | Components | Estimated LOC | Time |
|-------|-----------|---------------|------|
| Phase 1 (Foundation) | Norm, MLP, RoPE, Attention | ~300 | 1-2 days |
| Phase 2 (Distribution) | ISA, CrossAttn blocks | ~200 | 1 day |
| Phase 3 (Aggregation) | ColumnAggregator | ~150 | 1 day |
| Phase 4 (ICL) | ICLAttention, blocks | ~250 | 1-2 days |
| Phase 5 (Decoder + Integration) | Decoder, preprocessing, full model | ~400 | 2-3 days |
| Phase 6 (Optimization) | KV cache, chunking | ~200 | 1-2 days |
| **Testing + Validation** | Per-layer numerical checks | ~500 | 2-3 days |
| **Total** | | ~2000 | ~10-14 days |

This assumes an inference-only port (no training support, no torch.compile, no activation checkpointing). Adding training support would roughly double the effort.
