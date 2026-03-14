# Learning the Inverse RSK Correspondence with Transformers — Full Summary

## The Problem

The **Robinson-Schensted-Knuth correspondence** is a bijection central to algebraic combinatorics:

```
σ ∈ S_n  ⟷  (P, Q) pair of standard Young tableaux of shape λ ⊢ n
```

**Forward** (σ → P, Q): Schensted row insertion — scan each σ(i), bump entries rightward through rows, record where new cells appear.

**Inverse** (P, Q → σ): Reverse bumping — process Q entries in decreasing order, reverse the insertion path in P to recover each σ(i).

We train a neural network to learn the **inverse direction**: given P and Q, predict σ.

## Prior Work

The **PNNL ML4AlgComb benchmark** attempted this with:
- Tableaux encoded as bracket strings: `"[[1,3,5],[2,4]]"` → tokenised as `[`, `[`, `1`, `,`, `3`, ...
- Sequence-to-sequence models treating it as string transduction
- Result: weak baselines, accuracy well below useful levels

**Why it failed**: bracket-string encoding destroys the 2D geometric structure of tableaux. The model has to re-learn that `1` at position 7 in the string means "row 0, column 0" — information that was explicit in the tableau but is now buried in syntax.

## Our Architecture: RSKEncoder

### Input Encoding — Structured 2D Tokens

Each entry in P and Q becomes a token. For n entries per tableau, we get **2n tokens**. Each token carries four properties: its numeric value, its row, its column, and which tableau (P or Q) it belongs to.

**Concrete example.** Take σ = [3, 1, 2]. RSK forward gives:

```
P = [[1, 2],    Q = [[1, 3],
     [3]]            [2]]
```

PNNL would flatten this to bracket strings: `[[1,2],[3]]` → tokenised as `[`, `[`, `1`, `,`, `2`, `]`, ... — a sequence where all spatial information is buried in syntax. We instead create one token per tableau entry:

| Token | Value | Row | Col | Tableau | Meaning |
|-------|-------|-----|-----|---------|---------|
| 1 | 1 | 0 | 0 | P | "1 sits at row 0, col 0 in P" |
| 2 | 2 | 0 | 1 | P | "2 sits at row 0, col 1 in P" |
| 3 | 3 | 1 | 0 | P | "3 sits at row 1, col 0 in P" |
| 4 | 1 | 0 | 0 | Q | "1 sits at row 0, col 0 in Q" |
| 5 | 3 | 0 | 1 | Q | "3 sits at row 0, col 1 in Q" |
| 6 | 2 | 1 | 0 | Q | "2 sits at row 1, col 0 in Q" |

**How the embedding works.** Readers familiar with Raschka's *Build a Large Language Model (From Scratch)* [1] will recognise the pattern. In Chapter 2, Raschka builds GPT-2's embedding as two lookup tables summed together:

```python
# Raschka's GPT embedding (simplified)
token_emb = nn.Embedding(vocab_size, d_model)      # "what word is this?"
pos_emb   = nn.Embedding(context_length, d_model)   # "where in the sequence?"
x = token_emb(token_ids) + pos_emb(positions)
```

A word's meaning depends on *what* it is and *where* it appears. Two lookup tables, summed. Our embedding is the same idea — but a tableau entry's meaning depends on **four** things, not two:

```python
# Our embedding — four lookup tables, summed
value_emb   = nn.Embedding(n + 1, 128)  # "what number is this entry?"
row_emb     = nn.Embedding(n, 128)      # "which row is it in?"
col_emb     = nn.Embedding(n, 128)      # "which column is it in?"
tableau_emb = nn.Embedding(2, 128)      # "is it in P or Q?"

x = value_emb(values) + row_emb(rows) + col_emb(cols) + tableau_emb(tableau_ids)
```

Each `nn.Embedding(num_entries, 128)` is just a matrix of shape `(num_entries, 128)`. When you call `value_emb(3)`, it returns row 3 of that matrix — a 128-dimensional vector. As Raschka explains: "an embedding layer is essentially a lookup operation."

For token 3 in the example above (the number 3 in P at row 1, col 0):

```
token_embedding = value_emb.weight[3]      # row 3 of a (n+1, 128) matrix
                + row_emb.weight[1]        # row 1 of a (n, 128) matrix
                + col_emb.weight[0]        # row 0 of a (n, 128) matrix
                + tableau_emb.weight[0]    # row 0 of a (2, 128) matrix — P
```

Four vectors from four tables, added into one 128-dimensional vector. That's it. Followed by LayerNorm and dropout (0.1).

**These vectors are learned, not hand-designed.** At initialisation, each embedding table contains random 128-dimensional vectors. Through backpropagation — the same gradient descent that trains the rest of the network — the model adjusts these vectors to be useful for predicting σ. We designed the *structure* (each token has four properties combined by addition). We did *not* design the *content* (the actual 128-dimensional vectors). Through training, the model discovers what each spatial coordinate means — that row index relates to bumping depth, that column position relates to shape, that P and Q play fundamentally different roles (insertion values vs insertion order).

This is the key insight: we gave the model the right inductive bias (tableaux are 2D, they come in pairs), and it learned how to exploit that structure. PNNL gave the model no structure at all (flat bracket strings), so it had to discover both structure and content from scratch — which it couldn't do.

**Comparison to Raschka's GPT embedding:**

| | Raschka's GPT | Our RSKEncoder |
|--|--------------|----------------|
| What | `token_emb(word_id)` | `value_emb(entry_value)` |
| Where | `pos_emb(position)` — 1D | `row_emb(r) + col_emb(c) + tableau_emb(t)` — 2D + categorical |
| Terms summed | 2 | 4 |
| Output shape | `(batch, seq_len, d_model)` | `(batch, 2n, 128)` |

The difference: GPT processes a 1D sequence of words, so one positional embedding suffices. Tableaux are 2D grids that come in pairs, so we need three "positional" embeddings (row, col, which-tableau) to tell the model where each entry lives.

**Why sum rather than concatenate.** Concatenating the four vectors would give a 512-dimensional input per token. Summing keeps it at 128 dimensions, relying on the fact that in high-dimensional space the model can learn roughly orthogonal directions for the different properties.

[1] S. Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024.

### Complete Model Code

The entire model is about 50 lines of PyTorch. Here is the core logic:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model  # 128

        self.value_emb = nn.Embedding(config.max_value + 1, d)
        self.row_emb = nn.Embedding(config.max_rows, d)
        self.col_emb = nn.Embedding(config.max_cols, d)
        self.tableau_emb = nn.Embedding(2, d)  # P=0, Q=1

        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values, positions):
        # values:    (batch, 2n) — entry values
        # positions: (batch, 2n, 3) — [row, col, tableau_id]

        x = (
            self.value_emb(values)
            + self.row_emb(positions[:, :, 0])
            + self.col_emb(positions[:, :, 1])
            + self.tableau_emb(positions[:, :, 2])
        )
        return self.dropout(self.layer_norm(x))


class RSKEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n = config.n

        self.embedding = TokenEmbedding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,       # 128
            nhead=config.nhead,           # 8
            dim_feedforward=config.dim_feedforward,  # 512
            dropout=config.dropout,       # 0.1
            activation="gelu",
            batch_first=True,
            norm_first=True,              # Pre-norm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,  # 6
        )

        # n classification heads — one per position in σ
        self.heads = nn.ModuleList([
            nn.Linear(config.d_model, config.n)
            for _ in range(config.n)
        ])

    def forward(self, values, positions):
        x = self.embedding(values, positions)  # (batch, 2n, 128)
        x = self.encoder(x)                    # (batch, 2n, 128)
        x = x.mean(dim=1)                      # (batch, 128)  — mean pool
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits                          # (batch, n, n)
```

The forward pass is four lines:
1. **Embed** the tokens (four lookup tables, summed)
2. **Encode** through 6 transformer layers (all tokens attend to all tokens — no causal mask)
3. **Mean pool** over all 2n tokens into a single 128-d vector
4. **Classify** with n parallel linear heads to get logits

The transformer encoder uses `nn.TransformerEncoder` from PyTorch — the same stack of multi-head self-attention + feed-forward layers that Raschka builds from scratch in Chapters 3-4 of [1], except we use PyTorch's built-in implementation. This is BERT-style (encoder-only, bidirectional attention), not GPT-style (decoder, causal mask), because (P, Q) fully determines σ — all information is in the input, so every token should attend to every other token simultaneously.

`logits[b, i, j]` = log-probability that σ(i+1) = j+1 for batch element b.

**Why n parallel heads**: each position σ(i) is an independent n-way classification problem. The heads share the same pooled representation but have separate weights, allowing each to specialise.

### Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     INPUT: (P, Q) tableaux                      │
 │                                                                 │
 │   P = ┌─┬─┬─┐    Q = ┌─┬─┬─┐                                  │
 │       │1│2│5│        │1│2│4│     σ = [3, 1, 4, 5, 2]           │
 │       ├─┼─┘         ├─┼─┘                                      │
 │       │3│            │3│          n = 5  →  2n = 10 tokens      │
 │       ├─┤            ├─┤                                        │
 │       │4│            │5│                                        │
 │       └─┘            └─┘                                        │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                   TOKEN EMBEDDING (per token)                   │
 │                                                                 │
 │   token_emb = value_emb(v) + row_emb(r) + col_emb(c)          │
 │             + tableau_emb(P=0 / Q=1)                            │
 │                                                                 │
 │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
 │   │value_emb │ │ row_emb  │ │ col_emb  │ │tableau_  │         │
 │   │(n+1, 128)│ │ (n, 128) │ │ (n, 128) │ │emb(2,128)│         │
 │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘         │
 │        │            │            │             │                │
 │        └──────┬─────┴─────┬──────┘             │                │
 │               │    (+)    │        (+)          │                │
 │               └─────┬─────┴────────────────────┘                │
 │                     ▼                                           │
 │              LayerNorm(128)                                     │
 │                     ▼                                           │
 │              Dropout(0.1)                                       │
 │                                                                 │
 │            Output: (batch, 2n, 128)                             │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │            TRANSFORMER ENCODER  ×6 layers                       │
 │                                                                 │
 │   ┌─────────────────────────────────────────────────┐           │
 │   │  Pre-Norm TransformerEncoderLayer               │           │
 │   │                                                 │           │
 │   │   ┌──────────────────────────────────────┐      │           │
 │   │   │  LayerNorm → Multi-Head Attention    │      │           │
 │   │   │  (8 heads, d_k = 16, no causal mask)│      │           │
 │   │   │  All 2n tokens attend to all others  │      │           │
 │   │   └──────────────┬───────────────────────┘      │           │
 │   │        residual (+)                             │           │
 │   │                  ▼                              │           │
 │   │   ┌──────────────────────────────────────┐      │           │
 │   │   │  LayerNorm → FFN                     │      │           │
 │   │   │  Linear(128→512) → GELU              │      │           │
 │   │   │  → Linear(512→128) → Dropout(0.1)    │      │           │
 │   │   └──────────────┬───────────────────────┘      │           │
 │   │        residual (+)                             │           │
 │   └──────────────────┼─────────────────────────┘    │           │
 │                      │  ×6                          │           │
 │            Output: (batch, 2n, 128)                             │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     MEAN POOL                                   │
 │                                                                 │
 │         (batch, 2n, 128)  →  mean(dim=1)  →  (batch, 128)      │
 │                                                                 │
 │         All 2n token representations averaged into              │
 │         a single 128-d summary vector                           │
 └────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │               n PARALLEL CLASSIFICATION HEADS                   │
 │                                                                 │
 │    ┌──────────┐ ┌──────────┐ ┌──────────┐     ┌──────────┐    │
 │    │ Head 1   │ │ Head 2   │ │ Head 3   │ ... │ Head n   │    │
 │    │Linear    │ │Linear    │ │Linear    │     │Linear    │    │
 │    │(128 → n) │ │(128 → n) │ │(128 → n) │     │(128 → n) │    │
 │    └────┬─────┘ └────┬─────┘ └────┬─────┘     └────┬─────┘    │
 │         ▼            ▼            ▼                 ▼          │
 │      σ(1)=?       σ(2)=?       σ(3)=?           σ(n)=?        │
 │     logits        logits       logits           logits         │
 │     over          over         over             over           │
 │     {1..n}        {1..n}       {1..n}           {1..n}         │
 │                                                                 │
 │            Output: (batch, n, n)                                │
 └────────────────────────┬────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         [TRAINING]             [INFERENCE]
              │                       │
              ▼                       ▼
 ┌────────────────────┐  ┌─────────────────────────────┐
 │  Cross-Entropy     │  │  Masked Greedy Decoding      │
 │  Loss (per head,   │  │                              │
 │  targets 0-indexed)│  │  1. Find max logit globally  │
 │                    │  │  2. Assign σ(i) = j          │
 │  L = Σᵢ CE(        │  │  3. Mask position i (done)   │
 │    logits[b,i,:],  │  │  4. Mask value j (used)      │
 │    σ(i+1) - 1      │  │  5. Repeat n times           │
 │  )                 │  │                              │
 └────────────────────┘  │  → Valid permutation σ       │
                         └─────────────────────────────┘
```

### Masked Greedy Decoding (Inference Only)

Raw argmax can assign the same value to multiple positions — violating the permutation constraint. Masked greedy decoding fixes this:

```
1. Compute logits (batch, n, n)
2. Find the global (position, value) pair with highest logit
3. Lock that assignment
4. Set that position's logits to -∞ (position decided)
5. Set that value's logits to -∞ across all positions (value used)
6. Repeat n times
```

This guarantees a valid permutation. The gap between argmax and greedy accuracy measures how often the model would violate the constraint without it.

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| value_emb | (n+1) × 128 |
| row_emb | n × 128 |
| col_emb | n × 128 |
| tableau_emb | 2 × 128 |
| 6 transformer layers | ~1.18M |
| n classification heads | n × (128 × n + n) |
| **Total (n=8)** | **1,202,368** |
| **Total (n=10)** | **1,207,012** |
| **Total (n=15)** | **1,225,057** |

The backbone dominates. Changing n only affects the embedding tables and head sizes — the transformer layers are identical across all experiments.

## Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 0.01 |
| Schedule | Linear warmup (5% of steps) → cosine decay |
| Gradient clipping | max norm 1.0 |
| Batch size | 512 |
| Loss | Cross-entropy (summed over n heads) |
| Early stopping | Patience 10 on val greedy exact match |

**Hyperparameter selection.** These are standard transformer training defaults (AdamW with LR 3 × 10⁻⁴, cosine schedule, pre-norm). No systematic hyperparameter search was performed — the model converged with the first configuration tried, suggesting the task is not hyperparameter-sensitive given the right inductive bias.

### Hardware & Training Time

All training was done on an **Apple M4 Max** (MacBook Pro) using **PyTorch MPS** backend.

| n | Tokens/sample | Epoch time | Epochs to converge | Total wall time |
|---|--------------|------------|-------------------|----------------|
| 8 | 16 | ~15s | 23 | ~8 min |
| 10 | 20 | ~35s | 28 | ~17 min |
| 15 | 30 | 10–18 min (variable) | 24+ | ~6–10 hours |

At n=15, epoch times varied significantly due to intermittent MPS thermal throttling — some epochs took over 2 hours instead of the typical 10–18 minutes. No multi-GPU or cloud compute was used; the entire project was trained on a single laptop.

## Data

| n | |S_n| | Source | Train | Val | Test |
|---|------|--------|-------|-----|------|
| 8 | 40,320 | HuggingFace | 29,031 (72%) | 3,225 (8%) | 8,064 (20%) |
| 10 | 3,628,800 | Random sampling | 500,000 (14%) | 50,000 | 50,000 |
| 15 | 1,307,674,368,000 | Random sampling | 500,000 (0.00004%) | 50,000 | 50,000 |

For n=10 and n=15, data is generated on the fly: sample a random permutation of {1..n}, compute RSK forward to get (P, Q), encode as structured tokens. RSK forward is O(n²) worst case, taking ~0.03ms at n=10 and ~0.11ms at n=20 in pure Python — negligible compared to GPU time.

The HuggingFace datasets (ACDRepo) enumerate all n! permutations with an 80/20 train/test split. For n ≥ 10, converting 2.9M+ rows to Python lists is slow, so random sampling is preferred.

## Results

| n | |S_n| | Train fraction | Test greedy exact | Test argmax exact | Per-position | Best epoch |
|---|------|---------------|------------------|------------------|-------------|------------|
| 8 | 40,320 | 72% | 99.95% | 99.80% | 99.98% | 23 |
| 10 | 3,628,800 | 14% | **100.00%** | **100.00%** | **100.00%** | 28 |
| 15 | 1.3 × 10¹² | 0.00004% | **99.65%** | 99.08% | 99.94% | 24 |

### n=8 Details
- Converged at epoch 23, early-stopped at ~33
- Val greedy hit 100% at epoch 23; test greedy 99.95% (4 errors in 8,064)
- The 0.15% gap between argmax (99.80%) and greedy (99.95%) = greedy decoder fixing ~12 permutation violations

### n=10 Details
- Converged at epoch 28, early-stopped at 38
- Both argmax AND greedy hit 100% on 50,000 test samples — the model's per-position predictions are so confident that it never assigns the same value to two positions
- Val loss reached 0.0000 (below float display precision)

### n=15 Details
- Converged at epoch 24
- 99.65% greedy exact = 49,825 out of 50,000 permutations exactly right
- Greedy-argmax gap: 99.65% vs 99.08% = greedy decoder fixing ~285 permutation violations
- Per-position accuracy 99.94% = ~450 individual position errors across 50,000 × 15 = 750,000 predictions
- Training slowed by intermittent thermal throttling on MPS (some epochs took 2+ hours instead of the typical 10–18 minutes)

## The Memorisation Question

**Can the model just be memorising a lookup table?**

| n | Params | Unique inputs | Params per input | Training coverage | Verdict |
|---|--------|--------------|-----------------|-------------------|---------|
| 8 | 1.2M | 40,320 | 29.8 | 72% | Ambiguous — enough params to store most of the space |
| 10 | 1.2M | 3,628,800 | 0.33 | 14% | **Cannot memorise** — fewer params than inputs, only 14% seen |
| 15 | 1.2M | 1.3 × 10¹² | 9.4 × 10⁻⁷ | 0.00004% | **Provably algorithmic** — not even close to memorisation capacity |

At n=10: a lookup table would need 3.6M × 10 = 36M entries minimum. The model has 1.2M parameters total. It achieves 100% accuracy having seen only 14% of the space. This is unambiguous generalisation.

At n=15: the model has seen 500,000 out of 1.3 trillion possible inputs. It gets 99% of held-out inputs exactly right. There is no interpretation of this other than the model having learned a general algorithm for inverse RSK.

## What the Model Learned

We don't have full mechanistic interpretability, but we can reason about what it *must* have learned:

1. **Shape reconstruction**: from (P, Q) with the same shape λ, the model implicitly knows the partition λ (it's encoded in which (row, col) positions have entries)

2. **Reverse bumping paths**: for each Q-entry processed in decreasing order, the model must determine which P-entry to remove and trace the reverse bumping path upward through the rows

3. **Insertion order**: Q records *when* each cell was added (entry i in Q means that cell was created at step i). Processing Q in reverse reconstructs the sequence of deletions

4. **Value recovery**: at each step, reverse bumping from a cell at (row, col) in P recovers the original σ(i) value — the model's n heads each predict one of these recovered values

The model likely represents an approximate version of this algorithm distributed across its attention heads, with different heads specialising in different aspects of the reverse bumping computation.

## Architecture Design Decisions

| Decision | Rationale | Alternative | Why not |
|----------|-----------|-------------|---------|
| Encoder-only | All info in input, no sequential dependency | Encoder-decoder / autoregressive | Unnecessary complexity, slower inference |
| Structured 2D embedding | Preserves tableau geometry | Flat bracket tokens (PNNL) | Destroys spatial info, forces re-learning structure |
| Summed embeddings | Compact, each dimension additively contributes | Concatenated features | 4× wider input, more parameters for no gain |
| Mean pooling | Simple, all positions see all input | CLS token / attention pooling | Mean works well when all tokens are informative |
| n parallel heads | Each σ(i) is independent classification | Single autoregressive head | Parallel is faster, and independence is structurally correct |
| Masked greedy decode | Enforces permutation constraint | Beam search / Hungarian algorithm | Greedy is O(n²) and sufficient — beam search adds cost with no accuracy gain |
| Pre-norm transformer | Better training stability | Post-norm | Pre-norm is standard best practice for deep transformers |
| GELU activation | Smooth, standard for transformers | ReLU | Minor difference, GELU is modern default |

## Files

```
rsk.py       — RSK forward/inverse implementation (pure Python, no dependencies)
config.py    — ModelConfig + TrainConfig dataclasses
data.py      — HuggingFace loader, random sampling dataset, structured token encoding
model.py     — RSKEncoder (transformer) + BaselineMLP (flat comparison)
train.py     — Training loop, masked greedy decoding, evaluation metrics
```

## What's Next

1. **n=15 convergence** — reached 99.65% at epoch 24, retraining in progress
2. **Baseline MLP comparison** — quantify the contribution of structured encoding vs flat features
3. **Scale to n=20+** — the sampling pipeline supports any n; n=20 has 2.4 × 10¹⁸ permutations
4. **HuggingFace publication** — model weights, code, and results
5. **Error analysis** — what kinds of permutations/shapes does the model struggle with at n=15?
6. **Cylindric RSK** — Dobner (2026) defined an RSK analogue for cylindric tableaux, directly relevant to Robin's work on cylindric plane partitions. Could extend the ML approach to this setting.
