---
language: en
license: mit
tags:
  - algebraic-combinatorics
  - rsk-correspondence
  - young-tableaux
  - permutations
  - transformer
  - pytorch
datasets:
  - ACDRepo/robinson_schensted_knuth_correspondence_8
  - ACDRepo/robinson_schensted_knuth_correspondence_10
pipeline_tag: other
---

# RSK Transformer

A transformer that learns the **inverse Robinson-Schensted-Knuth correspondence**: given a pair of standard Young tableaux (P, Q), predict the permutation σ that produced them.

Achieves **100% exact-match accuracy** on held-out test data for n=10 and **99.99%** for n=15 (1.3 trillion permutations), significantly improving on the [PNNL ML4AlgComb benchmark](https://github.com/pnnl/ML4AlgComb/tree/master/rsk).

📄 **Paper**: [paper.pdf](paper.pdf)
💻 **Code**: [github.com/RaggedR/rsk-transformer](https://github.com/RaggedR/rsk-transformer)

## Results

| n | \|S_n\| | Training data | Test exact match | Best epoch |
|---|---------|--------------|-----------------|------------|
| 8 | 40,320 | 29,031 (72% of S_n) | 99.95% | 23 |
| 10 | 3,628,800 | 500,000 (14% of S_n) | **100.00%** | 28 |
| 15 | 1.3 × 10¹² | 500,000 (0.00004%) | **99.99%** | 52 |

The n=10 result rules out memorisation: a 1.2M-parameter model trained on 14% of the permutation space achieves perfect accuracy on 50,000 unseen test permutations. At n=15 (1.3 trillion permutations), training on 0.00004% of the space yields 99.99% — unambiguous algorithmic generalisation.

### Ablation: Transformer vs MLP

| n | Model | Parameters | Greedy exact | Argmax exact | Per-position |
|---|-------|-----------|-------------|-------------|-------------|
| 10 | **RSKEncoder** (transformer) | 1,207,012 | **100.00%** | **100.00%** | **100.00%** |
| 10 | BaselineMLP (flat) | 133,604 | 95.67% | 90.31% | 98.85% |
| 15 | **RSKEncoder** (transformer) | 1,225,057 | **99.99%** | **99.98%** | **100.00%** |
| 15 | BaselineMLP (flat) | 133,604 | 3.07% | 0.04% | 62.02% |

The MLP collapses from 95.67% to 3.07% as n increases from 10 to 15, while the transformer barely notices (100% → 99.99%). Without spatial structure, the MLP cannot coordinate predictions across positions.

## Key Idea: Structured 2D Token Embeddings

Previous work encoded tableaux as flat bracket strings, destroying 2D geometry. We encode each tableau entry as a token with four learned embeddings:

```
embedding(entry) = value_emb(v) + row_emb(r) + col_emb(c) + tableau_emb(P or Q)
```

## Architecture

```
Input: (P, Q) as 2n structured tokens
  → TokenEmbedding (value + row + col + tableau_id)
  → 6-layer TransformerEncoder (d=128, 8 heads, pre-norm, GELU)
  → Mean pool over all 2n tokens
  → n parallel classification heads → logits (batch, n, n)
  → Masked greedy decode → predicted σ
```

~1.2M parameters. Encoder-only (no autoregressive decoding needed).

## Checkpoints

| File | Description | Parameters |
|------|-------------|-----------|
| `checkpoints/encoder_n8/best.pt` | RSKEncoder trained on S₈ (HuggingFace data) | 1,202,368 |
| `checkpoints/encoder_n10/best.pt` | RSKEncoder trained on S₁₀ (sampled) | 1,207,012 |
| `checkpoints/encoder_n15/best.pt` | RSKEncoder trained on S₁₅ (sampled) | 1,225,057 |
| `checkpoints/mlp_n10/best.pt` | Baseline MLP on S₁₀ (for ablation) | 133,604 |
| `checkpoints/mlp_n15/best.pt` | Baseline MLP on S₁₅ (for ablation) | 133,604 |

### Loading a checkpoint

```python
import torch
from model import RSKEncoder
from config import ModelConfig

# Load n=10 model
ckpt = torch.load("checkpoints/encoder_n10/best.pt", map_location="cpu", weights_only=False)
config = ckpt["model_config"]
model = RSKEncoder(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Training

```bash
pip install torch datasets

# Reproduce n=10 result
python train.py --model encoder --n 10 --source sample --train-size 500000 --batch-size 512

# Reproduce n=8 result with HuggingFace data
python train.py --model encoder --n 8 --source hf
```

## Citation

```bibtex
@software{rsk_transformer,
  author={Langer, Robin},
  title={Learning the RSK Correspondence with Transformers},
  year={2026},
  url={https://github.com/RaggedR/rsk-transformer}
}
```

## Acknowledgements

- [PNNL ML4AlgComb](https://github.com/pnnl/ML4AlgComb/tree/master/rsk) for the original benchmark
- [ACDRepo](https://huggingface.co/ACDRepo) for pre-computed RSK datasets
