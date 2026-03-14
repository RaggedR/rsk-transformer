# Learning the RSK Correspondence with Transformers

A neural network that learns the **inverse Robinson-Schensted-Knuth correspondence**: given a pair of standard Young tableaux (P, Q), predict the permutation σ that produced them.

Achieves **100% exact-match accuracy** on held-out test data for n=10 and **99.99%** for n=15 (1.3 trillion permutations), significantly improving on the [PNNL ML4AlgComb benchmark](https://github.com/pnnl/ML4AlgComb/tree/master/rsk) which only achieved weak baselines on this task.

For a detailed writeup of the architecture, training, and results, see [SUMMARY.md](SUMMARY.md).

**Links**: [Trained models (HuggingFace)](https://huggingface.co/RobBobin/rsk-transformer) | HuggingFace datasets: [n=8](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_8), [n=9](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_9), [n=10](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_10)

## Results

| n | \|S_n\| | Training data | Source | Model params | Test exact match | Best epoch |
|---|---------|--------------|--------|-------------|-----------------|------------|
| 8 | 40,320 | 29,031 (72% of S_n) | [HuggingFace](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_8) | 1,202,368 | 99.95% | 23 |
| 10 | 3,628,800 | 500,000 (14% of S_n) | sampling | 1,207,012 | **100.00%** | 28 |
| 15 | 1.3 × 10¹² | 500,000 (0.00004%) | sampling | 1,225,057 | **99.99%** | 52 |

The n=10 result rules out memorisation: a 1.2M-parameter model trained on 14% of the permutation space achieves perfect accuracy on 50,000 unseen test permutations. At n=15 (1.3 trillion permutations), the same architecture trained on 0.00004% of the space gets 49,995 out of 50,000 held-out permutations exactly right — unambiguous algorithmic generalisation.

## Key Idea: Structured 2D Token Embeddings

Previous work (PNNL) encoded tableaux as flat bracket strings like `[[1,3,5],[2,4]]`, destroying the 2D geometric structure that makes RSK work. We instead encode each tableau entry as a token with four learned embeddings:

```
embedding(entry) = value_emb(v) + row_emb(r) + col_emb(c) + tableau_emb(P or Q)
```

This preserves the spatial relationships that govern Schensted insertion — the model can learn that bumping paths move down rows, that column-strictness constrains which values can appear where, and that P and Q share the same shape.

## Architecture

```
Input: (P, Q) as 2n structured tokens
  → TokenEmbedding (value + row + col + tableau_id)
  → 6-layer TransformerEncoder (d=128, 8 heads, pre-norm, GELU)
  → Mean pool over all 2n tokens
  → n parallel classification heads → logits (batch, n, n)
  → Masked greedy decode → predicted σ
```

**Encoder-only**: since (P, Q) fully determines σ, all information is in the input. No autoregressive decoding needed — all 2n tokens attend to each other simultaneously.

**Masked greedy decoding**: at inference, we enforce the permutation constraint by iteratively picking the highest-confidence (position, value) pair globally, locking it in, and masking that value from all other heads.

## Quick Start

```bash
pip install torch datasets

# Train on n=10 with random sampling
python train.py --model encoder --n 10 --source sample --train-size 500000 --batch-size 512

# Train on n=8 with HuggingFace data
python train.py --model encoder --n 8 --source hf

# Train on any n (sampling scales to arbitrary size)
python train.py --model encoder --n 20 --source sample --train-size 1000000

# Verify RSK implementation (round-trip bijection for S_1 through S_6)
python rsk.py
```

### Data Sources

| Source | Flag | Description |
|--------|------|-------------|
| HuggingFace | `--source hf` | [ACDRepo](https://huggingface.co/ACDRepo) datasets for [n=8](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_8), [n=9](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_9), [n=10](https://huggingface.co/datasets/ACDRepo/robinson_schensted_knuth_correspondence_10). Full enumeration with 80/20 split. |
| Sampling | `--source sample` | Random permutations with on-the-fly RSK computation. Works for **any n**. Use `--train-size` to control dataset size. |
| Generate | `--source generate` | Enumerates all n! permutations locally. Only feasible for n ≤ ~8. |

### CLI Options

```
--model {encoder,mlp}     Model architecture (default: encoder)
--n N                     Permutation size (default: 8)
--source {hf,generate,sample}  Data source (default: hf)
--train-size N            Training samples for --source sample (default: 500000)
--val-size N              Validation samples (default: 50000)
--test-size N             Test samples (default: 50000)
--epochs N                Max epochs (default: 100)
--batch-size N            Batch size (default: 256)
--lr FLOAT                Learning rate (default: 3e-4)
--device DEVICE           Device (default: mps)
--d-model N               Transformer dimension (default: 128)
--num-layers N            Transformer layers (default: 6)
--nhead N                 Attention heads (default: 8)
--resume                  Resume training from existing checkpoint
```

## Project Structure

```
rsk.py      RSK forward/inverse (Schensted insertion + reverse bumping)
data.py     Data pipeline: HuggingFace loading, sampling, structured encoding
model.py    RSKEncoder (transformer) and BaselineMLP (flat comparison)
config.py   ModelConfig and TrainConfig dataclasses
train.py    Training loop, masked greedy decoding, evaluation
```

## Background

The [RSK correspondence](https://en.wikipedia.org/wiki/Robinson%E2%80%93Schensted%E2%80%93Knuth_correspondence) is a bijection between permutations σ ∈ S_n and pairs (P, Q) of standard Young tableaux of the same shape λ ⊢ n. It is central to algebraic combinatorics, connecting representation theory, symmetric functions, and enumerative combinatorics.

The **forward** direction (σ → P, Q) uses Schensted row insertion (the bumping algorithm). The **inverse** (P, Q → σ) uses reverse bumping, processing entries in decreasing order. For a detailed treatment, see [Langer (2013), Chapter 2, §2.1–2.2, pp. 25–43](https://arxiv.org/abs/2110.12629). This project trains a neural network to learn the inverse direction.

## Why Not Memorisation?

At n=10, there are 3,628,800 permutations. The model has 1.2M parameters and was trained on 500,000 samples (14% of the space). A lookup table would need at least 3.6M × 10 = 36M values to store the full mapping — 30× more than the model's parameter count. Yet the model achieves 100% accuracy on held-out samples.

At n=15, there are 1.3 trillion permutations. The model sees 0.00004% of the space during training and gets 99.99% exact match — unambiguously algorithmic generalisation. A baseline MLP on the same data achieves only 3.07%, showing the transformer's structured 2D embedding and attention are essential.

## Citation

If you use this work, please cite:

```
@software{rsk_transformer,
  author={Langer, Robin},
  title={Learning the RSK Correspondence with Transformers},
  year={2026},
  url={https://github.com/RaggedR/rsk-transformer}
}
```

## Acknowledgements

- [PNNL ML4AlgComb](https://github.com/pnnl/ML4AlgComb/tree/master/rsk) for the original benchmark and HuggingFace datasets
- [ACDRepo](https://huggingface.co/ACDRepo) for pre-computed RSK datasets
