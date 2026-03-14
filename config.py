"""
Hyperparameter configuration for RSK neural network experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    n: int = 8  # permutation size (S_n)
    d_model: int = 128  # transformer embedding dimension
    nhead: int = 8  # number of attention heads
    num_layers: int = 6  # transformer encoder layers
    dim_feedforward: int = 512  # FFN hidden dimension (4× d_model)
    dropout: float = 0.1

    # Embedding table sizes (generous upper bounds for n ≤ 10)
    max_value: int = 10  # max entry value in tableaux (= n)
    max_rows: int = 10  # max number of rows in a tableau
    max_cols: int = 10  # max number of columns in a tableau

    # Baseline MLP config
    mlp_hidden: list[int] = field(default_factory=lambda: [256, 256, 128])

    def __post_init__(self):
        self.max_value = max(self.max_value, self.n)
        self.max_rows = max(self.max_rows, self.n)
        self.max_cols = max(self.max_cols, self.n)

    @property
    def num_tokens(self) -> int:
        """Number of input tokens: 2n (n entries from P + n entries from Q)."""
        return 2 * self.n

    @property
    def num_heads(self) -> int:
        """Number of classification heads (= n, one per position in σ)."""
        return self.n


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_fraction: float = 0.05  # fraction of total steps for warmup
    patience: int = 10  # early stopping patience (epochs)
    device: str = "mps"  # Apple Silicon GPU

    # Data
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    num_workers: int = 0  # DataLoader workers (0 = main process)
    seed: int = 42

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    log_every: int = 50  # log every N batches
