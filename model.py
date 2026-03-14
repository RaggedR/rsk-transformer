"""
Neural network architectures for learning the RSK correspondence.

Two models:
1. RSKEncoder — Encoder-only transformer with structured 2D token embeddings
   and n parallel classification heads. Our main model.
2. BaselineMLP — Flat MLP on flattened input for comparison with PNNL approach.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Structured embedding for tableau tokens.

    Each token represents one entry in a tableau. The embedding is:
        value_emb(entry) + row_emb(row) + col_emb(col) + tableau_emb(P_or_Q)

    This preserves the 2D geometric structure of the tableaux — the row/col
    embeddings let the model learn that "row 0, col 3" in P relates to
    "row 1, col 0" in Q through the RSK structure.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model

        # +1 because values/positions are 0-indexed but we include 0 as padding
        self.value_emb = nn.Embedding(config.max_value + 1, d)
        self.row_emb = nn.Embedding(config.max_rows, d)
        self.col_emb = nn.Embedding(config.max_cols, d)
        self.tableau_emb = nn.Embedding(2, d)  # P=0, Q=1

        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: (batch, 2n) — entry values (1..n)
            positions: (batch, 2n, 3) — [row, col, tableau_id] per token

        Returns:
            (batch, 2n, d_model) — token embeddings
        """
        rows = positions[:, :, 0]
        cols = positions[:, :, 1]
        tableau_ids = positions[:, :, 2]

        x = (
            self.value_emb(values)
            + self.row_emb(rows)
            + self.col_emb(cols)
            + self.tableau_emb(tableau_ids)
        )

        return self.dropout(self.layer_norm(x))


class RSKEncoder(nn.Module):
    """
    Encoder-only transformer for inverse RSK: (P, Q) → σ.

    Architecture:
    - TokenEmbedding: structured 2D embeddings for each tableau entry
    - TransformerEncoder: standard multi-head self-attention
    - n classification heads: each predicts σ(i) ∈ {1..n}

    The key insight: since (P, Q) fully determines σ, all information is present
    in the input — no autoregressive decoding needed. The encoder can attend to
    all 2n tokens simultaneously, and n parallel heads predict each position.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n = config.n

        self.embedding = TokenEmbedding(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # n classification heads, each predicting distribution over {0..n-1}
        # Each head sees the pooled encoder output
        self.heads = nn.ModuleList([
            nn.Linear(config.d_model, config.n)
            for _ in range(config.n)
        ])

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            values: (batch, 2n) — entry values
            positions: (batch, 2n, 3) — [row, col, tableau_id]

        Returns:
            logits: (batch, n, n) — logits[b, i, j] = log-prob that σ(i+1) = j+1
        """
        # Embed tokens: (batch, 2n, d_model)
        x = self.embedding(values, positions)

        # Encode: (batch, 2n, d_model)
        x = self.encoder(x)

        # Pool over all tokens: mean pooling → (batch, d_model)
        x = x.mean(dim=1)

        # n classification heads → (batch, n, n)
        logits = torch.stack([head(x) for head in self.heads], dim=1)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineMLP(nn.Module):
    """
    Flat MLP baseline for comparison with PNNL approach.

    Flattens both tableaux into a single vector and predicts each σ(i)
    independently via shared hidden layers + n heads.

    This destroys the 2D structure — comparing its performance against
    RSKEncoder isolates the contribution of our structured encoding.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n = config.n

        # Flatten: 2n tokens × 4 features (value, row, col, tableau_id)
        input_dim = 2 * config.n * 4

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.dropout),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # n classification heads
        self.heads = nn.ModuleList([
            nn.Linear(prev_dim, config.n)
            for _ in range(config.n)
        ])

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Same interface as RSKEncoder for drop-in comparison."""
        batch = values.shape[0]

        # Flatten: concatenate values and positions into a single vector
        # values: (batch, 2n) → (batch, 2n, 1)
        # positions: (batch, 2n, 3)
        # combined: (batch, 2n, 4) → (batch, 2n * 4)
        v = values.unsqueeze(-1).float()
        p = positions.float()
        x = torch.cat([v, p], dim=-1)  # (batch, 2n, 4)
        x = x.view(batch, -1)  # (batch, 2n * 4)

        x = self.shared(x)  # (batch, hidden)

        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = ModelConfig(n=8)

    # Test RSKEncoder
    model = RSKEncoder(config)
    print(f"RSKEncoder: {model.count_parameters():,} parameters")

    batch = 4
    values = torch.randint(1, 9, (batch, 16))
    positions = torch.zeros(batch, 16, 3, dtype=torch.long)
    positions[:, :, 2] = torch.cat([torch.zeros(8), torch.ones(8)]).long()

    logits = model(values, positions)
    print(f"Input: values {values.shape}, positions {positions.shape}")
    print(f"Output: logits {logits.shape}")  # (4, 8, 8)

    # Test BaselineMLP
    mlp = BaselineMLP(config)
    print(f"\nBaselineMLP: {mlp.count_parameters():,} parameters")
    logits_mlp = mlp(values, positions)
    print(f"Output: logits {logits_mlp.shape}")
