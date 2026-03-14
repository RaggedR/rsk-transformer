"""
Data pipeline for RSK neural network experiments.

Loads (P, Q, σ) data from HuggingFace or generates it with our RSK implementation.
Encodes tableaux into structured tokens: each entry becomes a token with
(value, row, col, tableau_id) — preserving the 2D geometric structure that
PNNL's bracket-token encoding destroys.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from rsk import rsk_forward, tableau_positions, Tableau
from config import ModelConfig, TrainConfig


# HuggingFace dataset names
HF_DATASET_TEMPLATE = "ACDRepo/robinson_schensted_knuth_correspondence_{n}"


def load_hf_dataset(n: int) -> tuple[list, list]:
    """
    Load RSK dataset from HuggingFace.

    Returns (train_data, test_data) where each is a list of dicts with keys:
        'Permutation', 'Standard Young tableau 1', 'Standard Young tableau 2'
    """
    from datasets import load_dataset

    name = HF_DATASET_TEMPLATE.format(n=n)
    ds = load_dataset(name)
    return list(ds["train"]), list(ds["test"])


def verify_hf_against_rsk(data: list[dict], n: int, max_check: int = 100) -> tuple[int, int]:
    """
    Cross-check HuggingFace data against our RSK implementation.

    Returns (num_checked, num_matched).
    """
    checked = 0
    matched = 0

    for item in data[:max_check]:
        sigma = item["Permutation"]
        P_hf = item["Standard Young tableau 1"]
        Q_hf = item["Standard Young tableau 2"]

        P_ours, Q_ours = rsk_forward(sigma)

        if P_ours == P_hf and Q_ours == Q_hf:
            matched += 1
        checked += 1

    return checked, matched


def encode_tableaux(P: Tableau, Q: Tableau) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a pair of SYTs into token representations.

    Each tableau entry becomes a token. For n entries in each tableau, we get 2n tokens.

    Returns:
        values: LongTensor of shape (2n,) — entry values (1-indexed)
        positions: LongTensor of shape (2n, 3) — [row, col, tableau_id] per token
            tableau_id: 0 for P, 1 for Q
    """
    P_positions = tableau_positions(P)  # list of (value, row, col)
    Q_positions = tableau_positions(Q)

    values = []
    positions = []

    # P tokens (tableau_id = 0)
    for val, row, col in P_positions:
        values.append(val)
        positions.append([row, col, 0])

    # Q tokens (tableau_id = 1)
    for val, row, col in Q_positions:
        values.append(val)
        positions.append([row, col, 1])

    return torch.tensor(values, dtype=torch.long), torch.tensor(positions, dtype=torch.long)


class RSKDataset(Dataset):
    """
    PyTorch Dataset for RSK data.

    Each item returns:
        values: (2n,) LongTensor — token values
        positions: (2n, 3) LongTensor — [row, col, tableau_id] per token
        target: (n,) LongTensor — permutation σ (0-indexed for cross-entropy)
    """

    def __init__(self, data: list[dict]):
        """
        Args:
            data: list of dicts with keys 'Permutation',
                  'Standard Young tableau 1', 'Standard Young tableau 2'
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        P = item["Standard Young tableau 1"]
        Q = item["Standard Young tableau 2"]
        sigma = item["Permutation"]

        values, positions = encode_tableaux(P, Q)

        # Convert σ to 0-indexed for cross-entropy loss
        target = torch.tensor([v - 1 for v in sigma], dtype=torch.long)

        return values, positions, target


class RSKSamplingDataset(Dataset):
    """
    On-the-fly RSK dataset for arbitrary n.

    Samples random permutations and computes RSK at access time.
    No enumeration needed — works for any n where RSK is computable.
    """

    def __init__(self, n: int, size: int, seed: int | None = None):
        self.n = n
        self.size = size
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Random permutation of {1..n}
        sigma = (torch.randperm(self.n, generator=self.rng) + 1).tolist()
        P, Q = rsk_forward(sigma)
        values, positions = encode_tableaux(P, Q)
        target = torch.tensor([v - 1 for v in sigma], dtype=torch.long)
        return values, positions, target


def generate_our_dataset(n: int) -> list[dict]:
    """
    Generate all (P, Q, σ) triples for S_n using our RSK implementation.

    Returns data in the same format as HuggingFace for compatibility.
    """
    from itertools import permutations

    data = []
    for perm in permutations(range(1, n + 1)):
        sigma = list(perm)
        P, Q = rsk_forward(sigma)
        data.append({
            "Permutation": sigma,
            "Standard Young tableau 1": P,
            "Standard Young tableau 2": Q,
        })
    return data


def make_dataloaders(
    n: int,
    train_config: TrainConfig,
    source: str = "hf",
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        n: permutation size
        train_config: training configuration
        source: "hf" for HuggingFace, "generate" for enumeration, "sample" for random sampling
        train_size: for source="sample", number of training examples
        val_size: for source="sample", number of validation examples
        test_size: for source="sample", number of test examples

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if source == "sample":
        # On-the-fly sampling — works for any n
        if train_size is None:
            train_size = 500_000
        if val_size is None:
            val_size = 50_000
        if test_size is None:
            test_size = 50_000

        train_ds = RSKSamplingDataset(n, train_size, seed=train_config.seed)
        val_ds = RSKSamplingDataset(n, val_size, seed=train_config.seed + 1)
        test_ds = RSKSamplingDataset(n, test_size, seed=train_config.seed + 2)
    else:
        if source == "hf":
            train_data, test_data = load_hf_dataset(n)
        else:
            all_data = generate_our_dataset(n)
            split = int(0.8 * len(all_data))
            import random
            random.seed(train_config.seed)
            random.shuffle(all_data)
            train_data = all_data[:split]
            test_data = all_data[split:]

        # Split train into train + val
        full_train = RSKDataset(train_data)
        v_size = int(train_config.val_fraction * len(full_train))
        t_size = len(full_train) - v_size

        generator = torch.Generator().manual_seed(train_config.seed)
        train_ds, val_ds = random_split(full_train, [t_size, v_size], generator=generator)
        test_ds = RSKDataset(test_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test encoding
    P = [[1, 3, 5, 8], [2, 6], [4, 7]]
    Q = [[1, 3, 5, 8], [2, 4], [6, 7]]
    values, positions = encode_tableaux(P, Q)
    print(f"Values shape: {values.shape}, Positions shape: {positions.shape}")
    print(f"Values: {values}")
    print(f"Positions:\n{positions}")

    # Test dataset generation for small n
    data = generate_our_dataset(4)
    ds = RSKDataset(data)
    vals, pos, target = ds[0]
    print(f"\nn=4, first item:")
    print(f"  values={vals}, positions={pos}, target={target}")
    print(f"  sigma (1-indexed) = {[t.item() + 1 for t in target]}")
