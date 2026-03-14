"""
Training loop and evaluation for RSK neural network experiments.

Features:
- AdamW optimizer with cosine decay and linear warmup
- Masked greedy decoding for valid permutation inference
- Exact-match and per-position accuracy metrics
- Early stopping on validation exact-match accuracy
- Checkpoint saving
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import ModelConfig, TrainConfig
from model import RSKEncoder, BaselineMLP
from data import make_dataloaders, verify_hf_against_rsk, load_hf_dataset


def masked_greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Masked greedy decoding to produce valid permutations.

    Process: pick the highest-confidence (head, value) pair across all heads,
    assign it, mask that value from all other heads, repeat.

    Args:
        logits: (batch, n, n) — raw logits from model

    Returns:
        preds: (batch, n) — predicted permutation (0-indexed)
    """
    batch, n, _ = logits.shape
    preds = torch.full((batch, n), -1, dtype=torch.long, device=logits.device)

    # Work with a copy we can mask
    scores = logits.clone()

    for _ in range(n):
        # Find the global argmax across all (head, value) pairs per batch
        # Reshape to (batch, n*n) to find global max
        flat = scores.view(batch, -1)
        max_idx = flat.argmax(dim=1)  # (batch,)

        head_idx = max_idx // n  # which position
        val_idx = max_idx % n  # which value

        # Assign
        preds[torch.arange(batch, device=logits.device), head_idx] = val_idx

        # Mask: set this head and this value to -inf everywhere
        for b in range(batch):
            h, v = head_idx[b].item(), val_idx[b].item()
            scores[b, h, :] = float("-inf")  # this position is decided
            scores[b, :, v] = float("-inf")  # this value is used

    return preds


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        logits: (batch, n, n) — model output
        targets: (batch, n) — ground truth (0-indexed)

    Returns:
        dict with 'exact_match', 'per_position', 'greedy_exact_match'
    """
    batch, n, _ = logits.shape

    # Per-position accuracy (argmax, ignoring permutation constraint)
    argmax_preds = logits.argmax(dim=-1)  # (batch, n)
    per_pos = (argmax_preds == targets).float().mean().item()

    # Exact match with unconstrained argmax
    exact = (argmax_preds == targets).all(dim=1).float().mean().item()

    # Exact match with masked greedy decoding
    greedy_preds = masked_greedy_decode(logits)
    greedy_exact = (greedy_preds == targets).all(dim=1).float().mean().item()

    return {
        "exact_match": exact,
        "per_position": per_pos,
        "greedy_exact_match": greedy_exact,
    }


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    log_every: int = 50,
) -> dict[str, float]:
    """Train for one epoch. Returns average metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_positions = 0
    num_batches = 0

    for batch_idx, (values, positions, targets) in enumerate(loader):
        values = values.to(device)
        positions = positions.to(device)
        targets = targets.to(device)

        logits = model(values, positions)  # (batch, n, n)
        batch_size, n, _ = logits.shape

        # Cross-entropy across all n heads
        # Reshape: (batch * n, n) vs (batch * n,)
        loss = F.cross_entropy(
            logits.view(-1, n), targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_positions += batch_size * n
        num_batches += 1

        if log_every and (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_correct / total_positions
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  batch {batch_idx + 1:4d}/{len(loader)}: "
                f"loss={avg_loss:.4f} pos_acc={avg_acc:.4f} lr={lr:.2e}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "per_position": total_correct / max(total_positions, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate on a dataset. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    num_batches = 0

    for values, positions, targets in loader:
        values = values.to(device)
        positions = positions.to(device)
        targets = targets.to(device)

        logits = model(values, positions)
        batch_size, n, _ = logits.shape

        loss = F.cross_entropy(logits.view(-1, n), targets.view(-1))
        total_loss += loss.item()
        num_batches += 1

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def train(
    model_name: str = "encoder",
    n: int = 8,
    source: str = "hf",
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    resume: bool = False,
):
    """
    Full training pipeline.

    Args:
        model_name: "encoder" for RSKEncoder, "mlp" for BaselineMLP
        n: permutation size
        source: "hf" for HuggingFace, "generate" for our own data
        model_config: override model config
        train_config: override train config
        resume: if True, load existing checkpoint and continue training
    """
    if model_config is None:
        model_config = ModelConfig(n=n)
    if train_config is None:
        train_config = TrainConfig()

    device = torch.device(train_config.device)
    print(f"Device: {device}")

    # Data
    print(f"\nLoading data for n={n} (source={source})...")
    train_loader, val_loader, test_loader = make_dataloaders(
        n, train_config, source,
        train_size=train_size, val_size=val_size, test_size=test_size,
    )
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")

    # Verify HuggingFace data if using it
    if source == "hf":
        print("\nCross-checking HuggingFace data against our RSK...")
        train_data, _ = load_hf_dataset(n)
        checked, matched = verify_hf_against_rsk(train_data, n, max_check=200)
        print(f"  {matched}/{checked} match our RSK implementation")
        if matched < checked:
            print("  WARNING: Some mismatches detected!")
        del train_data

    # Model
    if model_name == "encoder":
        model = RSKEncoder(model_config)
    elif model_name == "mlp":
        model = BaselineMLP(model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    print(f"\n{model_name}: {model.count_parameters():,} parameters")

    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    total_steps = len(train_loader) * train_config.epochs
    warmup_steps = int(train_config.warmup_fraction * total_steps)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    # Checkpoint directory
    ckpt_dir = Path(train_config.checkpoint_dir) / f"{model_name}_n{n}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_exact = 0.0
    patience_counter = 0

    if resume:
        ckpt_path = ckpt_dir / "best.pt"
        if ckpt_path.exists():
            print(f"\nResuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_exact = ckpt["val_metrics"].get("greedy_exact_match", 0.0)
            # Advance scheduler to the right step
            steps_done = len(train_loader) * ckpt["epoch"]
            for _ in range(steps_done):
                scheduler.step()
            print(f"  Resumed at epoch {start_epoch}, best greedy={best_val_exact:.4f}")
        else:
            print(f"\nNo checkpoint found at {ckpt_path}, starting fresh.")

    print(f"\nTraining for up to {train_config.epochs} epochs...")
    print(f"{'='*80}")

    for epoch in range(start_epoch, train_config.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            log_every=train_config.log_every,
        )

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{train_config.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} train_pos={train_metrics['per_position']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_pos={val_metrics['per_position']:.4f} "
            f"val_exact={val_metrics['exact_match']:.4f} "
            f"val_greedy={val_metrics['greedy_exact_match']:.4f}"
        )

        # Save checkpoint every epoch (always have something to evaluate)
        ckpt_path = ckpt_dir / "best.pt"

        # Early stopping on greedy exact match
        val_exact = val_metrics["greedy_exact_match"]
        if val_exact > best_val_exact or epoch == 1:
            best_val_exact = val_exact
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_config": model_config,
            }, ckpt_path)
            if val_exact > 0:
                print(f"  -> New best! Saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= train_config.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={train_config.patience})")
                break

    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Loading best model for test evaluation...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest Results (n={n}, {model_name}):")
    print(f"  Loss:              {test_metrics['loss']:.4f}")
    print(f"  Per-position acc:  {test_metrics['per_position']:.4f}")
    print(f"  Exact match:       {test_metrics['exact_match']:.4f}")
    print(f"  Greedy exact match:{test_metrics['greedy_exact_match']:.4f}")
    print(f"  Best val greedy:   {best_val_exact:.4f} (epoch {ckpt['epoch']})")

    return test_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RSK neural network")
    parser.add_argument("--model", choices=["encoder", "mlp"], default="encoder")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--source", choices=["hf", "generate", "sample"], default="hf")
    parser.add_argument("--train-size", type=int, default=None, help="Training set size (for --source sample)")
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    model_config = ModelConfig(
        n=args.n,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    train(
        model_name=args.model,
        n=args.n,
        source=args.source,
        model_config=model_config,
        train_config=train_config,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        resume=args.resume,
    )
