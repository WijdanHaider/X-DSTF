"""
train.py — Training script for XDual-Stream Transformer Fusion (XDSTF) model.

Usage example:
    python train.py --data_dir data/ --batch_size 16 --lr 1e-4 --epochs 25

Hardware used in paper experiments:
    Training  : 2× NVIDIA T4 (15 GB VRAM each), Intel Xeon (2-core, 2.2 GHz), 32 GB RAM
    Inference : 1× NVIDIA L4 (22.16 GB VRAM), Intel Xeon (6-core, 2.2 GHz), 53 GB RAM
    Framework : PyTorch 2.8.0, CUDA 12.6
"""

import os
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.xdstf import DualStreamModel
from datasets.forgery_dataset import ForgeryDataset
from transforms.dft import get_transforms          # expected: get_transforms(split) -> (spatial_tf, freq_tf)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the XDSTF dual-stream deepfake detection model."
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",    type=str, default="data/",
                        help="Root directory containing train/ and val/ splits.")
    parser.add_argument("--output_dir",  type=str, default="checkpoints/",
                        help="Directory where model checkpoints are saved.")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to a .pth.tar checkpoint to resume from.")

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--d_model",     type=int, default=512,
                        help="Transformer hidden dimension.")
    parser.add_argument("--nhead",       type=int, default=8,
                        help="Number of attention heads.")
    parser.add_argument("--num_layers",  type=int, default=2,
                        help="Number of Transformer encoder layers.")

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=int,   default=25,
                        help="Maximum number of training epochs.")
    parser.add_argument("--batch_size",  type=int,   default=16,
                        help="Batch size per iteration.")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Initial learning rate for AdamW.")
    parser.add_argument("--weight_decay",type=float, default=1e-4,
                        help="AdamW weight decay.")
    parser.add_argument("--num_workers", type=int,   default=4,
                        help="DataLoader worker processes.")

    # ── Scheduler / Early Stopping ─────────────────────────────────────────
    parser.add_argument("--lr_patience", type=int,   default=3,
                        help="ReduceLROnPlateau patience (epochs).")
    parser.add_argument("--lr_factor",   type=float, default=0.2,
                        help="ReduceLROnPlateau reduction factor.")
    parser.add_argument("--es_patience", type=int,   default=5,
                        help="Early stopping patience (epochs).")

    # ── Augmentation ───────────────────────────────────────────────────────
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["none", "light", "mild", "extensive"],
        help=(
            "Augmentation strategy applied to raw images before DFT conversion. "
            "See Table 15 in the paper for per-strategy results. "
            "Required — choose from: none | light | mild | extensive."
        ),
        required=True,
    )

    # ── Misc ───────────────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--input_size",  type=int,   default=224,
                        help="Spatial input resolution (square).")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  Early Stopping Helper
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def step(self, val_loss: float) -> bool:
        score = -val_loss                              # lower loss = better score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter    = 0
        return self.stop


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == labels.long()).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
#  One Epoch: Train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch_idx, batch in enumerate(loader):
        # ForgeryDataset is expected to return (spatial_img, freq_img, label)
        spatial_imgs, freq_imgs, labels = batch
        spatial_imgs = spatial_imgs.to(device, non_blocking=True)
        freq_imgs    = freq_imgs.to(device, non_blocking=True)
        labels       = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(spatial_imgs, freq_imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = binary_accuracy(logits.detach(), labels.detach())
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch [{epoch}] Step [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}  Acc: {acc:.4f}")

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────────────────────
#  One Epoch: Validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch in loader:
        spatial_imgs, freq_imgs, labels = batch
        spatial_imgs = spatial_imgs.to(device, non_blocking=True)
        freq_imgs    = freq_imgs.to(device, non_blocking=True)
        labels       = labels.float().unsqueeze(1).to(device, non_blocking=True)

        logits = model(spatial_imgs, freq_imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        total_acc  += binary_accuracy(logits, labels)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler):
    print(f"[Resume] Loading checkpoint from: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed from epoch {ckpt['epoch']}  |  best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus    = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  XDSTF Training Script")
    print(f"{'='*60}")
    print(f"  Device      : {device}  ({n_gpus} GPU(s) detected)")
    print(f"  Seed        : {args.seed}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR          : {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Transforms ──────────────────────────────────────────────────────────
    # get_transforms(augmentation) returns (spatial_transform, freq_transform).
    # Augmentation is applied only during training; validation always uses clean images.
    train_spatial_tf, train_freq_tf = get_transforms(augmentation=args.augmentation)
    val_spatial_tf,   val_freq_tf   = get_transforms(augmentation="none")

    # ── Datasets & Loaders ──────────────────────────────────────────────────
    train_dataset = ForgeryDataset(
        root=os.path.join(args.data_dir, "train"),
        spatial_transform=train_spatial_tf,
        freq_transform=train_freq_tf,
    )
    val_dataset = ForgeryDataset(
        root=os.path.join(args.data_dir, "val"),
        spatial_transform=val_spatial_tf,
        freq_transform=val_freq_tf,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"  Train samples : {len(train_dataset):,}")
    print(f"  Val samples   : {len(val_dataset):,}\n")

    # ── Model ───────────────────────────────────────────────────────────────
    model = DualStreamModel(
        num_classes=1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )

    # Multi-GPU support (paper used 2× T4)
    if n_gpus > 1:
        print(f"  Using DataParallel across {n_gpus} GPUs.\n")
        model = nn.DataParallel(model)

    model = model.to(device)

    # ── Loss / Optimizer / Scheduler ────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
    )

    early_stopping = EarlyStopping(patience=args.es_patience)

    # ── Optional Resume ─────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # ── Training Loop ────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print("  Starting Training")
    print(f"{'─'*60}\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch [{epoch}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  ── Epoch {epoch} Summary ─────────────────────────")
        print(f"  Train  Loss: {train_loss:.4f}  |  Acc: {train_acc:.4f}")
        print(f"  Val    Loss: {val_loss:.4f}  |  Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.2e}  |  Time: {elapsed:.1f}s")
        print(f"  {'─'*45}\n")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Scheduler step (monitors val loss)
        scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch":         epoch,
                    "state_dict":    model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "scheduler":     scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args":          vars(args),
                },
                path=os.path.join(args.output_dir, "best_model.pth.tar"),
            )

        # Save latest checkpoint (for resuming)
        save_checkpoint(
            {
                "epoch":         epoch,
                "state_dict":    model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "args":          vars(args),
            },
            path=os.path.join(args.output_dir, "latest_checkpoint.pth.tar"),
        )

        # Early stopping check
        if early_stopping.step(val_loss):
            print(f"\n  [EarlyStopping] Triggered at epoch {epoch}. Training stopped.")
            break

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training Complete")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Best model    : {os.path.join(args.output_dir, 'best_model.pth.tar')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
