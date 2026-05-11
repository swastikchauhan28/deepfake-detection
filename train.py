"""
train.py — Training loop for DeepfakeDetector
Week 9 stage: Basic train/val loop with checkpointing and CSV logging.
No mixed precision or schedulers yet — kept intentionally simple.

Usage:
    python train.py --train_dir data/train --val_dir data/val \
                    --epochs 10 --batch_size 4 --lr 1e-4
"""

import os
import csv
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import build_dataloaders
from model   import DeepfakeDetector


# ──────────────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Deepfake Detector")
    p.add_argument("--train_dir",  type=str,   default="data/train")
    p.add_argument("--val_dir",    type=str,   default="data/val")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--n_frames",   type=int,   default=16)
    p.add_argument("--lstm_hidden",type=int,   default=256)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--save_dir",   type=str,   default="checkpoints")
    p.add_argument("--num_workers",type=int,   default=2)
    return p.parse_args()


# ──────────────────────────────────────────────
# One epoch of training
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss   = 0.0
    correct        = 0
    total          = 0

    for step, (frames, labels) in enumerate(loader):
        frames = frames.to(device)          # (B, T, 3, 224, 224)
        labels = labels.to(device)          # (B,)

        optimizer.zero_grad()
        logits, _ = model(frames)           # (B, 2)
        loss      = criterion(logits, labels)
        loss.backward()
        # Gradient clipping — prevents exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        if (step + 1) % 10 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} "
                  f"| Loss: {running_loss/(step+1):.4f} "
                  f"| Acc: {100*correct/total:.1f}%")

    avg_loss = running_loss / len(loader)
    avg_acc  = 100 * correct / total
    return avg_loss, avg_acc


# ──────────────────────────────────────────────
# Validation pass
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits, _ = model(frames)
        loss      = criterion(logits, labels)

        running_loss += loss.item()
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    avg_loss = running_loss / len(loader)
    avg_acc  = 100 * correct / total
    return avg_loss, avg_acc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Deepfake Detector — Training (Week 9 Stage)")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_dir   = args.train_dir,
        val_dir     = args.val_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers
    )

    # ── Model ─────────────────────────────────
    model = DeepfakeDetector(
        n_frames    = args.n_frames,
        lstm_hidden = args.lstm_hidden,
        dropout     = args.dropout,
        pretrained  = True
    ).to(device)

    # ── Loss, Optimiser, Scheduler ────────────
    # We use class weighting if the dataset is unbalanced.
    # For now keep equal weights — adjust if real:fake ratio skews.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Reduce LR by 0.5 every 4 epochs — mild warmup for a college project
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    # ── Checkpoint dir ────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)

    # ── CSV log ───────────────────────────────
    log_path = os.path.join(args.save_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc"])

    best_val_acc = 0.0

    # ── Training loop ─────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss,   val_acc   = validate(
            model, val_loader,   criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n[Epoch {epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.1f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.1f}%  "
              f"({elapsed:.0f}s)\n")

        # CSV logging
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                        f"{val_loss:.4f}",   f"{val_acc:.2f}"])

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:02d}.pth")
        torch.save({
            "epoch":      epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc":    val_acc,
            "val_loss":   val_loss,
        }, ckpt_path)

        # Save best model separately
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path    = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ✔ New best model saved (Val Acc: {val_acc:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Logs saved to: {log_path}")


if __name__ == "__main__":
    main()
