"""
evaluate.py — Compute test-set metrics: accuracy, precision, recall, F1, confusion matrix
Week 9 stage: Basic sklearn metrics, no ROC curve yet.

Usage:
    python evaluate.py --test_dir data/test \
                       --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

from dataset import DeepfakeVideoDataset
from model   import DeepfakeDetector
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir",   type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_frames",   type=int, default=16)
    p.add_argument("--lstm_hidden",type=int, default=256)
    p.add_argument("--num_workers",type=int, default=2)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────
    test_ds = DeepfakeVideoDataset(args.test_dir, split="val",
                                   n_frames=args.n_frames)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────
    model = DeepfakeDetector(
        n_frames=args.n_frames, lstm_hidden=args.lstm_hidden, pretrained=False
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # ── Collect predictions ───────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            logits, _ = model(frames)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────
    acc  = accuracy_score(all_labels,  all_preds) * 100
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    rec  = recall_score(all_labels,    all_preds, zero_division=0) * 100
    f1   = f1_score(all_labels,        all_preds, zero_division=0) * 100
    cm   = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 45)
    print("  Evaluation Results")
    print("=" * 45)
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred:REAL  Pred:FAKE")
    print(f"  Actual:REAL   {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"  Actual:FAKE   {cm[1][0]:>6}     {cm[1][1]:>6}")
    print("=" * 45)


if __name__ == "__main__":
    main()
