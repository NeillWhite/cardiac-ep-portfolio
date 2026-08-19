"""
Train the ECGConvNet baseline on preprocessed PTB-XL data.

Usage:
    python scripts/train_baseline.py --data data/processed --epochs 30
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import PTBXLDataset
from model import ECGConvNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, correct / total, macro_f1


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, seed: {args.seed}")

    X_train = np.load(os.path.join(args.data, "X_train.npy"))
    y_train = np.load(os.path.join(args.data, "y_train.npy"))
    X_val = np.load(os.path.join(args.data, "X_val.npy"))
    y_val = np.load(os.path.join(args.data, "y_val.npy"))

    train_ds = PTBXLDataset(X_train, y_train)
    val_ds = PTBXLDataset(X_val, y_val)
    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    n_classes = len(np.unique(y_train))
    n_leads = X_train.shape[-1]
    model = ECGConvNet(n_leads=n_leads, n_classes=n_classes).to(device)

    # Class-weighted loss since PTB-XL superclasses are imbalanced (NORM dominates).
    # sqrt(1/count) rather than raw 1/count: the plain inverse-frequency weighting
    # overcorrected for HYP (the smallest class, 3.3% of records) and tanked its
    # precision (0.12) without much recall gain -- see README Verification section,
    # Phase 1 run 2026-08-18.
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / np.sqrt(class_counts), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Track macro-F1 (not val_loss) for both LR scheduling and early stopping/checkpointing:
    # val_loss is computed under the same class weights as training, so it can plateau or
    # stall before macro-F1 (the actual target metric) does -- this is what cut training
    # short in the first Phase 1 run.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    # Resolve models/ relative to the repo root (parent of this scripts/ dir),
    # not the caller's cwd, so this works whether you run from repo root or scripts/.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "baseline.pt")
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_macro_f1)
        print(f"epoch {epoch:3d}/{args.epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no val_macro_f1 improvement in {args.patience} epochs.")
                break

    print(f"\nBest val macro F1: {best_val_macro_f1:.4f}. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop if val_loss doesn't improve for this many epochs")
    args = parser.parse_args()
    main(args)
