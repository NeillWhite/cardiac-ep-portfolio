"""
Evaluate a trained checkpoint on the held-out test split, and write metrics +
a confusion matrix plot to results/.

Usage:
    python scripts/evaluate.py --data data/processed --checkpoint models/baseline.pt
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import PTBXLDataset
from model import ECGConvNet


def plot_confusion_matrix(cm: np.ndarray, classes: list, out_path: str) -> None:
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                     ha="center", va="center", color=text_color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_traces(X_test, y_test, preds, classes, out_path, n_examples=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n_examples, len(X_test)), replace=False)
    fig, axes = plt.subplots(len(idx), 1, figsize=(9, 2.2 * len(idx)), sharex=True)
    if len(idx) == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        lead_ii = X_test[i][:, 1]  # lead II, a standard reference lead for display
        ax.plot(lead_ii, color="#1f4e8c", linewidth=1)
        true_label = classes[y_test[i]]
        pred_label = classes[preds[i]]
        correct = true_label == pred_label
        ax.set_ylabel("Lead II (mV)", fontsize=8)
        ax.set_title(f"true={true_label}  pred={pred_label}  {'✓' if correct else '✗'}",
                      fontsize=9, color="#1a7a3c" if correct else "#b3271e")
    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = np.load(os.path.join(args.data, "X_test.npy"))
    y_test = np.load(os.path.join(args.data, "y_test.npy"))
    with open(os.path.join(args.data, "classes.txt")) as f:
        classes = f.read().splitlines()

    test_ds = PTBXLDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = ECGConvNet(n_leads=X_test.shape[-1], n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"macro_f1": macro_f1, "report": report, "confusion_matrix": cm.tolist()}, f, indent=2)

    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_plot_path)

    traces_plot_path = os.path.join(results_dir, "example_traces.png")
    plot_example_traces(X_test, all_labels, all_preds, classes, traces_plot_path)

    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"\nSaved full metrics to {metrics_path}")
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    print(f"Saved example trace plot to {traces_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--checkpoint", type=str, default="models/baseline.pt")
    args = parser.parse_args()
    main(args)
