"""
Export a compact JSON bundle of test-set predictions + waveforms for the
interactive ECG viewer artifact.

Reusable across models: this script only depends on a checkpoint file and the
existing ECGConvNet architecture (same pattern as evaluate.py). To view a
different/future model (e.g. a resnet-1d), point --checkpoint at its weights,
adjust the model import/instantiation the same way you would in evaluate.py,
and pass --model-name so the viewer's header reflects it. The output JSON
schema itself does not change, so the viewer HTML does not need to change.

Usage:
    python scripts/export_viewer_data.py --data data/processed --checkpoint models/baseline.pt \
        --metrics results/metrics.json --output results/viewer_data.json \
        --model-name "ECGConvNet (baseline 1D-CNN)"
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import PTBXLDataset
from model import ECGConvNet

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = np.load(os.path.join(args.data, "X_test.npy"))  # (N, time, leads), raw mV
    y_test = np.load(os.path.join(args.data, "y_test.npy"))
    with open(os.path.join(args.data, "classes.txt")) as f:
        classes = f.read().splitlines()

    test_ds = PTBXLDataset(X_test, y_test)
    model = ECGConvNet(n_leads=X_test.shape[-1], n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(test_ds), 128):
            batch = torch.stack([test_ds[j][0] for j in range(i, min(i + 128, len(test_ds)))]).to(device)
            probs = F.softmax(model(batch), dim=1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = all_probs.argmax(axis=1)

    rng = np.random.default_rng(args.seed)
    samples = []
    for class_idx, class_name in enumerate(classes):
        class_indices = np.where(y_test == class_idx)[0]
        chosen = rng.choice(class_indices, size=min(args.samples_per_class, len(class_indices)), replace=False)
        for idx in chosen:
            leads = X_test[idx][::args.downsample, :]  # (T, 12) raw mV, downsampled
            samples.append({
                "id": f"test_{idx:05d}",
                "true": class_name,
                "pred": classes[all_preds[idx]],
                "correct": bool(all_preds[idx] == class_idx),
                "probs": {c: round(float(p), 4) for c, p in zip(classes, all_probs[idx])},
                "leads": {name: [round(float(v), 3) for v in leads[:, i]]
                          for i, name in enumerate(LEAD_NAMES)},
            })

    with open(args.metrics) as f:
        metrics = json.load(f)

    bundle = {
        "model_name": args.model_name,
        "generated": args.run_label,
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": {c: metrics["report"][c]["f1-score"] for c in classes},
        "classes": classes,
        "sample_rate_hz": 100 / args.downsample,
        "samples": samples,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Wrote {len(samples)} samples ({size_mb:.2f} MB) to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--checkpoint", type=str, default="models/baseline.pt")
    parser.add_argument("--metrics", type=str, default="results/metrics.json")
    parser.add_argument("--output", type=str, default="results/viewer_data.json")
    parser.add_argument("--model-name", type=str, default="ECGConvNet (baseline 1D-CNN)")
    parser.add_argument("--run-label", type=str, default="2026-08-18")
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--downsample", type=int, default=2, help="Keep every Nth timepoint")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
