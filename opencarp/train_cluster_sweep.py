#!/usr/bin/env python3
"""
Electrode-count sweep: train the same classifier (same spatial split, same model config) at
each k = 1..max(k) from opencarp/extract_cluster_features.py's output, and plot performance
vs. number of local electrodes probed.

Usage:
    python opencarp/train_cluster_sweep.py \
        --data opencarp/runs/.../cluster_features.csv --out results/phase2_reentry_2026-08-20
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = [
    "bipolar_amp_min", "bipolar_amp_mean", "bipolar_amp_std",
    "fractionation_mean", "fractionation_max",
    "lat_spread_ms", "lat_std_ms",
    "dom_freq_mean", "dom_freq_std",
]
BLOCK_SIZE_MM = 5.0
SEED = 42


def spatial_split(df):
    bx = (df["x_mm"] // BLOCK_SIZE_MM).astype(int)
    by = (df["y_mm"] // BLOCK_SIZE_MM).astype(int)
    is_test = (bx + by) % 2 == 0
    return ~is_test, is_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-repeats", type=int, default=10,
                     help="repeat with different spatial-split phase offsets for error bars")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    full = pd.read_csv(args.data)
    ks = sorted(full["k"].unique())

    results = []
    rng = np.random.RandomState(args.seed)
    for k in ks:
        df = full[full["k"] == k].reset_index(drop=True)
        aucs, aps = [], []
        for rep in range(args.n_repeats):
            # jitter the checkerboard phase per repeat for a rough stability estimate
            phase_x, phase_y = rng.randint(0, 1000, size=2)
            bx = ((df["x_mm"] * 10).astype(int) + phase_x) // int(BLOCK_SIZE_MM * 10)
            by = ((df["y_mm"] * 10).astype(int) + phase_y) // int(BLOCK_SIZE_MM * 10)
            is_test = (bx + by) % 2 == 0
            train, test = df[~is_test], df[is_test]
            if train["ablation_target"].sum() == 0 or test["ablation_target"].sum() == 0:
                continue

            X_train, y_train = train[FEATURES].values, train["ablation_target"].values.astype(int)
            X_test, y_test = test[FEATURES].values, test["ablation_target"].values.astype(int)

            pos_rate = y_train.mean()
            sw = np.where(y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate))

            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05,
                                                  random_state=args.seed + rep)
            clf.fit(X_train, y_train, sample_weight=sw)
            y_prob = clf.predict_proba(X_test)[:, 1]

            aucs.append(roc_auc_score(y_test, y_prob))
            aps.append(average_precision_score(y_test, y_prob))

        results.append({
            "k": int(k), "n_splits": len(aucs),
            "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
        })
        print(f"k={k}: ROC-AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  "
              f"AP {np.mean(aps):.3f} +/- {np.std(aps):.3f}  ({len(aucs)}/{args.n_repeats} valid splits)")

    with open(os.path.join(args.out, "cluster_sweep_metrics.json"), "w") as f:
        json.dump({"n_repeats": args.n_repeats, "block_size_mm": BLOCK_SIZE_MM,
                    "features": FEATURES, "results": results}, f, indent=2)

    ks_plot = [r["k"] for r in results]
    auc_mean = [r["roc_auc_mean"] for r in results]
    auc_std = [r["roc_auc_std"] for r in results]
    ap_mean = [r["ap_mean"] for r in results]
    ap_std = [r["ap_std"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(ks_plot, auc_mean, yerr=auc_std, marker="o", capsize=3)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
    axes[0].set_xlabel("local electrodes probed (k)")
    axes[0].set_ylabel("ROC-AUC (mean +/- std over spatial-split repeats)")
    axes[0].set_title("ROC-AUC vs. electrode count")
    axes[0].legend()

    base_rate = full.drop_duplicates("node_id")["ablation_target"].mean()
    axes[1].errorbar(ks_plot, ap_mean, yerr=ap_std, marker="o", capsize=3, color="tab:orange")
    axes[1].axhline(base_rate, color="gray", linestyle="--", linewidth=1, label="base rate")
    axes[1].set_xlabel("local electrodes probed (k)")
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("Average precision vs. electrode count")
    axes[1].legend()

    fig.suptitle("How many local electrodes does rotor-adjacency prediction need?")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cluster_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}/cluster_sweep_metrics.json, cluster_sweep.png")


if __name__ == "__main__":
    main()
