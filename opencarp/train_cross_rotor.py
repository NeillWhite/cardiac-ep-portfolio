#!/usr/bin/env python3
"""
Cross-rotor generalization check: train the classifier on one independently-induced rotor's
electrode features, evaluate on a second, genuinely separate rotor (different stimulus site,
same substrate) -- and vice versa. This is the real test of whether the electrode-count-sweep
finding (opencarp/train_cluster_sweep.py) generalizes, versus just interpolating within one
rotor's spatially-correlated data (which the checkerboard split in that script controls for,
but only within a single simulation instance).

Usage:
    python opencarp/train_cross_rotor.py \
        --data-a opencarp/runs/reentry_induction/2026-08-20_RP_B_final/point_7830/beat_1/cluster_features.csv \
        --data-b opencarp/runs/reentry_induction_site2/2026-08-21_RP_B_site2/point_7919/beat_1/cluster_features.csv \
        --out results/phase2_reentry_2026-08-20
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
SEED = 42


def fit_eval(train_df, test_df, k, seed):
    tr = train_df[train_df["k"] == k]
    te = test_df[test_df["k"] == k]
    X_train, y_train = tr[FEATURES].values, tr["ablation_target"].values.astype(int)
    X_test, y_test = te[FEATURES].values, te["ablation_target"].values.astype(int)

    pos_rate = y_train.mean()
    sw = np.where(y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate))
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
    clf.fit(X_train, y_train, sample_weight=sw)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), average_precision_score(y_test, y_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-a", required=True)
    ap.add_argument("--data-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df_a = pd.read_csv(args.data_a)
    df_b = pd.read_csv(args.data_b)
    ks = sorted(set(df_a["k"].unique()) & set(df_b["k"].unique()))

    print(f"Rotor A: {df_a.drop_duplicates('node_id')['ablation_target'].sum()} positive sites")
    print(f"Rotor B: {df_b.drop_duplicates('node_id')['ablation_target'].sum()} positive sites")

    results = []
    for k in ks:
        auc_ab, ap_ab = fit_eval(df_a, df_b, k, args.seed)  # train A, test B
        auc_ba, ap_ba = fit_eval(df_b, df_a, k, args.seed)  # train B, test A
        results.append({
            "k": int(k),
            "train_A_test_B_auc": auc_ab, "train_A_test_B_ap": ap_ab,
            "train_B_test_A_auc": auc_ba, "train_B_test_A_ap": ap_ba,
            "mean_cross_auc": float(np.mean([auc_ab, auc_ba])),
            "mean_cross_ap": float(np.mean([ap_ab, ap_ba])),
        })
        print(f"k={k}: train A->test B AUC={auc_ab:.3f} AP={ap_ab:.3f}  |  "
              f"train B->test A AUC={auc_ba:.3f} AP={ap_ba:.3f}")

    with open(os.path.join(args.out, "cross_rotor_metrics.json"), "w") as f:
        json.dump({"features": FEATURES, "results": results}, f, indent=2)

    ks_plot = [r["k"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ks_plot, [r["train_A_test_B_auc"] for r in results], marker="o", label="train rotor A -> test rotor B")
    axes[0].plot(ks_plot, [r["train_B_test_A_auc"] for r in results], marker="s", label="train rotor B -> test rotor A")
    axes[0].plot(ks_plot, [r["mean_cross_auc"] for r in results], marker="^", linestyle="--",
                 color="black", label="mean (both directions)")
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("local electrodes probed (k)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Cross-rotor ROC-AUC")
    axes[0].legend(fontsize=8)

    axes[1].plot(ks_plot, [r["train_A_test_B_ap"] for r in results], marker="o", label="train A -> test B")
    axes[1].plot(ks_plot, [r["train_B_test_A_ap"] for r in results], marker="s", label="train B -> test A")
    axes[1].plot(ks_plot, [r["mean_cross_ap"] for r in results], marker="^", linestyle="--",
                 color="black", label="mean")
    axes[1].set_xlabel("local electrodes probed (k)")
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("Cross-rotor average precision")
    axes[1].legend(fontsize=8)

    fig.suptitle("Cross-rotor generalization: train on one independently-induced rotor, test on another")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cross_rotor_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}/cross_rotor_metrics.json, cross_rotor_sweep.png")


if __name__ == "__main__":
    main()
