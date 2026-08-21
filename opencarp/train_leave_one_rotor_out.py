#!/usr/bin/env python3
"""
Leave-one-rotor-out cross-validation across N independently-induced rotors: for each k, and
each rotor R, train on the pooled data from all OTHER rotors and test on R, then average
across the N folds. This is the properly powered version of opencarp/train_cross_rotor.py's
pairwise check -- pooling N-1 rotors' worth of training data per fold gives a less noisy
estimate than a single train-rotor/test-rotor pair, and is the standard way to evaluate
generalization when you have a handful of independent instances rather than one.

Usage:
    python opencarp/train_leave_one_rotor_out.py \
        --data rotorA=path/to/A/cluster_features.csv \
        --data rotorB=path/to/B/cluster_features.csv \
        --data rotorC=path/to/C/cluster_features.csv \
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


def parse_data_arg(items):
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name] = pd.read_csv(path)
    return out


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
    ap.add_argument("--data", action="append", required=True,
                     help="name=path/to/cluster_features.csv, repeatable (>=3 recommended)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rotors = parse_data_arg(args.data)
    names = list(rotors.keys())
    print(f"Rotors: {names}")
    for name, df in rotors.items():
        n_pos = df.drop_duplicates("node_id")["ablation_target"].sum()
        print(f"  {name}: {n_pos} positive sites")

    ks = sorted(set.intersection(*[set(df["k"].unique()) for df in rotors.values()]))

    results = []
    for k in ks:
        fold_aucs, fold_aps = [], []
        per_fold = {}
        for held_out in names:
            train_df = pd.concat([df for n, df in rotors.items() if n != held_out], ignore_index=True)
            test_df = rotors[held_out]
            auc, ap = fit_eval(train_df, test_df, k, args.seed)
            fold_aucs.append(auc)
            fold_aps.append(ap)
            per_fold[held_out] = {"auc": auc, "ap": ap}
        results.append({
            "k": int(k), "mean_auc": float(np.mean(fold_aucs)), "std_auc": float(np.std(fold_aucs)),
            "mean_ap": float(np.mean(fold_aps)), "std_ap": float(np.std(fold_aps)),
            "per_fold": per_fold,
        })
        fold_str = "  ".join(f"{n}(held out): AUC={per_fold[n]['auc']:.3f}" for n in names)
        print(f"k={k}: mean AUC={np.mean(fold_aucs):.3f}+/-{np.std(fold_aucs):.3f}  "
              f"mean AP={np.mean(fold_aps):.3f}+/-{np.std(fold_aps):.3f}  |  {fold_str}")

    with open(os.path.join(args.out, "leave_one_rotor_out_metrics.json"), "w") as f:
        json.dump({"rotors": names, "features": FEATURES, "results": results}, f, indent=2)

    ks_plot = [r["k"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(ks_plot, [r["mean_auc"] for r in results], yerr=[r["std_auc"] for r in results],
                      marker="o", capsize=3, color="black", label="mean (leave-one-out)")
    for name in names:
        axes[0].plot(ks_plot, [r["per_fold"][name]["auc"] for r in results],
                      marker=".", linestyle=":", alpha=0.6, label=f"{name} held out")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("local electrodes probed (k)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title(f"Leave-one-rotor-out ROC-AUC (N={len(names)} rotors)")
    axes[0].legend(fontsize=7)

    axes[1].errorbar(ks_plot, [r["mean_ap"] for r in results], yerr=[r["std_ap"] for r in results],
                      marker="o", capsize=3, color="black", label="mean (leave-one-out)")
    for name in names:
        axes[1].plot(ks_plot, [r["per_fold"][name]["ap"] for r in results],
                      marker=".", linestyle=":", alpha=0.6, label=f"{name} held out")
    axes[1].set_xlabel("local electrodes probed (k)")
    axes[1].set_ylabel("Average precision")
    axes[1].set_title("Leave-one-rotor-out average precision")
    axes[1].legend(fontsize=7)

    fig.suptitle(f"Generalization across {len(names)} independently-induced rotors "
                 "(train on N-1, test on the held-out one)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "leave_one_rotor_out.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}/leave_one_rotor_out_metrics.json, leave_one_rotor_out.png")


if __name__ == "__main__":
    main()
