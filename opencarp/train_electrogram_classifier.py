#!/usr/bin/env python3
"""
Train a gradient-boosted-tree classifier to predict rotor-adjacency (ablation-target label)
from local electrogram features alone (plan docs/IMPLEMENTATION_PLAN.md Sec 4.2, step 8).

Deliberately not deep learning on raw waveforms -- a small, interpretable GBT on a handful
of standard EP signal features (bipolar amplitude, fractionation, LAT, dominant frequency)
is the whole point per the plan: this is the model whose feature importances a domain expert
can sanity-check.

Runs on the host (this project's normal venv has scikit-learn; the openCARP container does
not, and doesn't need to for this step -- see opencarp/extract_electrode_features.py for the
IGB-reading half of the pipeline, which does run in the container).

Important caveat handled here, not glossed over: all 676 samples come from ONE simulated
rotor on ONE mesh, so nearby electrodes are highly spatially correlated. A naive random
train/test split would let near-duplicate neighbors leak across the split and overstate
performance. Instead this uses a spatial checkerboard split (5mm blocks) so no test point's
immediate neighborhood is in the training set.

Usage:
    python opencarp/train_electrogram_classifier.py \
        --data opencarp/runs/reentry_induction/2026-08-20_RP_B_final/point_7830/beat_1/electrode_features.csv \
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

SINGLE_SITE_FEATURES = ["bipolar_amplitude_mv", "fractionation", "lat_ms", "dominant_freq_hz"]
CLUSTER_FEATURES = [
    "bipolar_amp_min", "bipolar_amp_mean", "bipolar_amp_std",
    "fractionation_mean", "fractionation_max",
    "lat_spread_ms", "lat_std_ms",
    "dom_freq_mean", "dom_freq_std",
]
BLOCK_SIZE_MM = 5.0
SEED = 42


def spatial_split(df, block_size_mm=BLOCK_SIZE_MM):
    """Checkerboard split by spatial block so train/test are not immediate neighbors."""
    bx = (df["x_mm"] // block_size_mm).astype(int)
    by = (df["y_mm"] // block_size_mm).astype(int)
    is_test = (bx + by) % 2 == 0
    return ~is_test, is_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=int, default=None,
                     help="if the CSV has a 'k' column (cluster-feature format), filter to this k")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    if "k" in df.columns:
        k = args.k if args.k is not None else int(df["k"].max())
        df = df[df["k"] == k].reset_index(drop=True)
        FEATURES = CLUSTER_FEATURES
        print(f"Using cluster features at k={k}")
    else:
        FEATURES = SINGLE_SITE_FEATURES
    print(f"Loaded {len(df)} electrode samples, "
          f"{df['ablation_target'].sum()} positive ({df['ablation_target'].mean():.1%})")

    train_mask, test_mask = spatial_split(df)
    train, test = df[train_mask], df[test_mask]
    print(f"Spatial checkerboard split: {len(train)} train / {len(test)} test "
          f"(test positive rate {test['ablation_target'].mean():.1%})")

    X_train, y_train = train[FEATURES].values, train["ablation_target"].values.astype(int)
    X_test, y_test = test[FEATURES].values, test["ablation_target"].values.astype(int)

    # class-weighted sample weights (positive class is ~5% of data) -- same rationale as
    # Phase 1's class-weighted loss (README Sec 6): don't let the majority class dominate
    pos_rate = y_train.mean()
    sample_weight = np.where(y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate))

    clf = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.05, random_state=args.seed,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    y_prob = clf.predict_proba(X_test)[:, 1]

    # The default 0.5 probability threshold is not a sensible operating point here: the
    # aggressive class-reweighting needed for ~5% positive prevalence shifts calibration,
    # so .predict()'s hard 0/1 output ends up spatially scattered (visually obvious in
    # classifier_spatial_check.png) even when ranking quality (ROC-AUC/AP) is good. Instead,
    # flag the top train_positive_rate fraction of test electrodes by predicted probability
    # -- a threshold matched to known prevalence, standard practice for imbalanced problems,
    # and the practically relevant framing anyway ("check the top-N riskiest sites").
    threshold = np.quantile(y_prob, 1 - pos_rate)
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(y_test, y_pred, target_names=["background", "ablation_target"],
                                    output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else float("nan")
    ap_score = average_precision_score(y_test, y_prob) if y_test.sum() > 0 else float("nan")
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(json.dumps(report, indent=2))
    print(f"ROC-AUC: {roc_auc:.3f}  Average Precision: {ap_score:.3f}")
    print(f"Confusion matrix: {cm}")

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "train_positive_rate": float(pos_rate), "test_positive_rate": float(y_test.mean()),
        "classification_report": report,
        "roc_auc": roc_auc, "average_precision": ap_score, "confusion_matrix": cm,
        "decision_threshold": float(threshold),
        "threshold_rule": "prevalence-matched (top train_positive_rate fraction by probability)",
        "features": FEATURES, "split": "spatial checkerboard, 5mm blocks", "seed": args.seed,
    }
    with open(os.path.join(args.out, "classifier_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                            display_labels=["background", "ablation\ntarget"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix (spatial test set)")

    if y_test.sum() > 0:
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1])
    axes[1].set_title(f"ROC curve (AUC={roc_auc:.3f})")

    importances = clf.feature_importances_ if hasattr(clf, "feature_importances_") else None
    if importances is None:
        # HistGradientBoostingClassifier has no built-in feature_importances_; use permutation
        from sklearn.inspection import permutation_importance
        result = permutation_importance(clf, X_test, y_test, n_repeats=20, random_state=args.seed)
        importances = result.importances_mean
    axes[2].barh(FEATURES, importances)
    axes[2].set_title("Feature importance")
    axes[2].set_xlabel("importance")

    fig.suptitle("Rotor-adjacency classifier (electrogram features -> ablation-target label)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "classifier_results.png"), dpi=150)
    plt.close(fig)

    # spatial map: predictions vs ground truth, so a mismatch is visually obvious
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, title in [(axes[0], "ablation_target", "Ground truth"),
                            (axes[1], None, "Test-set prediction")]:
        ax.scatter(train["x_mm"], train["y_mm"], c="lightgray", s=15, label="train (not shown as pred)")
        if col is not None:
            ax.scatter(test["x_mm"], test["y_mm"], c=test[col], cmap="coolwarm", s=20, vmin=0, vmax=1)
        else:
            ax.scatter(test["x_mm"], test["y_mm"], c=y_pred, cmap="coolwarm", s=20, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    fig.suptitle("Spatial check: test-set predictions vs. ground truth ablation-target labels")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "classifier_spatial_check.png"), dpi=150)
    plt.close(fig)

    print(f"Wrote {args.out}/classifier_metrics.json, classifier_results.png, classifier_spatial_check.png")


if __name__ == "__main__":
    main()
