#!/usr/bin/env python3
"""
Phase 4 surrogate: per-electrode classifier predicting "within R mm of the functional
core" from a short-window electrogram feature set, evaluated leave-one-config-out.
Then, per held-out config, the predicted-positive centroid vs the true functional core
= the localisation error the surrogate achieves without running the biophysics.

Runs in the project venv (no Docker).
  python opencarp/phase4/train.py
"""
import glob
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

R_MM = 5.0
FEATURES = ["uni_amp", "bp_amp", "bpx_amp", "bpy_amp", "n_act", "fractionation", "dom_freq",
            "uni_amp_rel", "uni_amp_rank", "uni_amp_local_min", "uni_amp_nbr_mean",
            "bp_amp_nbr_std", "dist_to_min_amp"]


def load():
    frames = []
    for d in sorted(glob.glob("opencarp/runs/phase4/d*")):
        try:
            df = pd.read_csv(f"{d}/features.csv")
            lab = np.load(f"{d}/label.npz")
        except FileNotFoundError:
            continue
        fc = lab["functional_core"]
        df["config"] = d.split("/")[-1]
        df["fc_x"], df["fc_y"] = float(fc[0]), float(fc[1])
        df["dist_fc"] = np.hypot(df.x_mm - fc[0], df.y_mm - fc[1])
        df["y"] = (df.dist_fc <= R_MM).astype(int)
        frames.append(df)
    if not frames:
        raise SystemExit("no phase4/d* configs with features.csv + label.npz yet")
    return pd.concat(frames, ignore_index=True)


def main():
    data = load()
    configs = sorted(data.config.unique())
    print(f"{len(configs)} configs, {len(data)} electrodes, "
          f"{data.y.mean():.1%} positive (within {R_MM} mm of functional core)\n")

    per_cfg = []
    oof = np.zeros(len(data))
    t_train = t_infer = 0.0
    for held in configs:
        tr = data[data.config != held]
        te = data[data.config == held]
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=250, learning_rate=0.05,
                                             random_state=0)
        t = time.perf_counter(); clf.fit(tr[FEATURES], tr.y); t_train += time.perf_counter() - t
        t = time.perf_counter(); p = clf.predict_proba(te[FEATURES])[:, 1]; t_infer += time.perf_counter() - t
        oof[te.index] = p
        # predicted functional core = probability-weighted centroid of the top electrodes
        k = te.assign(p=p).nlargest(max(3, int(0.1 * len(te))), "p")
        px = np.average(k.x_mm, weights=k.p); py = np.average(k.y_mm, weights=k.p)
        err = float(np.hypot(px - te.fc_x.iloc[0], py - te.fc_y.iloc[0]))
        # baselines
        b_center = float(np.hypot(25 - te.fc_x.iloc[0], 25 - te.fc_y.iloc[0]))
        lowk = te.nsmallest(5, "uni_amp")
        b_amp = float(np.hypot(lowk.x_mm.mean() - te.fc_x.iloc[0], lowk.y_mm.mean() - te.fc_y.iloc[0]))
        per_cfg.append(dict(config=held, pred_x=round(px, 2), pred_y=round(py, 2),
                            true_x=round(te.fc_x.iloc[0], 2), true_y=round(te.fc_y.iloc[0], 2),
                            err_mm=err, base_center_mm=b_center, base_lowamp_mm=b_amp))

    auc = roc_auc_score(data.y, oof)
    ap = average_precision_score(data.y, oof)
    pc = pd.DataFrame(per_cfg)
    print(f"per-electrode leave-one-config-out:  ROC-AUC {auc:.3f}   AP {ap:.3f}\n")
    print("localisation error (mm), median [IQR]:")
    for c in ["err_mm", "base_lowamp_mm", "base_center_mm"]:
        q = pc[c].quantile([.25, .5, .75]).values
        name = {"err_mm": "surrogate", "base_lowamp_mm": "baseline: lowest-5-amplitude centroid",
                "base_center_mm": "baseline: always mesh centre"}[c]
        print(f"  {name:42s} {q[1]:.1f}  [{q[0]:.1f}, {q[2]:.1f}]")
    print(f"\ntiming: fit {t_train/len(configs)*1000:.0f} ms/fold, "
          f"inference {t_infer/len(configs)*1000:.1f} ms/config ({len(data)//len(configs)} electrodes)")

    # permutation importance on the pooled fit (quick, indicative)
    from sklearn.inspection import permutation_importance
    fit_all = HistGradientBoostingClassifier(max_depth=4, max_iter=250, random_state=0).fit(
        data[FEATURES], data.y)
    pi = permutation_importance(fit_all, data[FEATURES], data.y, n_repeats=5, random_state=0,
                                scoring="roc_auc")
    print("\npermutation importance (ROC-AUC drop):")
    for f, v in sorted(zip(FEATURES, pi.importances_mean), key=lambda t: -t[1])[:6]:
        print(f"  {f:22s} {v:+.3f}")

    json.dump(dict(roc_auc=auc, ap=ap,
                   err_median=float(pc.err_mm.median()),
                   base_lowamp_median=float(pc.base_lowamp_mm.median())),
              open("opencarp/runs/phase4/train_summary.json", "w"), indent=1)
    pc.to_csv("opencarp/runs/phase4/per_config_error.csv", index=False)
    print("\nwrote opencarp/runs/phase4/{train_summary.json, per_config_error.csv}")


if __name__ == "__main__":
    main()
