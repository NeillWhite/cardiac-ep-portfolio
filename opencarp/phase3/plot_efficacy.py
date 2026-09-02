#!/usr/bin/env python3
"""
Phase 3 plots: ablation-efficacy map (does a lesion here terminate the rotor?) and the
lesion-radius calibration curve. Reads the sweep_*.csv files from lesion_sweep.py.
Runs in the normal project venv (no Docker) -- reads the portable rotor .npz for the
core trajectory overlay.

  python opencarp/phase3/plot_efficacy.py A
"""
import argparse
import csv
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--base", default="opencarp/runs/phase3")
    ap.add_argument("--npz", default=None, help="rotor .npz for core-trajectory overlay")
    args = ap.parse_args()
    base = f"{args.base}/{args.site}"
    npz = args.npz or f"{base}/rotor{args.site}_phase3.npz"

    gt = np.load(npz)
    tx, ty = gt["traj_x"] / 1000, gt["traj_y"] / 1000
    # fibrotic patch outline (radius 1.42 cm at mesh centre, per the openCARP example)
    patch_c, patch_r = (25.0, 25.0), 14.2

    figs = []

    # ---- radius calibration curve ----
    rp = f"{base}/sweep_radius_probe.csv"
    if not os.path.isfile(rp):
        rp = f"{base}/sweep_results.csv"  # legacy name
    if os.path.isfile(rp):
        rows = [r for r in read_csv(rp) if r["status"] == "ok"]
        rr = np.array([float(r["r_mm"]) for r in rows])
        term = np.array([int(r["terminated"]) for r in rows])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rr, term, "o-", color="#3b6ea5", ms=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["sustained", "terminated"])
        ax.set_xlabel("lesion radius (mm)")
        ax.set_title(f"rotor {args.site}: lesion at meander centroid vs. radius")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figs.append((fig, f"{base}/efficacy_radius_curve.png"))

    # ---- position efficacy map(s) ----
    mc = (float(np.mean(tx)), float(np.mean(ty)))  # core-path centroid

    for gp in sorted(glob.glob(f"{base}/sweep_grid_r*.csv")):
        rmm = gp.split("_r")[-1].replace(".csv", "").replace("_v2", "")
        rows = [r for r in read_csv(gp) if r["status"] == "ok"]
        is_core = np.array([r["name"] == "core" for r in rows])
        gx = np.array([float(r["x_mm"]) for r in rows])
        gy = np.array([float(r["y_mm"]) for r in rows])
        term = np.array([int(r["terminated"]) for r in rows])
        tt = np.array([float(r["t_term_ms"]) if str(r["t_term_ms"]).replace(".", "").isdigit()
                       else np.nan for r in rows])

        def base_ax(ax):
            ax.plot(tx, ty, "-", color="#666", lw=0.8, alpha=0.8, label="rotor core path (3.8 s)")
            th = np.linspace(0, 2 * np.pi, 120)
            ax.plot(patch_c[0] + patch_r * np.cos(th), patch_c[1] + patch_r * np.sin(th),
                    ":", color="#999", lw=1.2, label="fibrotic patch")
            ax.plot(*mc, "x", color="k", ms=9, mew=2, label="core-path centroid")
            ax.set_aspect("equal"); ax.set_xlim(2, 42); ax.set_ylim(6, 42)
            ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

        # left: terminated y/n
        ax = axes[0]; base_ax(ax)
        for t, col, lab in [(1, "#3f8f4f", "terminated"), (0, "#c44", "sustained")]:
            m = (term == t) & ~is_core
            ax.scatter(gx[m], gy[m], s=300, marker="s", c=col, edgecolor="w", linewidth=1.0,
                       label=f"{lab} ({int((term == t).sum())})")
        if is_core.any():
            ci = np.where(is_core)[0][0]
            ax.scatter(gx[ci], gy[ci], s=430, marker="*",
                       c="#3f8f4f" if term[ci] else "#c44", edgecolor="k", linewidth=1.2,
                       label="tracked core site")
        ax.set_title("does a 6 mm lesion here terminate the rotor?")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        # right: time to termination
        ax = axes[1]; base_ax(ax)
        m = term == 1
        sc = ax.scatter(gx[m], gy[m], s=300, marker="s", c=tt[m], cmap="viridis_r",
                        edgecolor="w", linewidth=1.0, vmin=0, vmax=1200)
        ax.scatter(gx[~m], gy[~m], s=120, marker="s", c="#eee", edgecolor="#ccc")
        ax.set_title("time to termination (ms), terminating sites only")
        plt.colorbar(sc, ax=ax, shrink=0.85, label="ms")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        fig.suptitle(f"Rotor {args.site} ablation-efficacy map  |  lesion radius 6 mm  |  "
                     f"{int(term.sum())}/{len(rows)} positions terminate", y=1.00, fontsize=13)
        fig.tight_layout()
        figs.append((fig, f"{base}/efficacy_map_r{rmm}.png"))
        print(f"[{args.site} r={rmm}um] {int(term.sum())}/{len(rows)} positions terminated")

    for fig, path in figs:
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print("wrote", path)


if __name__ == "__main__":
    main()
