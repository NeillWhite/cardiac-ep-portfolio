#!/usr/bin/env python3
"""
The observable clue for WHERE to ablate: not the phase singularity, but the region of
tissue that never fully activates over a rotor cycle ("weak-activation core"). Its
centroid predicts the terminating-lesion cluster for all three rotors.

venv, no Docker.  -> opencarp/runs/phase3/fig_target.png
"""
import csv
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P3 = "opencarp/runs/phase3"
PATCH_C, PATCH_R = (25.0, 25.0), 14.2
GREEN, RED = "#1b7f3b", "#b23b3b"
CYCLE = {"A": (1600, 205), "B": (1600, 201), "C": (1800, 201)}
SUB = {"A": "loose meander", "B": "moderate", "C": "tightly pinned"}


def analyse(S):
    t0, cl = CYCLE[S]
    d = np.load(f"{P3}/{S}/rotor{S}_phase3.npz")
    vm, pts = d["vm"], d["pts"] / 1000.0
    nx = len(np.unique(pts[:, 0])); ny = len(np.unique(pts[:, 1]))
    X = pts[:, 0].reshape(ny, nx); Y = pts[:, 1].reshape(ny, nx)
    seg = vm[:, t0:t0 + cl]
    ampl = (seg.max(1) - seg.min(1)).reshape(ny, nx)     # activation strength this cycle
    # the fibrotic "holes" are always ~0 mV -- not informative; blank them and smooth
    # over a ~2 mm window so what's left is the functional weak zone
    a = ampl.copy()
    a[a < 12] = np.nan
    m = ~np.isnan(a)
    filled = np.where(m, np.nan_to_num(a), 0.0)
    sm = uniform_filter(filled, 6) / np.maximum(uniform_filter(m.astype(float), 6), 1e-6)
    weak = sm < (np.nanpercentile(sm[np.hypot(X - 25, Y - 25) < PATCH_R], 20))
    wc = (float(X[weak].mean()), float(Y[weak].mean()))
    tx, ty = d["traj_x"] / 1000.0, d["traj_y"] / 1000.0
    ps = (float(np.mean(tx)), float(np.mean(ty)))
    rows = [r for r in csv.DictReader(open(f"{P3}/{S}/sweep_grid_r6000.csv"))
            if r["status"] == "ok"]
    gx = np.array([float(r["x_mm"]) for r in rows])
    gy = np.array([float(r["y_mm"]) for r in rows])
    term = np.array([r["terminated"] == "1" for r in rows])
    return X, Y, sm, weak, wc, ps, (tx, ty), gx, gy, term


fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
for ax, S in zip(axes, "ABC"):
    X, Y, sm, weak, wc, ps, (tx, ty), gx, gy, term = analyse(S)
    pm = ax.pcolormesh(X, Y, sm, cmap="cividis", vmin=20, vmax=100, shading="auto",
                       rasterized=True)
    ax.contourf(X, Y, weak.astype(float), levels=[0.5, 1.5], colors=["#ff2d55"], alpha=0.28)
    ax.contour(X, Y, weak.astype(float), levels=[0.5], colors="#ff2d55", linewidths=1.8)
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(PATCH_C[0] + PATCH_R * np.cos(th), PATCH_C[1] + PATCH_R * np.sin(th),
            ls=(0, (4, 3)), lw=1.2, color="w")
    ax.plot(tx, ty, "-", color="#8fd3ff", lw=1.2, alpha=0.8)
    ax.plot(*ps, "o", ms=13, mfc="none", mec="#8fd3ff", mew=2.2,
            label="phase singularity (pivot)")
    ax.plot(*wc, "*", ms=22, color="#ff2d55", mec="w", mew=1.0,
            label="weak-core centroid")
    ax.scatter(gx[~term], gy[~term], s=70, marker="o", facecolor="none",
               edgecolor=RED, linewidth=1.4, zorder=6)
    ax.scatter(gx[term], gy[term], s=130, marker="X", color=GREEN, edgecolor="w",
               linewidth=0.9, zorder=7, label="lesion terminates rotor")
    ax.set_aspect("equal"); ax.set_xlim(6, 44); ax.set_ylim(8, 42)
    ax.set_xlabel("x (mm)")
    d_wc = np.mean([np.hypot(x - wc[0], y - wc[1]) for x, y in zip(gx[term], gy[term])])
    d_ps = np.mean([np.hypot(x - ps[0], y - ps[1]) for x, y in zip(gx[term], gy[term])])
    ax.set_title(f"Rotor {S} - {SUB[S]}\n"
                 f"green X's: {d_wc:.0f} mm from weak core, {d_ps:.0f} mm from pivot",
                 fontsize=10)
axes[0].set_ylabel("y (mm)")
axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
cb = fig.colorbar(pm, ax=list(axes), shrink=0.8, pad=0.02)
cb.set_label("activation strength over one rotor cycle (smoothed max - min $V_m$, mV)\n"
             "dark / red-shaded = the functional core: tissue the wave circles but never fully fires",
             fontsize=8.5)
fig.suptitle("Where to ablate is NOT the phase singularity - it's the centre of the "
             "functional core (the region that never fully activates)", fontsize=12, y=1.02)
fig.savefig(f"{P3}/fig_target.png", dpi=120, bbox_inches="tight")
print("wrote fig_target.png")
