#!/usr/bin/env python3
"""
Show the reentry CIRCUIT (activation sequence over one rotor cycle) for rotors A and C,
with the ablation-outcome markers on top. Answers: what distinguishes the lesion
positions that terminate the rotor from the ones that don't?

Runs in the normal venv (reads the portable .npz -- no Docker).
  python opencarp/phase3/illustrate_circuit.py
-> opencarp/runs/phase3/fig_circuit.png
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P3 = "opencarp/runs/phase3"
PATCH_C, PATCH_R = (25.0, 25.0), 14.2
GREEN, RED = "#1b7f3b", "#b23b3b"


def cycle_activation(npz, t0, cl):
    d = np.load(npz)
    vm, pts = d["vm"], d["pts"] / 1000.0
    nx = len(np.unique(pts[:, 0])); ny = len(np.unique(pts[:, 1]))
    X = pts[:, 0].reshape(ny, nx); Y = pts[:, 1].reshape(ny, nx)
    seg = vm[:, t0:t0 + cl]
    above = seg > -20
    lat = np.full(vm.shape[0], np.nan)
    for i in range(vm.shape[0]):
        cr = np.where((~above[i, :-1]) & (above[i, 1:]))[0]
        if len(cr):
            lat[i] = cr[0]
    return d, X, Y, lat.reshape(ny, nx), nx, ny


def panel(ax, S, t0, cl, subtitle):
    d, X, Y, L, nx, ny = cycle_activation(f"{P3}/{S}/rotor{S}_phase3.npz", t0, cl)
    tx, ty = d["traj_x"] / 1000.0, d["traj_y"] / 1000.0

    pm = ax.pcolormesh(X, Y, L, cmap="twilight", shading="auto", rasterized=True)
    cs = ax.contour(X, Y, L, levels=np.arange(0, cl, 15), colors="k",
                    linewidths=0.5, alpha=0.55)
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(PATCH_C[0] + PATCH_R * np.cos(th), PATCH_C[1] + PATCH_R * np.sin(th),
            ls=(0, (4, 3)), lw=1.3, color="#c81e1e")
    ax.plot(tx, ty, "-", color="#00e0ff", lw=1.6, alpha=0.9)
    ax.plot(np.mean(tx), np.mean(ty), "P", ms=12, color="#00e0ff", mec="k", mew=1)

    rows = [r for r in csv.DictReader(open(f"{P3}/{S}/sweep_grid_r6000.csv"))
            if r["status"] == "ok"]
    gx = np.array([float(r["x_mm"]) for r in rows])
    gy = np.array([float(r["y_mm"]) for r in rows])
    term = np.array([r["terminated"] == "1" for r in rows])
    ax.scatter(gx[~term], gy[~term], s=80, marker="o", facecolor="none",
               edgecolor=RED, linewidth=1.6, zorder=6)
    ax.scatter(gx[term], gy[term], s=150, marker="X", color=GREEN, edgecolor="w",
               linewidth=1.0, zorder=7)
    ax.set_aspect("equal"); ax.set_xlim(4, 46); ax.set_ylim(6, 44)
    ax.set_xlabel("x (mm)")
    ax.set_title(f"Rotor {S} - {subtitle}", fontsize=11)
    return pm


fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
pm = panel(axes[0], "A", 1600, 205, "loose meander: circuit sweeps a wide area")
panel(axes[1], "C", 1800, 201, "pinned: circuit loops around the patch")
axes[0].set_ylabel("y (mm)")

cb = fig.colorbar(pm, ax=axes.tolist(), shrink=0.8, pad=0.02)
cb.set_label("activation time within one rotor cycle (ms)\n"
             "colour cycles once around the circuit; crowded contours = slow conduction",
             fontsize=8.5)
fig.suptitle("The reentry circuit, and the ablation-outcome markers on top\n"
             "cyan line/+ = rotor-core (phase-singularity) path    red dashed = fibrotic patch    "
             "green X = lesion terminates    red O = lesion fails",
             fontsize=11, y=1.02)
fig.savefig(f"{P3}/fig_circuit.png", dpi=120, bbox_inches="tight")
print("wrote fig_circuit.png")
