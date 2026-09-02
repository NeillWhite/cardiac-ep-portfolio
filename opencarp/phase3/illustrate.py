#!/usr/bin/env python3
"""
Phase 3 illustration figures. Runs INSIDE the openCARP container (needs carputils'
IGB reader):

  docker run --rm -v $(pwd):/repo -w /repo/opencarp \
    docker.opencarp.org/opencarp/opencarp:latest \
    python3 /repo/opencarp/phase3/illustrate.py

Fig 1  opencarp/runs/phase3/fig_overview.png
  3 rotors x {spiral-wave snapshot + core trajectory} / {ablation-outcome map}

Fig 2  opencarp/runs/phase3/fig_mechanism_A.png
  rotor A, 3 rows x 4 timepoints: no lesion / lesion ON the core dwell zone
  (rotor relocates, survives) / lesion OFF-core (rotor collapses)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from carputils.carpio import igb

REPO = "/repo"
P3 = f"{REPO}/opencarp/runs/phase3"
VM_CMAP = "magma"           # perceptually uniform, CVD-safe, good for a scalar field
VM_LO, VM_HI = -85, 20      # mV
GREEN, RED = "#1b7f3b", "#b23b3b"
PATCH_C, PATCH_R = (25.0, 25.0), 14.2   # fibrotic disc (mesh units mm)
LABEL = {"A": "A - loose meander", "B": "B - moderate", "C": "C - tightly pinned"}


def load_field(npz_path):
    d = np.load(npz_path)
    pts = d["pts"] / 1000.0                    # -> mm
    nx = len(np.unique(pts[:, 0]))
    ny = len(np.unique(pts[:, 1]))
    X = pts[:, 0].reshape(ny, nx)
    Y = pts[:, 1].reshape(ny, nx)
    return d, X, Y, nx, ny


def vm_from_igb(path, n_nodes):
    data, _, _ = igb.read(path)
    vm = data if data.shape[0] == n_nodes else data.T
    return vm


def draw_patch(ax):
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(PATCH_C[0] + PATCH_R * np.cos(th), PATCH_C[1] + PATCH_R * np.sin(th),
            ls=(0, (4, 3)), lw=1.1, color="#8a8f98", zorder=4)


def snap(ax, X, Y, vm_frame, nx, ny):
    ax.pcolormesh(X, Y, vm_frame.reshape(ny, nx), cmap=VM_CMAP, vmin=VM_LO, vmax=VM_HI,
                  shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlim(0, 50); ax.set_ylim(0, 50)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- Fig 1
def fig_overview():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    for j, S in enumerate("ABC"):
        d, X, Y, nx, ny = load_field(f"{P3}/{S}/rotor{S}_phase3.npz")
        vm = d["vm"]
        tx, ty = d["traj_x"] / 1000.0, d["traj_y"] / 1000.0
        tfr = d["traj_frame"]

        # a frame where the spiral arm is well developed: mid-record
        k = vm.shape[1] // 2
        ax = axes[0, j]
        snap(ax, X, Y, vm[:, k], nx, ny)
        draw_patch(ax)
        ax.plot(tx, ty, "-", color="#8fd3ff", lw=1.6, alpha=0.95, zorder=5)
        ax.plot(tx[0], ty[0], "o", ms=5, color="#8fd3ff", zorder=6)
        ax.set_title(f"Rotor {LABEL[S]}\nspiral wave + rotor-core path (3.8 s)",
                     fontsize=10)

        # outcome map
        ax = axes[1, j]
        ax.set_aspect("equal"); ax.set_xlim(0, 50); ax.set_ylim(0, 50)
        ax.set_xticks([0, 25, 50]); ax.set_yticks([0, 25, 50])
        ax.set_xlabel("x (mm)")
        if j == 0:
            ax.set_ylabel("y (mm)")
        draw_patch(ax)
        ax.plot(tx, ty, "-", color="#3a86c8", lw=1.8, alpha=0.95, zorder=3,
                label="rotor-core path")
        # mark the core's dwell centroid so the offset to the green X's is obvious
        cx, cy = float(np.mean(tx)), float(np.mean(ty))
        ax.plot(cx, cy, "P", ms=11, color="#12324a", mec="w", mew=1.0, zorder=7,
                label="core-path centroid")
        rows = [r for r in csv.DictReader(open(f"{P3}/{S}/sweep_grid_r6000.csv"))
                if r["status"] == "ok"]
        gx = np.array([float(r["x_mm"]) for r in rows])
        gy = np.array([float(r["y_mm"]) for r in rows])
        term = np.array([r["terminated"] == "1" for r in rows])
        ax.scatter(gx[~term], gy[~term], s=95, marker="o", facecolor="none",
                   edgecolor=RED, linewidth=1.7, zorder=5,
                   label=f"lesion here: rotor survives ({int((~term).sum())})")
        ax.scatter(gx[term], gy[term], s=130, marker="X", color=GREEN, edgecolor="w",
                   linewidth=0.9, zorder=6,
                   label=f"lesion here: rotor stops ({int(term.sum())})")
        ax.legend(loc="upper left", fontsize=6.8, framealpha=0.92)
        ax.set_title("does a 6 mm lesion at this spot stop the rotor?", fontsize=10)

    # one shared colorbar for the Vm row
    sm = plt.cm.ScalarMappable(cmap=VM_CMAP, norm=plt.Normalize(VM_LO, VM_HI))
    cb = fig.colorbar(sm, ax=list(axes[0, :]), shrink=0.7, pad=0.02)
    cb.set_label("transmembrane voltage $V_m$ (mV)", fontsize=9)
    fig.suptitle("Phase 3 - three simulated rotors and where ablation terminates each",
                 fontsize=13, y=0.99)
    fig.savefig(f"{P3}/fig_overview.png", dpi=120, bbox_inches="tight")
    print("wrote fig_overview.png")


# ---------------------------------------------------------------- Fig 2
def fig_mechanism():
    S = "A"
    d, X, Y, nx, ny = load_field(f"{P3}/{S}/rotor{S}_phase3.npz")
    n_nodes = d["pts"].shape[0]
    tx, ty = d["traj_x"] / 1000.0, d["traj_y"] / 1000.0

    # sustained lesion sitting on where the core actually dwells; terminating off-core lesion
    on_core = ("g_15_27", 15.0, 27.0)      # sustained
    off_core = ("g_25_27", 25.0, 27.0)     # terminates @ ~225 ms
    lesion_r = 6.0

    runs = [
        ("no lesion\n(rotor keeps spinning)", f"{P3}/A/sweep/CONTROL_nolesion/vm.igb", None),
        ("lesion ON the core dwell zone\n(rotor rides around it, survives)",
         f"{P3}/A/sweep/{on_core[0]}/vm.igb", (on_core[1], on_core[2])),
        ("lesion 10 mm to the side\n(rotor collapses)",
         f"{P3}/A/sweep/{off_core[0]}/vm.igb", (off_core[1], off_core[2])),
    ]
    times_ms = [0, 150, 400, 800]

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for i, (label, path, lesion) in enumerate(runs):
        vm = vm_from_igb(path, n_nodes)
        for jc, t in enumerate(times_ms):
            ax = axes[i, jc]
            k = min(t, vm.shape[1] - 1)
            snap(ax, X, Y, vm[:, k], nx, ny)
            draw_patch(ax)
            # pre-lesion core path + its centroid, faint, for reference
            ax.plot(tx, ty, "-", color="#c7ccd1", lw=0.8, alpha=0.55, zorder=4)
            ax.plot(np.mean(tx), np.mean(ty), "P", ms=8, color="#8fd3ff", mec="w",
                    mew=0.8, zorder=6)
            if lesion is not None:
                ax.add_patch(Circle(lesion, lesion_r, facecolor="#0c0f13", edgecolor="w",
                                    linewidth=1.4, zorder=7))
            if i == 0:
                ax.set_title(f"+{t} ms" if t else "lesion applied (t = 0)", fontsize=10)
            if jc == 0:
                ax.text(-0.10, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=VM_CMAP, norm=plt.Normalize(VM_LO, VM_HI))
    cb = fig.colorbar(sm, ax=list(axes.flat), shrink=0.5, pad=0.02)
    cb.set_label("transmembrane voltage $V_m$ (mV)", fontsize=9)
    fig.suptitle("Rotor A - why ablating the phase singularity fails, and what works\n"
                 "black disc = 6 mm non-conductive lesion   |   grey line = pre-lesion core path   "
                 "|   blue + = core-path centroid",
                 fontsize=12, y=0.985)
    fig.savefig(f"{P3}/fig_mechanism_A.png", dpi=120, bbox_inches="tight")
    print("wrote fig_mechanism_A.png")


if __name__ == "__main__":
    fig_overview()
    fig_mechanism()
