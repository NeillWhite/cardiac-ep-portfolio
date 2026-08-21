#!/usr/bin/env python3
"""
Review-gate visualization for phase_singularity.py output.

Produces:
  - rotor_trajectory.png: full tissue substrate (healthy + fibrotic patch, both
    rendered) with the tracked rotor-core trajectory and the labeled ablation-target
    region.
  - vm_snapshots.png: a handful of Vm frames across the window with the tracked core
    marked, showing the rotor wavefront rotating around the fibrotic patch.

Run inside the openCARP container (matplotlib/numpy already present there):
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/plot_phase_singularity.py <sim_dir> <mesh_prefix> --out <out_dir>
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from carputils.carpio import igb

HEALTHY_COLOR = "#f4e9d8"
FIBROTIC_COLOR = "#9a9088"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("mesh_prefix")
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--stim-node", type=int, default=None,
                     help="mesh node index of the S2 stimulus site, for reference marker")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or args.sim_dir
    os.makedirs(out_dir, exist_ok=True)

    results = np.load(os.path.join(args.sim_dir, args.results))
    pts = results["pts"]
    tris = results["tris"]
    tags = results["tags"]
    traj_frame = results["traj_frame"]
    traj_x = results["traj_x"]
    traj_y = results["traj_y"]
    ablation_target = results["ablation_target"]
    radius_um = float(results["radius_um"])

    is_fibrotic = (tags == tags.max())
    healthy_tri = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, tris[~is_fibrotic])
    fibrotic_tri = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, tris[is_fibrotic])
    full_tri = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, tris)

    data, header, t = igb.read(os.path.join(args.sim_dir, args.vm_file))
    vm = data if data.shape[0] == pts.shape[0] else data.T
    n_time = vm.shape[1]

    def draw_substrate(ax):
        ax.tripcolor(healthy_tri, facecolors=np.zeros(healthy_tri.triangles.shape[0]),
                     cmap=matplotlib.colors.ListedColormap([HEALTHY_COLOR]),
                     vmin=0, vmax=1, shading="flat")
        ax.tripcolor(fibrotic_tri, facecolors=np.zeros(fibrotic_tri.triangles.shape[0]),
                     cmap=matplotlib.colors.ListedColormap([FIBROTIC_COLOR]),
                     vmin=0, vmax=1, shading="flat")

    # ---------- Figure 1: substrate + trajectory + ablation-target labels ----------
    fig, ax = plt.subplots(figsize=(8, 6.5))
    draw_substrate(ax)

    target_pts = pts[ablation_target]
    ax.scatter(target_pts[:, 0] / 1000.0, target_pts[:, 1] / 1000.0,
               s=8, c="tab:red", alpha=0.45,
               label=f"ablation target (within {radius_um / 1000:.1f} mm of core)")

    sc = ax.scatter(traj_x / 1000.0, traj_y / 1000.0, c=traj_frame, cmap="viridis", s=16,
                     label="rotor core position, one dot per ~1ms")

    if args.stim_node is not None:
        sx, sy = pts[args.stim_node, 0] / 1000.0, pts[args.stim_node, 1] / 1000.0
        ax.scatter([sx], [sy], marker="x", s=120, c="black", linewidths=2.5,
                   label=f"S2 stimulus site (node {args.stim_node})")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("time (ms since window start)")

    healthy_patch = matplotlib.patches.Patch(facecolor=HEALTHY_COLOR, edgecolor="0.4",
                                              label="healthy myocardium")
    fibrotic_patch = matplotlib.patches.Patch(facecolor=FIBROTIC_COLOR, edgecolor="0.4",
                                               label="fibrotic patch (low-conductivity substrate)")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend([healthy_patch, fibrotic_patch] + handles,
              ["healthy myocardium", "fibrotic patch (low-conductivity substrate)"] + labels_,
              loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Rotor core trajectory and ablation-target labels\n"
                 "(synthetic 2D tissue patch, not an anatomical chamber -- see README Limitations)",
                 fontsize=10)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, "rotor_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: Vm snapshots with core marker ----------
    n_panels = 6
    frame_idxs = np.linspace(0, n_time - 1, n_panels).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for i, (ax, fidx) in enumerate(zip(axes.ravel(), frame_idxs)):
        vm_frame = vm[:, fidx]
        tpc = ax.tripcolor(full_tri, vm_frame, cmap="RdBu_r", vmin=-80, vmax=20, shading="gouraud")
        mask = traj_frame == fidx
        if mask.any():
            ax.scatter(traj_x[mask] / 1000.0, traj_y[mask] / 1000.0,
                       marker="*", s=220, c="lime", edgecolors="black", linewidths=1.0,
                       label="rotor core (phase singularity)", zorder=5)
        t_label = f"{t[fidx]:.0f} ms" if t is not None and len(t) > fidx else f"frame {fidx}"
        ax.set_title(f"t = {t_label}")
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(tpc, ax=axes.ravel().tolist(), shrink=0.7, label="Vm (mV): red = depolarized, blue = resting")
    fig.suptitle("Induced rotor: transmembrane voltage snapshots with tracked core\n"
                 "(same 5cm x 5cm tissue patch as rotor_trajectory.png, viewed at 6 instants)")
    fig.savefig(os.path.join(out_dir, "vm_snapshots.png"), dpi=150)
    plt.close(fig)

    print("Wrote:")
    print(" ", os.path.join(out_dir, "rotor_trajectory.png"))
    print(" ", os.path.join(out_dir, "vm_snapshots.png"))


if __name__ == "__main__":
    main()
