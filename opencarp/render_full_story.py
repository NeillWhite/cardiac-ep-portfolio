#!/usr/bin/env python3
"""
Render a single continuous frame sequence spanning the S2 stimulus firing (vm_prop.igb,
60ms) followed seamlessly by the free-running window used for rotor-core tracking (vm.igb,
400ms) -- i.e. the full story from "stimulus fires at one site" through "sustained rotor",
not just the post-induction tail.

Run inside the openCARP container:
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/render_full_story.py <sim_dir> <mesh_prefix> --out <frames_dir>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("mesh_prefix")
    ap.add_argument("--prop-file", default="vm_prop.igb")
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stim-node", type=int, default=None)
    ap.add_argument("--patch-center-mm", type=float, nargs=2, default=(25.0, 25.0))
    ap.add_argument("--patch-radius-mm", type=float, default=14.2)
    args = ap.parse_args()

    frames_dir = args.out or os.path.join(args.sim_dir, "frames_full")
    os.makedirs(frames_dir, exist_ok=True)

    results = np.load(os.path.join(args.sim_dir, args.results))
    pts = results["pts"]
    tris = results["tris"]
    traj_frame = results["traj_frame"]
    traj_x = results["traj_x"]
    traj_y = results["traj_y"]

    triang = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, tris)

    prop_data, prop_hdr, prop_t = igb.read(os.path.join(args.sim_dir, args.prop_file))
    vm_prop = prop_data if prop_data.shape[0] == pts.shape[0] else prop_data.T

    main_data, main_hdr, main_t = igb.read(os.path.join(args.sim_dir, args.vm_file))
    vm_main = main_data if main_data.shape[0] == pts.shape[0] else main_data.T

    # vm_prop's last frame (t=60ms post-stimulus) is the same instant vm_main's first frame
    # restarts from -- drop the duplicate so the concatenation doesn't stutter.
    n_prop = vm_prop.shape[1] - 1
    n_main = vm_main.shape[1]
    n_total = n_prop + n_main

    vm_combined = np.concatenate([vm_prop[:, :n_prop], vm_main], axis=1)
    # global time in ms since the S2 stimulus fired
    t_combined = np.concatenate([np.arange(n_prop), np.arange(n_main) + n_prop])

    # PS trajectory frame indices are relative to vm_main -- shift onto the combined timeline
    core_by_frame = {}
    for f, x, y in zip(traj_frame, traj_x, traj_y):
        core_by_frame[int(f) + n_prop] = (x / 1000.0, y / 1000.0)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    tpc = ax.tripcolor(triang, vm_combined[:, 0], cmap="RdBu_r", vmin=-80, vmax=20, shading="gouraud")
    fig.colorbar(tpc, ax=ax, shrink=0.85, label="Vm (mV): red = depolarized, blue = resting")

    ax.add_patch(matplotlib.patches.Circle(
        args.patch_center_mm, args.patch_radius_mm,
        fill=False, edgecolor="black", linewidth=1.2, linestyle="--",
        label="fibrotic patch boundary"))

    if args.stim_node is not None:
        sx, sy = pts[args.stim_node, 0] / 1000.0, pts[args.stim_node, 1] / 1000.0
        ax.scatter([sx], [sy], marker="x", s=110, c="black", linewidths=2.2,
                   label="S2 stimulus site", zorder=6)

    star = ax.scatter([], [], marker="*", s=260, c="lime", edgecolors="black",
                       linewidths=1.0, label="rotor core (phase singularity)", zorder=5)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    title = ax.set_title("")
    fig.tight_layout()

    for k in range(n_total):
        tpc.set_array(vm_combined[:, k])
        if k in core_by_frame:
            cx, cy = core_by_frame[k]
            star.set_offsets([[cx, cy]])
        else:
            star.set_offsets(np.empty((0, 2)))
        phase = "stimulus + initial propagation" if k < n_prop else "established rotor tracking"
        title.set_text(f"t = {t_combined[k]:.0f} ms since S2 stimulus  ({phase})")
        fig.savefig(os.path.join(frames_dir, f"frame_{k:04d}.png"), dpi=130)
        if k % 50 == 0:
            print(f"rendered frame {k}/{n_total}")

    plt.close(fig)
    print(f"Done. {n_total} frames written to {frames_dir} "
          f"({n_prop} stimulus/propagation + {n_main} tracked rotor)")


if __name__ == "__main__":
    main()
