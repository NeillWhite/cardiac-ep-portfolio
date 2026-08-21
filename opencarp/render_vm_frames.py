#!/usr/bin/env python3
"""
Render every frame of the post-induction Vm window as a PNG, for assembly into a video
(ffmpeg isn't available inside the openCARP Docker image, so that step runs on the host --
see opencarp/make_vm_video.sh).

Run inside the openCARP container:
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/render_vm_frames.py <sim_dir> <mesh_prefix> --out <frames_dir>
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
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--out", default=None, help="output dir for frame_%04d.png")
    ap.add_argument("--stim-node", type=int, default=None)
    ap.add_argument("--patch-center-mm", type=float, nargs=2, default=(25.0, 25.0))
    ap.add_argument("--patch-radius-mm", type=float, default=14.2)
    args = ap.parse_args()

    frames_dir = args.out or os.path.join(args.sim_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    results = np.load(os.path.join(args.sim_dir, args.results))
    pts = results["pts"]
    tris = results["tris"]
    traj_frame = results["traj_frame"]
    traj_x = results["traj_x"]
    traj_y = results["traj_y"]

    triang = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, tris)

    data, header, t = igb.read(os.path.join(args.sim_dir, args.vm_file))
    vm = data if data.shape[0] == pts.shape[0] else data.T
    n_time = vm.shape[1]

    core_by_frame = {}
    for f, x, y in zip(traj_frame, traj_x, traj_y):
        core_by_frame[int(f)] = (x / 1000.0, y / 1000.0)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    tpc = ax.tripcolor(triang, vm[:, 0], cmap="RdBu_r", vmin=-80, vmax=20, shading="gouraud")
    fig.colorbar(tpc, ax=ax, shrink=0.85, label="Vm (mV): red = depolarized, blue = resting")

    patch_circle = matplotlib.patches.Circle(
        args.patch_center_mm, args.patch_radius_mm,
        fill=False, edgecolor="black", linewidth=1.2, linestyle="--",
        label="fibrotic patch boundary")
    ax.add_patch(patch_circle)

    if args.stim_node is not None:
        sx, sy = pts[args.stim_node, 0] / 1000.0, pts[args.stim_node, 1] / 1000.0
        ax.scatter([sx], [sy], marker="x", s=90, c="black", linewidths=2.0,
                   label=f"S2 stimulus site")

    star = ax.scatter([], [], marker="*", s=260, c="lime", edgecolors="black",
                       linewidths=1.0, label="rotor core (phase singularity)", zorder=5)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    title = ax.set_title("")
    fig.tight_layout()

    for k in range(n_time):
        tpc.set_array(vm[:, k])
        if k in core_by_frame:
            cx, cy = core_by_frame[k]
            star.set_offsets([[cx, cy]])
        else:
            star.set_offsets(np.empty((0, 2)))
        t_ms = t[k] if t is not None and len(t) > k else k
        title.set_text(f"t = {t_ms:.0f} ms  (frame {k + 1}/{n_time})")
        fig.savefig(os.path.join(frames_dir, f"frame_{k:04d}.png"), dpi=130)
        if k % 50 == 0:
            print(f"rendered frame {k}/{n_time}")

    plt.close(fig)
    print(f"Done. {n_time} frames written to {frames_dir}")


if __name__ == "__main__":
    main()
