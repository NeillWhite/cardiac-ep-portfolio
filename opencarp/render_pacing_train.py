#!/usr/bin/env python3
"""
Render the complete rapid-pacing induction story: all 6 attempted stimuli (BCL 200 down
to 150ms), most of which fail to induce anything, followed seamlessly by the established,
tracked rotor. Each of the 6 beats only has its first 60ms captured (that's all carputils'
RP_B algorithm saves per attempt); between beats there's an untracked gap while carputils
continues watching for reentry before trying a shorter coupling interval -- shown as a
paused "time skip" title card rather than pretended-away.

Run inside the openCARP container:
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/render_pacing_train.py <sim_dir> <archive_dir> <mesh_prefix> --out <frames_dir>
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

# (archived filename, start_S2 ms, bcl ms) for the 5 failed attempts, in order,
# then the winning 6th attempt (unarchived -- still at its canonical path).
FAILED_BEATS = [
    ("01_vm_prop.igb", 2000.0, 200.0, "beat 1/6"),
    ("02_vm_prop.igb", 2190.0, 190.0, "beat 2/6"),
    ("03_vm_prop.igb", 2370.0, 180.0, "beat 3/6"),
    ("04_vm_prop.igb", 2540.0, 170.0, "beat 4/6"),
    ("05_vm_prop.igb", 2700.0, 160.0, "beat 5/6"),
]
WINNING_BEAT_START_S2 = 2850.0
WINNING_BEAT_BCL = 150.0
T0 = 2000.0  # simulation clock value corresponding to "t=0" in our video (prepace end)
PAUSE_FRAMES = 15  # ~0.75s hold at 20fps between beats, to mark the untracked gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir", help="point_7830/beat_1 dir with the winning attempt + vm.igb")
    ap.add_argument("archive_dir", help="dir with 01_vm_prop.igb .. 05_vm_prop.igb")
    ap.add_argument("mesh_prefix")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stim-node", type=int, default=None)
    ap.add_argument("--patch-center-mm", type=float, nargs=2, default=(25.0, 25.0))
    ap.add_argument("--patch-radius-mm", type=float, default=14.2)
    args = ap.parse_args()

    frames_dir = args.out or os.path.join(args.sim_dir, "frames_pacing_train")
    os.makedirs(frames_dir, exist_ok=True)

    results = np.load(os.path.join(args.sim_dir, args.results))
    pts = results["pts"]
    tris = results["tris"]
    traj_frame = results["traj_frame"]
    traj_x = results["traj_x"]
    traj_y = results["traj_y"]

    triang = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, tris)

    def load(path):
        data, hdr, t = igb.read(path)
        vm = data if data.shape[0] == pts.shape[0] else data.T
        return vm

    # Build the full ordered list of (vm_array, global_t_offset_ms, label, is_pause_source)
    segments = []
    for fname, start_s2, bcl, tag in FAILED_BEATS:
        vm = load(os.path.join(args.archive_dir, fname))
        segments.append({
            "vm": vm, "t_offset": start_s2 - T0, "bcl": bcl,
            "label": f"{tag}, BCL={bcl:.0f}ms: no reentry",
        })

    vm_winning_prop = load(os.path.join(args.sim_dir, "vm_prop.igb"))
    segments.append({
        "vm": vm_winning_prop, "t_offset": WINNING_BEAT_START_S2 - T0, "bcl": WINNING_BEAT_BCL,
        "label": f"beat 6/6, BCL={WINNING_BEAT_BCL:.0f}ms: wavebreak forms",
    })

    vm_tracked = load(os.path.join(args.sim_dir, "vm.igb"))
    tracked_t_offset = (WINNING_BEAT_START_S2 - T0) + (vm_winning_prop.shape[1] - 1)

    core_by_global_frame = {}
    for f, x, y in zip(traj_frame, traj_x, traj_y):
        core_by_global_frame[int(round(tracked_t_offset)) + int(f)] = (x / 1000.0, y / 1000.0)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    tpc = ax.tripcolor(triang, segments[0]["vm"][:, 0], cmap="RdBu_r", vmin=-80, vmax=20, shading="gouraud")
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
    title = ax.set_title("", fontsize=9, wrap=True, loc="left")
    fig.subplots_adjust(top=0.86)

    frame_idx = 0

    def save_frame():
        nonlocal frame_idx
        fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), dpi=130)
        frame_idx += 1

    # 6 beat attempts, each with a pause card afterward except the winning one (flows
    # straight into the tracked window).
    for seg_i, seg in enumerate(segments):
        vm = seg["vm"]
        n = vm.shape[1] - 1  # drop last frame; either a pause duplicate or continuous handoff
        is_last_beat = (seg_i == len(segments) - 1)
        for k in range(n):
            tpc.set_array(vm[:, k])
            star.set_offsets(np.empty((0, 2)))
            t_global = seg["t_offset"] + k
            title.set_text(f"t = {t_global:.0f} ms  |  {seg['label']}")
            save_frame()
        if not is_last_beat:
            # hold the last real frame with a "time skip" caption -- carputils keeps
            # watching for up to ~200-600ms here before trying the next interval
            tpc.set_array(vm[:, n])
            star.set_offsets(np.empty((0, 2)))
            t_global = seg["t_offset"] + n
            for _ in range(PAUSE_FRAMES):
                title.set_text(f"t = {t_global:.0f} ms  |  no reentry -- next stimulus soon (gap not tracked)")
                save_frame()
        print(f"beat {seg_i + 1}/{len(segments)} done ({frame_idx} frames so far)")

    # established, tracked rotor -- continuous with the winning beat's last frame
    n_tracked = vm_tracked.shape[1]
    for k in range(n_tracked):
        tpc.set_array(vm_tracked[:, k])
        gframe = int(round(tracked_t_offset)) + k
        if gframe in core_by_global_frame:
            star.set_offsets([core_by_global_frame[gframe]])
        else:
            star.set_offsets(np.empty((0, 2)))
        t_global = tracked_t_offset + k
        title.set_text(f"t = {t_global:.0f} ms  |  established, tracked rotor")
        save_frame()
        if k % 100 == 0:
            print(f"tracked frame {k}/{n_tracked} ({frame_idx} total so far)")

    plt.close(fig)
    print(f"Done. {frame_idx} frames written to {frames_dir}")


if __name__ == "__main__":
    main()
