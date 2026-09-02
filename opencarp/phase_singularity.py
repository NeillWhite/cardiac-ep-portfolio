#!/usr/bin/env python3
"""
Phase-singularity / rotor-core tracking and ablation-target labeling.

Runs against the output of run_tutorial.sh / the 21_reentry_induction example.
Not part of openCARP/carputils -- this is the new post-processing code described in
docs/IMPLEMENTATION_PLAN.md Sec 4.1 steps 3-6. Intended to run *inside* the openCARP
Docker container (it imports carputils' IGB reader and needs scipy/matplotlib, which
are present there but not assumed on the host):

    docker run --rm -v $(pwd)/runs:/shared docker.opencarp.org/opencarp/opencarp:latest \
        python3 /shared/../phase_singularity.py <sim_dir> <mesh_prefix> [--out OUT_DIR]

Method (standard cardiac optical-mapping phase mapping, not invented for this project):
  1. Hilbert-transform each node's (mean-subtracted) Vm time series -> instantaneous phase.
     (Reflect-padded before the transform so its edge distortion is trimmed away.)
  2. For every elementary unit cell of the regular mesh grid, sum the wrapped phase
     difference around the 4 corners each frame; a total near +-2*pi marks a phase
     singularity (topological charge +-1) -- the rotor core for that frame.
  3. Link per-frame PS into a trajectory: seed from the first singularity that persists
     ~in place (not frame 0), associate each frame within a TIGHT gate (default 1 mm)
     preferring charge continuity, and COAST through detection gaps with a gate that
     widens per missed frame before re-acquiring. See track_trajectory().
  4. Label mesh nodes within a fixed radius of any trajectory point as ablation targets.
"""
import argparse
import os

import numpy as np
from scipy.signal import hilbert

from carputils.carpio import igb


def wrap(phase_diff):
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


def load_vm(igb_path):
    data, header, t = igb.read(igb_path)
    # carputils' igb.read returns DOFs-by-time ("DOFs in lines, time in columns")
    if data.shape[0] < data.shape[1]:
        # sanity: we expect far more spatial nodes than timesteps for this mesh
        pass
    return data, header, np.asarray(t)


def load_grid_points(pts_path):
    pts = np.loadtxt(pts_path, skiprows=1)
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    nx, ny = len(xs), len(ys)
    assert nx * ny == pts.shape[0], (
        f"mesh is not a clean {nx}x{ny} regular grid ({pts.shape[0]} points) -- "
        "this script assumes the reentry_induction example's regular block mesh"
    )
    return pts, nx, ny, xs, ys


def load_elem_tags(elem_path):
    tags = []
    tris = []
    with open(elem_path) as f:
        n = int(f.readline())
        for _ in range(n):
            parts = f.readline().split()
            tris.append([int(parts[1]), int(parts[2]), int(parts[3])])
            tags.append(int(parts[4]))
    return np.array(tris), np.array(tags)


def compute_phase(vm, detrend=True, pad_frac=0.25, pad_max=250):
    """vm: (n_nodes, n_time) -> phase: (n_nodes, n_time) in (-pi, pi].

    The Hilbert transform is non-local and its output is distorted near the ends
    of the signal. Since the tracker seeds from the *first* usable frames, that
    distortion lands exactly where it hurts. Mitigate by reflect-padding the time
    axis before the transform and trimming the pad afterwards, so the retained
    signal's edges are interior to the padded transform.
    """
    x = vm - vm.mean(axis=1, keepdims=True) if detrend else vm
    n_time = x.shape[1]
    pad = min(int(pad_frac * n_time), pad_max)
    if pad > 0:
        xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
        analytic = hilbert(xp, axis=1)[:, pad:pad + n_time]
    else:
        analytic = hilbert(x, axis=1)
    return np.angle(analytic)


def detect_ps_frame(phase_grid, charge_tol=0.15, border=2):
    """phase_grid: (ny, nx) phase at one instant. Returns list of (ix, iy, charge)
    for unit cells whose corner phases wind by ~+-2*pi (a phase singularity),
    excluding a border margin to avoid mesh-edge artifacts."""
    p00 = phase_grid[:-1, :-1]
    p01 = phase_grid[:-1, 1:]
    p11 = phase_grid[1:, 1:]
    p10 = phase_grid[1:, :-1]

    loop = wrap(p01 - p00) + wrap(p11 - p01) + wrap(p10 - p11) + wrap(p00 - p10)
    charge = loop / (2 * np.pi)

    ny_cells, nx_cells = charge.shape
    hits = []
    for iy in range(border, ny_cells - border):
        for ix in range(border, nx_cells - border):
            c = charge[iy, ix]
            if abs(abs(c) - 1.0) < charge_tol:
                hits.append((ix, iy, np.sign(c)))
    return hits


def _frame_candidates(hits, xs, ys):
    """cell (ix, iy) -> (centroid_x, centroid_y, charge)."""
    return [
        (0.5 * (xs[ix] + xs[ix + 1]), 0.5 * (ys[iy] + ys[iy + 1]), charge)
        for (ix, iy, charge) in hits
    ]


def _find_seed(frames, start, persist_frames, seed_radius_um):
    """First candidate (from frame index >= start) that stays within
    `seed_radius_um` of its own position for `persist_frames` consecutive frames.
    Avoids seeding the tracker on a transient / edge-effect detection.
    Returns (frame_index, x, y, charge) or None."""
    for i in range(start, len(frames)):
        _, cands = frames[i]
        for (cx, cy, cch) in cands:
            held = 0
            for j in range(i, min(i + persist_frames, len(frames))):
                _, cj = frames[j]
                if any(np.hypot(x - cx, y - cy) <= seed_radius_um for (x, y, _) in cj):
                    held += 1
                else:
                    break
            if held >= persist_frames:
                return (i, cx, cy, cch)
    return None


def track_trajectory(ps_by_frame, xs, ys,
                     gate_um=1000.0, gate_growth_um=900.0, max_gate_um=5000.0,
                     max_coast_frames=60, seed_persist_frames=10,
                     seed_radius_um=1200.0):
    """Link per-frame phase singularities into a trajectory with a TIGHT per-frame
    gate plus explicit gap handling.

    - Seed from the first singularity that persists ~in place for
      `seed_persist_frames` frames (not frame 0, which is Hilbert-edge noisy).
    - Each frame, associate to the nearest detection within `gate_um`, preferring
      the same charge sign. A detection outside the gate is NOT linked.
    - On a miss, COAST: hold the last position, append nothing, and on later
      frames widen the search gate by `gate_growth_um` per coasted frame (capped
      at `max_gate_um`) so a briefly-lost core can be re-acquired.
    - After `max_coast_frames` with no re-acquisition, end the segment and try to
      re-seed further along; segments are concatenated, with `gap=True` marking
      the first point after any coast or break.

    Returns list of dict(frame, x, y, charge, gap).
    """
    frames = [(fi, _frame_candidates(hits, xs, ys)) for fi, hits in ps_by_frame]
    traj = []

    seed = _find_seed(frames, 0, seed_persist_frames, seed_radius_um)
    if seed is None:  # degenerate: never persistent -- fall back to first hit
        for i, (_, cands) in enumerate(frames):
            if cands:
                seed = (i, cands[0][0], cands[0][1], cands[0][2])
                break
    if seed is None:
        return traj

    i, px, py, pch = seed
    traj.append({"frame": frames[i][0], "x": px, "y": py, "charge": pch,
                 "gap": False, "reseed": False})
    coast = 0
    i += 1
    while i < len(frames):
        fi, cands = frames[i]
        gate = min(gate_um + gate_growth_um * coast, max_gate_um)
        in_gate = [(np.hypot(x - px, y - py), x, y, ch) for (x, y, ch) in cands
                   if np.hypot(x - px, y - py) <= gate]
        if in_gate:
            in_gate.sort(key=lambda t: (t[3] != pch, t[0]))  # same charge first, then nearest
            _, px, py, pch = in_gate[0]
            traj.append({"frame": fi, "x": px, "y": py, "charge": pch,
                         "gap": coast > 0, "reseed": False})
            coast = 0
            i += 1
            continue

        coast += 1
        i += 1
        if coast > max_coast_frames:
            reseed = _find_seed(frames, i, seed_persist_frames, seed_radius_um)
            if reseed is None:
                break
            i, px, py, pch = reseed
            traj.append({"frame": frames[i][0], "x": px, "y": py, "charge": pch,
                         "gap": True, "reseed": True})
            coast = 0
            i += 1
    return traj


def label_ablation_targets(pts, traj, radius_um=3000.0):
    if not traj:
        return np.zeros(pts.shape[0], dtype=bool)
    traj_xy = np.array([[p["x"], p["y"]] for p in traj])
    labels = np.zeros(pts.shape[0], dtype=bool)
    for i, (x, y, _) in enumerate(pts):
        d = np.hypot(traj_xy[:, 0] - x, traj_xy[:, 1] - y)
        labels[i] = d.min() <= radius_um
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir", help="directory containing vm.igb (e.g. point_7830/beat_1)")
    ap.add_argument("mesh_prefix", help="mesh path prefix, e.g. .../block_i")
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--out", default=None, help="output dir (default: sim_dir)")
    ap.add_argument("--radius-um", type=float, default=3000.0,
                     help="ablation-target labeling radius around PS trajectory")
    ap.add_argument("--gate-um", type=float, default=1000.0,
                     help="tight per-frame association gate for the tracker (um)")
    ap.add_argument("--t-dim", type=int, default=None,
                     help="override auto-detected axis if reshape sanity check fails")
    args = ap.parse_args()

    out_dir = args.out or args.sim_dir
    os.makedirs(out_dir, exist_ok=True)

    vm_path = os.path.join(args.sim_dir, args.vm_file)
    data, header, t = load_vm(vm_path)
    print(f"Loaded {vm_path}: raw shape {data.shape}, header t-dim {header.get('t')}")

    pts, nx, ny, xs, ys = load_grid_points(args.mesh_prefix + ".pts")
    tris, tags = load_elem_tags(args.mesh_prefix + ".elem")
    n_nodes = pts.shape[0]

    # Orient as (n_nodes, n_time)
    if data.shape[0] == n_nodes:
        vm = data
    elif data.shape[1] == n_nodes:
        vm = data.T
    else:
        raise ValueError(f"vm.igb data shape {data.shape} doesn't match {n_nodes} mesh nodes")
    n_time = vm.shape[1]
    print(f"vm: {n_nodes} nodes x {n_time} timesteps")

    phase = compute_phase(vm)

    ps_by_frame = []
    for k in range(n_time):
        frame = phase[:, k].reshape(ny, nx)
        hits = detect_ps_frame(frame)
        ps_by_frame.append((k, hits))
    n_frames_with_ps = sum(1 for _, h in ps_by_frame if h)
    print(f"Phase singularities detected in {n_frames_with_ps}/{n_time} frames")

    traj = track_trajectory(ps_by_frame, xs, ys, gate_um=args.gate_um)
    if traj:
        n_coast = sum(1 for p in traj if p["gap"] and not p["reseed"])
        n_reseed = sum(1 for p in traj if p["reseed"])
        span = traj[-1]["frame"] - traj[0]["frame"] + 1
        print(f"Tracked trajectory: {len(traj)} points over frames "
              f"{traj[0]['frame']}-{traj[-1]['frame']} "
              f"({100.0 * len(traj) / span:.0f}% coverage; {n_coast} brief gaps re-acquired, "
              f"{n_reseed} segment re-seeds)")
    else:
        print("Tracked trajectory: 0 points (no persistent singularity found)")

    labels = label_ablation_targets(pts, traj, radius_um=args.radius_um)
    print(f"Ablation-target nodes: {labels.sum()} / {n_nodes} "
          f"({100.0 * labels.sum() / n_nodes:.1f}%)")

    np.savez(
        os.path.join(out_dir, "phase_singularity_results.npz"),
        pts=pts, tris=tris, tags=tags,
        traj_frame=np.array([p["frame"] for p in traj]),
        traj_x=np.array([p["x"] for p in traj]),
        traj_y=np.array([p["y"] for p in traj]),
        traj_charge=np.array([p["charge"] for p in traj]),
        traj_gap=np.array([p["gap"] for p in traj], dtype=bool),
        traj_reseed=np.array([p["reseed"] for p in traj], dtype=bool),
        ablation_target=labels,
        t=t,
        radius_um=args.radius_um,
    )
    print(f"Saved {os.path.join(out_dir, 'phase_singularity_results.npz')}")


if __name__ == "__main__":
    main()
