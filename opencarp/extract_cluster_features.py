#!/usr/bin/env python3
"""
Electrode-count sweep: for each candidate site, build a small local cluster of neighbor
electrodes (2mm spacing, added one direction at a time: E, N, W, S, then the 4 diagonals --
up to 8 neighbors, mimicking a small grid/basket mapping catheter), and compute AGGREGATE
electrogram features (min/mean/std of bipolar amplitude and fractionation, spread/std of
local activation time, mean/std of dominant frequency) using only the first k neighbors, for
every k = 1..8. Feature *count* stays fixed as k grows -- only how many electrodes' worth of
spatial information feeds each aggregate statistic changes -- so classifier performance vs.
k isolates the effect of "how many local electrodes do we need to probe."

Rationale (this is new code for docs/IMPLEMENTATION_PLAN.md Sec 4.2, an evolution of
opencarp/extract_electrode_features.py after review found single fixed-direction-bipole
features didn't separate ablation-target sites from background, ROC-AUC 0.609): a phase
singularity is inherently a *relational* concept (phase winds around a loop of neighboring
points), so a single point's own waveform is a weak proxy for it. LAT spread across a small
local cluster directly captures "this cluster's activation isn't behaving like a smooth
planar wavefront" -- the actual electrophysiological signature of being near a wavebreak/core
-- without needing the full dense-mesh phase map (see README for why that would be
tautological to use directly).

Writes a long-format CSV (one row per candidate-site x k) so the downstream classifier
script can slice by k.

Run inside the openCARP container:
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/extract_cluster_features.py <sim_dir> <mesh_prefix> --out <csv_path>
"""
import argparse
import os

import numpy as np
import pandas as pd

from carputils.carpio import igb

OFFSET_NODES = 5  # 5 * 400um resolution = 2mm, typical clinical electrode spacing
# (dx, dy) in mesh-node steps, in the order neighbors get added as k grows: 4 cardinal
# directions first (an "omnidirectional bipolar" set at k=4), then the 4 diagonals.
DIRECTIONS = [
    (OFFSET_NODES, 0), (0, OFFSET_NODES), (-OFFSET_NODES, 0), (0, -OFFSET_NODES),
    (OFFSET_NODES, OFFSET_NODES), (-OFFSET_NODES, OFFSET_NODES),
    (-OFFSET_NODES, -OFFSET_NODES), (OFFSET_NODES, -OFFSET_NODES),
]
FRACTIONATION_DERIV_THRESHOLD_FRAC = 0.05
DF_BAND_HZ = (3.0, 15.0)
STRIDE = 5  # candidate-site grid spacing in mesh nodes (2mm)


def load_grid(mesh_prefix):
    pts = np.loadtxt(mesh_prefix + ".pts", skiprows=1)
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    nx, ny = len(xs), len(ys)
    assert nx * ny == pts.shape[0]
    return pts, nx, ny


def clip_dir(ix, iy, dx, dy, nx, ny):
    """Mirror an offset back into bounds if it would fall off the mesh edge."""
    jx, jy = ix + dx, iy + dy
    if jx < 0 or jx >= nx:
        jx = ix - dx
    if jy < 0 or jy >= ny:
        jy = iy - dy
    jx = min(max(jx, 0), nx - 1)
    jy = min(max(jy, 0), ny - 1)
    return jx, jy


def bipolar_amplitude(unipolar_a, unipolar_b):
    bp = unipolar_a - unipolar_b
    return float(bp.max() - bp.min()), bp


def fractionation(bp):
    thresh = FRACTIONATION_DERIV_THRESHOLD_FRAC * (bp.max() - bp.min() + 1e-9)
    d = np.diff(bp)
    signs = np.sign(d)
    signs[np.abs(d) < thresh] = 0
    nz = signs[signs != 0]
    return int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0


def dominant_freq(bp, fs_hz):
    n = len(bp)
    spectrum = np.abs(np.fft.rfft(bp - bp.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    band = (freqs >= DF_BAND_HZ[0]) & (freqs <= DF_BAND_HZ[1])
    return float(freqs[band][np.argmax(spectrum[band])]) if band.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("mesh_prefix")
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_csv = args.out or os.path.join(args.sim_dir, "cluster_features.csv")

    pts, nx, ny = load_grid(args.mesh_prefix)
    results = np.load(os.path.join(args.sim_dir, args.results))
    ablation_target = results["ablation_target"]

    data, header, t = igb.read(os.path.join(args.sim_dir, args.vm_file))
    vm = data if data.shape[0] == pts.shape[0] else data.T
    dt_ms = float(t[1] - t[0]) if t is not None and len(t) > 1 else 1.0
    fs_hz = 1000.0 / dt_ms

    # Precompute unipolar LAT (steepest downstroke) for every mesh node once.
    d_all = np.diff(vm, axis=1)
    lat_idx_all = np.argmin(d_all, axis=1)
    lat_all_ms = lat_idx_all * dt_ms

    def node_id(ix, iy):
        return iy * nx + ix

    rows = []
    for iy in range(0, ny, STRIDE):
        for ix in range(0, nx, STRIDE):
            cand_id = node_id(ix, iy)
            uni_c = vm[cand_id, :]
            lat_c = lat_all_ms[cand_id]

            neighbor_ids = []
            for dx, dy in DIRECTIONS[:args.max_k]:
                jx, jy = clip_dir(ix, iy, dx, dy, nx, ny)
                neighbor_ids.append(node_id(jx, jy))

            bp_amps, fracs, dfs, lats = [], [], [], [lat_c]
            for k in range(1, args.max_k + 1):
                nb_id = neighbor_ids[k - 1]
                amp, bp = bipolar_amplitude(uni_c, vm[nb_id, :])
                bp_amps.append(amp)
                fracs.append(fractionation(bp))
                dfs.append(dominant_freq(bp, fs_hz))
                lats.append(lat_all_ms[nb_id])

                lats_arr = np.array(lats)
                rows.append({
                    "node_id": cand_id, "x_mm": pts[cand_id, 0] / 1000.0,
                    "y_mm": pts[cand_id, 1] / 1000.0, "k": k,
                    "bipolar_amp_min": float(np.min(bp_amps)),
                    "bipolar_amp_mean": float(np.mean(bp_amps)),
                    "bipolar_amp_std": float(np.std(bp_amps)),
                    "fractionation_mean": float(np.mean(fracs)),
                    "fractionation_max": float(np.max(fracs)),
                    "lat_spread_ms": float(lats_arr.max() - lats_arr.min()),
                    "lat_std_ms": float(lats_arr.std()),
                    "dom_freq_mean": float(np.mean(dfs)),
                    "dom_freq_std": float(np.std(dfs)),
                    "ablation_target": bool(ablation_target[cand_id]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    n_sites = df["node_id"].nunique()
    print(f"{n_sites} candidate sites x k=1..{args.max_k} -> {len(df)} rows")
    print(f"Positive rate: {df.drop_duplicates('node_id')['ablation_target'].mean():.3f}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
