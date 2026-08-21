#!/usr/bin/env python3
"""
Virtual electrode sampling + electrogram feature extraction (plan docs/IMPLEMENTATION_PLAN.md
Sec 4.2-4.3 step 7). New code, not part of openCARP/carputils.

Simplifying assumption (documented in README Limitations): this is a monodomain-only
simulation, so there is no true extracellular potential (phie) to sample. Unipolar EGM is
approximated as the local transmembrane voltage Vm(t) -- standard practice for
monodomain-only feature-engineering studies where full bidomain/lead-field forward modeling
is out of scope. Bipolar EGM = difference of two such "unipolar" signals ~2mm apart, which
is exactly how real bipolar electrograms are derived from two unipolar ones, so that part is
not an approximation.

Run inside the openCARP container:
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/extract_electrode_features.py <sim_dir> <mesh_prefix> --out <csv_path>
"""
import argparse
import os

import numpy as np
import pandas as pd

from carputils.carpio import igb

BIPOLAR_OFFSET_NODES = 5  # 5 * 400um resolution = 2mm, typical clinical bipolar spacing
FRACTIONATION_DERIV_THRESHOLD_FRAC = 0.05  # fraction of signal's own amplitude range
DF_BAND_HZ = (3.0, 15.0)  # standard AF dominant-frequency search band (literature convention);
# an unrestricted FFT argmax picks up high-frequency edge content from sharp, largely
# non-periodic deflections (electrodes the wavefront only crosses once or twice), not
# genuine periodicity -- restricting to the physiological band is the standard fix.


def load_grid(mesh_prefix):
    pts = np.loadtxt(mesh_prefix + ".pts", skiprows=1)
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    nx, ny = len(xs), len(ys)
    assert nx * ny == pts.shape[0]
    return pts, nx, ny


def compute_features(unipolar, bipolar, dt_ms):
    fs_hz = 1000.0 / dt_ms

    bp_amp = float(bipolar.max() - bipolar.min())

    d = np.diff(unipolar)
    lat_idx = int(np.argmin(d))  # steepest downstroke = activation
    lat_ms = lat_idx * dt_ms

    thresh = FRACTIONATION_DERIV_THRESHOLD_FRAC * (bipolar.max() - bipolar.min() + 1e-9)
    bp_d = np.diff(bipolar)
    signs = np.sign(bp_d)
    signs[np.abs(bp_d) < thresh] = 0
    nonzero = signs[signs != 0]
    fractionation = int(np.sum(np.diff(nonzero) != 0)) if len(nonzero) > 1 else 0

    n = len(bipolar)
    spectrum = np.abs(np.fft.rfft(bipolar - bipolar.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    band = (freqs >= DF_BAND_HZ[0]) & (freqs <= DF_BAND_HZ[1])
    if band.any():
        dom_freq_hz = float(freqs[band][np.argmax(spectrum[band])])
    else:
        dom_freq_hz = 0.0

    return bp_amp, fractionation, lat_ms, dom_freq_hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("mesh_prefix")
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--stride", type=int, default=5, help="electrode grid stride in mesh nodes (5 = 2mm)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_csv = args.out or os.path.join(args.sim_dir, "electrode_features.csv")

    pts, nx, ny = load_grid(args.mesh_prefix)
    results = np.load(os.path.join(args.sim_dir, args.results))
    ablation_target = results["ablation_target"]

    data, header, t = igb.read(os.path.join(args.sim_dir, args.vm_file))
    vm = data if data.shape[0] == pts.shape[0] else data.T
    dt_ms = float(t[1] - t[0]) if t is not None and len(t) > 1 else 1.0

    def node_id(ix, iy):
        return iy * nx + ix

    rows = []
    for iy in range(0, ny, args.stride):
        for ix in range(0, nx, args.stride):
            nid = node_id(ix, iy)
            jx = ix + BIPOLAR_OFFSET_NODES
            if jx >= nx:
                jx = ix - BIPOLAR_OFFSET_NODES
            if jx < 0:
                continue  # domain too small in this direction; skip (shouldn't happen here)
            nid2 = node_id(jx, iy)

            unipolar = vm[nid, :]
            unipolar2 = vm[nid2, :]
            bipolar = unipolar - unipolar2

            bp_amp, fractionation, lat_ms, dom_freq = compute_features(unipolar, bipolar, dt_ms)

            rows.append({
                "node_id": nid,
                "x_mm": pts[nid, 0] / 1000.0,
                "y_mm": pts[nid, 1] / 1000.0,
                "bipolar_amplitude_mv": bp_amp,
                "fractionation": fractionation,
                "lat_ms": lat_ms,
                "dominant_freq_hz": dom_freq,
                "ablation_target": bool(ablation_target[nid]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Sampled {len(df)} virtual electrodes ({args.stride * 0.4:.1f}mm spacing)")
    print(f"Positive (ablation target) rate: {df['ablation_target'].mean():.3f}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
