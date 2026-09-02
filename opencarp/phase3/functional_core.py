#!/usr/bin/env python3
"""
Print the functional-core centroid (mm, "x y") for a rotor: the centroid of tissue
that activates weakly over one rotor cycle -- i.e. the region the wave circles but
never fully excites. This is the ablation target (see docs/PHASE3_FINDINGS.md),
distinct from the phase-singularity pivot.

  python3 opencarp/phase3/functional_core.py A   ->  "25.4 26.6"
"""
import sys
import numpy as np
from scipy.ndimage import uniform_filter

REPO = "/repo" if __import__("os").path.isdir("/repo/opencarp") else "."


def functional_core(S):
    d = np.load(f"{REPO}/opencarp/runs/phase3/{S}/rotor{S}_phase3.npz")
    vm, pts = d["vm"], d["pts"] / 1000.0
    nx = len(np.unique(pts[:, 0])); ny = len(np.unique(pts[:, 1]))
    X = pts[:, 0].reshape(ny, nx); Y = pts[:, 1].reshape(ny, nx)

    # cycle length from a well-activated far-field point
    far = int(np.argmax((vm.max(1) - vm.min(1)) * (np.hypot(pts[:, 0] - 25, pts[:, 1] - 25) > 18)))
    tr = vm[far] > -20
    acts = np.where((~tr[:-1]) & (tr[1:]))[0]
    acts = acts[np.diff(np.concatenate([[-999], acts])) > 60]
    cl = int(np.median(np.diff(acts))) if len(acts) > 2 else 200

    t0 = vm.shape[1] // 2
    seg = vm[:, t0:t0 + cl]
    ampl = (seg.max(1) - seg.min(1)).reshape(ny, nx)
    a = ampl.copy(); a[a < 12] = np.nan          # blank the always-dead fibrotic holes
    m = ~np.isnan(a)
    sm = (uniform_filter(np.nan_to_num(a), 6)
          / np.maximum(uniform_filter(m.astype(float), 6), 1e-6))
    inpatch = np.hypot(X - 25, Y - 25) < 14.2
    weak = (sm < np.nanpercentile(sm[inpatch], 20)) & inpatch
    return float(X[weak].mean()), float(Y[weak].mean())


if __name__ == "__main__":
    x, y = functional_core(sys.argv[1])
    print(f"{x:.2f} {y:.2f}")
