#!/usr/bin/env python3
"""
Phase 4 feature extraction. For one generated config, from a SHORT recording window
(default 600 ms -- a fraction of what the full pipeline needs), compute per-virtual-
electrode features + local spatial context, and write them alongside the ground-truth
functional core (computed from the FULL record -- the "expensive" answer the surrogate
is learning to shortcut).

Runs INSIDE the openCARP container.
  python3 extract_features.py <config_dir>       # e.g. opencarp/runs/phase4/d07
"""
import sys
import os
import numpy as np
from scipy.ndimage import uniform_filter
from carputils.carpio import igb

WIN_MS = 600
STRIDE = 5              # electrode grid: every 5 nodes = 2 mm
BIP = 5                 # bipolar pair offset (2 mm)
DF_BAND = (3.0, 15.0)


def load(cfg_dir):
    man = dict(l.split("=", 1) for l in open(f"{cfg_dir}/manifest.txt").read().splitlines() if "=" in l)
    mesh = man["mesh"]
    vmigb = f"{man['job']}/vm.igb"
    if not os.path.isfile(vmigb):
        vmigb = f"{cfg_dir}/vm_field.igb"
    d, _, t = igb.read(vmigb)
    pts = np.loadtxt(mesh + ".pts", skiprows=1)
    vm = d if d.shape[0] == pts.shape[0] else d.T
    dt = float(t[1] - t[0]) if t is not None and len(t) > 1 else 1.0
    return vm, pts / 1000.0, dt, man


def per_electrode(vm, pts, dt, x0):
    """features from a WIN_MS window starting at frame x0."""
    nx = len(np.unique(pts[:, 0])); ny = len(np.unique(pts[:, 1]))
    seg = vm[:, x0:x0 + int(WIN_MS / dt)]
    fs = 1000.0 / dt

    def nid(ix, iy):
        return iy * nx + ix

    rows = []
    for iy in range(0, ny, STRIDE):
        for ix in range(0, nx, STRIDE):
            n = nid(ix, iy)
            jx = ix + BIP if ix + BIP < nx else ix - BIP
            jy2 = iy + BIP if iy + BIP < ny else iy - BIP
            uni = seg[n]
            bpx = uni - seg[nid(jx, iy)]
            bpy = uni - seg[nid(ix, jy2)]

            uni_amp = float(uni.max() - uni.min())
            bpx_amp = float(bpx.max() - bpx.min())
            bpy_amp = float(bpy.max() - bpy.min())
            bp_amp = 0.5 * (bpx_amp + bpy_amp)

            above = uni > -20
            n_act = int(np.sum((~above[:-1]) & (above[1:])))

            thr = 0.05 * (bpx.max() - bpx.min() + 1e-9)
            sg = np.sign(np.diff(bpx)); sg[np.abs(np.diff(bpx)) < thr] = 0
            nz = sg[sg != 0]
            frac = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0

            spec = np.abs(np.fft.rfft(bpx - bpx.mean()))
            fr = np.fft.rfftfreq(len(bpx), d=1.0 / fs)
            m = (fr >= DF_BAND[0]) & (fr <= DF_BAND[1])
            dfq = float(fr[m][np.argmax(spec[m])]) if m.any() else 0.0

            rows.append(dict(node_id=n, x_mm=pts[n, 0], y_mm=pts[n, 1],
                             uni_amp=uni_amp, bp_amp=bp_amp, bpx_amp=bpx_amp, bpy_amp=bpy_amp,
                             n_act=n_act, fractionation=frac, dom_freq=dfq))
    return rows


def add_spatial_context(rows):
    """position-relative features so the model doesn't just memorise coordinates."""
    x = np.array([r["x_mm"] for r in rows]); y = np.array([r["y_mm"] for r in rows])
    ua = np.array([r["uni_amp"] for r in rows]); ba = np.array([r["bp_amp"] for r in rows])
    for i, r in enumerate(rows):
        d = np.hypot(x - x[i], y - y[i])
        near = (d > 0) & (d <= 6.0)
        r["uni_amp_rel"] = r["uni_amp"] - np.median(ua)          # vs field
        r["uni_amp_rank"] = float(np.mean(ua <= r["uni_amp"]))   # 0 = lowest
        r["uni_amp_local_min"] = float(r["uni_amp"] <= ua[near].min()) if near.any() else 0.0
        r["uni_amp_nbr_mean"] = float(ua[near].mean()) if near.any() else r["uni_amp"]
        r["bp_amp_nbr_std"] = float(ba[near].std()) if near.any() else 0.0
        r["dist_to_min_amp"] = float(d[np.argmin(ua)])           # how far the global amp minimum is
    return rows


def functional_core(vm, pts):
    nx = len(np.unique(pts[:, 0])); ny = len(np.unique(pts[:, 1]))
    X = pts[:, 0].reshape(ny, nx); Y = pts[:, 1].reshape(ny, nx)
    far = int(np.argmax((vm.max(1) - vm.min(1)) * (np.hypot(pts[:, 0] - 25, pts[:, 1] - 25) > 18)))
    tr = vm[far] > -20
    a = np.where((~tr[:-1]) & (tr[1:]))[0]
    a = a[np.diff(np.concatenate([[-999], a])) > 60]
    cl = int(np.median(np.diff(a))) if len(a) > 2 else 160
    t0 = vm.shape[1] // 2
    amp = (vm[:, t0:t0 + cl].max(1) - vm[:, t0:t0 + cl].min(1)).reshape(ny, nx)
    am = amp.copy(); am[am < 12] = np.nan
    m = ~np.isnan(am)
    sm = uniform_filter(np.nan_to_num(am), 6) / np.maximum(uniform_filter(m.astype(float), 6), 1e-6)
    w = sm < np.nanpercentile(sm, 12)
    return float(X[w].mean()), float(Y[w].mean()), cl


def main():
    cfg = sys.argv[1].rstrip("/")
    vm, pts, dt, man = load(cfg)
    x0 = vm.shape[1] // 2                    # mid-record window
    rows = add_spatial_context(per_electrode(vm, pts, dt, x0))
    fcx, fcy, cl = functional_core(vm, pts)
    cols = list(rows[0].keys())
    with open(f"{cfg}/features.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.5g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    np.savez(f"{cfg}/label.npz", functional_core=np.array([fcx, fcy]), cycle_ms=cl,
             config=man.get("config", ""))
    print(f"{os.path.basename(cfg)}: {len(rows)} electrodes, functional core ({fcx:.1f},{fcy:.1f})")


if __name__ == "__main__":
    main()
