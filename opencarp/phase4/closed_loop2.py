#!/usr/bin/env python3
"""
Phase 4 Gate 4d, done properly: take an ESTABLISHED rotor (from a re-run PSD config with
a late state checkpoint), branch it forward 1.5 s three ways --
  none  : no lesion            (should sustain)
  pred  : 5 mm lesion at the surrogate's predicted functional core   (should terminate)
  ctrl  : 5 mm lesion ~13 mm away                                    (should sustain)
-- and score termination from the voltage field. Runs inside the container.

  python3 closed_loop2.py <cfg_dir> <pred_x> <pred_y> <ctrl_x> <ctrl_y>
"""
import sys
import os
import glob
import subprocess
import numpy as np
from carputils.carpio import igb

LES_R_UM = 5000.0
RUN_MS = 1500


def load_mesh(prefix):
    pts = np.loadtxt(prefix + ".pts", skiprows=1)
    elems = []
    with open(prefix + ".elem") as f:
        n = int(f.readline())
        for _ in range(n):
            p = f.readline().split()
            elems.append((int(p[1]), int(p[2]), int(p[3])))
    return pts, np.array(elems), pts[np.array(elems)].mean(1)


def base_cmd(txt):
    """Reconstruct the PSD monodomain command, dropping the bits that make it (re)build a
    rotor rather than CONTINUE one: the LAT-prepacing options especially -- with those left
    in, -start_statef is ignored and the rotor is re-seeded from the Eikonal map every time,
    so a lesion has no effect."""
    toks = [t for t in open(txt).read().splitlines() if t]
    out, i = [], 0
    skip = {"-tend", "-num_tsav", "-start_statef", "-simID", "-meshname",
            "-write_statef", "-prepacing_lats", "-prepacing_beats", "-prepacing_bcl"}
    while i < len(toks):
        t = toks[i]
        if t in skip:
            i += 2; continue
        if t.startswith("-tsav["):
            i += 2; continue
        if len(t) >= 2 and t[0] == '"' == t[-1]:
            t = t[1:-1]
        out.append(t); i += 1
    if out and out[0].endswith("openCARP"):
        out = out[1:]
    return out


def alive(vmigb, n_nodes, win=250, swing=40.0):
    d, _, _ = igb.read(vmigb)
    vm = d if d.shape[0] == n_nodes else d.T
    tail = vm[:, -win:]
    return float(np.max(tail.max(1) - tail.min(1))) > swing


def main():
    cfg, px, py, cx, cy = sys.argv[1], *map(float, sys.argv[2:6])
    psd = glob.glob(f"{cfg}/ex/*_PSD")[0]
    mesh = f"{psd}/block_i"
    statef = f"{psd}/state.2500.0"
    assert os.path.isfile(statef + ".roe"), f"no {statef}.roe"
    bc = base_cmd(f"{psd}/freerun_cmd_psd.txt")
    pts, elems, cent = load_mesh(mesh)
    tend = 2500 + RUN_MS

    results = {}
    for name, xy in [("none", None), ("giant", (25.0, 25.0)), ("pred", (px, py)), ("ctrl", (cx, cy))]:
        d = f"{cfg}/cl_{name}"
        os.makedirs(d, exist_ok=True)
        cmd = ["openCARP"] + bc + ["-simID", d, "-start_statef", statef,
                                   "-tend", f"{tend:.1f}", "-meshname", mesh]
        if xy is not None:
            dist = np.hypot(cent[:, 0] - xy[0] * 1000, cent[:, 1] - xy[1] * 1000)
            rr = 16000.0 if name == "giant" else LES_R_UM   # giant = sanity check
            v = np.ones(len(elems)); v[dist <= rr] = 1e-6
            sf = f"{d}/gi.dat"
            with open(sf, "w") as f:
                f.write("1\n"); np.savetxt(f, v, fmt="%.6g")
            cmd += ["-gi_scale_vec", sf]
        r = subprocess.run(cmd, cwd=psd, capture_output=True, text=True)
        vmigb = f"{d}/vm.igb"
        if r.returncode != 0 or not os.path.isfile(vmigb):
            results[name] = "CARP_ERR"
            continue
        results[name] = "sustained" if alive(vmigb, pts.shape[0]) else "TERMINATED"
        try:
            os.remove(vmigb)
        except OSError:
            pass
    print(f"{os.path.basename(cfg):8s}  none={results['none']:10s}  giant={results['giant']:10s}  "
          f"pred={results['pred']:10s}  ctrl={results['ctrl']:10s}", flush=True)


if __name__ == "__main__":
    main()
