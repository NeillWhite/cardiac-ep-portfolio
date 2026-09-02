#!/usr/bin/env python3
"""
Phase 3: simulate ablation outcomes.

For a sustained rotor (produced by phase3/induce.sh), branch the simulation from a
mid-rotor state checkpoint, drop a circular non-conductive lesion at each of a grid of
candidate sites, free-run ~1.5 s, and record whether the rotor terminated and how fast.
Output: one row per candidate -> phase3/<site>/sweep_results.csv, consumed by
plot_efficacy.py.

Runs INSIDE the openCARP container:
  docker run --rm -v $(pwd):/repo -w /repo/opencarp \
    docker.opencarp.org/opencarp/opencarp:latest \
    python3 /repo/opencarp/phase3/lesion_sweep.py A --branch-ms 1500 --lesion-radius-um 2500
"""
import argparse
import csv
import os
import subprocess
import sys
import numpy as np

REPO = "/repo"


def read_manifest(path):
    d = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v
    return d


def load_mesh(prefix):
    pts = np.loadtxt(prefix + ".pts", skiprows=1)
    elems = []
    with open(prefix + ".elem") as f:
        n = int(f.readline())
        for _ in range(n):
            p = f.readline().split()
            # p[0]=type (Tr), p[1:4]=node ids, p[4]=tag
            elems.append((int(p[1]), int(p[2]), int(p[3])))
    elems = np.array(elems)
    centroids = pts[elems].mean(axis=1)  # (n_elem, 3)
    return pts, elems, centroids


def write_scale_vec(path, n_elem, lesion_elems, lesion_scale=1e-6):
    """Element-wise conductivity multiplier: 1.0 everywhere, ~0 on the lesion.
    Touches conductivity only -- no tags, no ionic regions -- so the mid-rotor
    state checkpoint still restores cleanly."""
    v = np.ones(n_elem)
    v[lesion_elems] = lesion_scale
    with open(path, "w") as f:
        f.write("1\n")
        np.savetxt(f, v, fmt="%.6g")


def elems_within(centroids, x_um, y_um, r_um):
    d = np.hypot(centroids[:, 0] - x_um, centroids[:, 1] - y_um)
    return np.where(d <= r_um)[0]


def build_free_run_cmd(freerun_cmd_txt):
    """Read the induction's dumped free-run command; strip the run-specific bits we override."""
    toks = [t for t in open(freerun_cmd_txt).read().splitlines() if t != ""]
    out, i = [], 0
    skip_flags = {"-tend", "-num_tsav", "-start_statef", "-simID", "-meshname"}
    while i < len(toks):
        t = toks[i]
        if t in skip_flags:
            i += 2
            continue
        if t.startswith("-tsav["):
            i += 2
            continue
        # carputils dumped some values with literal surrounding quotes ("healthy")
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t = t[1:-1]
        out.append(t)
        i += 1
    # first token is the openCARP executable path from the dump; drop it, caller re-adds
    if out and out[0].endswith("openCARP"):
        out = out[1:]
    return out


def termination_time_ms(vm, win_ms=250, swing_mv=40.0):
    """Frame after which NO node ever again produces an action-potential swing
    (>swing_mv peak-to-peak) within a trailing `win_ms` window. Returns None if
    the rotor is still active at the end of the record (sustained).

    This is tracker-independent: an alive rotor always has some tissue
    depolarising in any >1-cycle window; a terminated one relaxes flat to rest.
    """
    n = vm.shape[1]
    # rolling peak-to-peak over a win_ms window, cheap via cumulative max/min is
    # overkill here -- just scan windows on a stride
    stride = 25
    last_active_end = None
    for end in range(win_ms, n + 1, stride):
        seg = vm[:, end - win_ms:end]
        if float(np.max(seg.max(1) - seg.min(1))) > swing_mv:
            last_active_end = end
    if last_active_end is None:
        return 0
    if last_active_end >= n - stride:
        return None  # still active at the end -> sustained
    return last_active_end - win_ms // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--branch-ms", type=float, default=1500.0,
                    help="ms into the free-run to branch from (must match a saved checkpoint)")
    ap.add_argument("--lesion-radius-um", type=float, default=2500.0)
    ap.add_argument("--run-ms", type=float, default=1500.0, help="post-lesion free-run length")
    ap.add_argument("--grid-mm", type=float, nargs=4, default=None,
                    help="x0 x1 y0 y1 candidate grid extent in mm "
                         "(default: auto -- centred on the core, span core-range + 2*margin)")
    ap.add_argument("--grid-step-mm", type=float, default=5.0)
    ap.add_argument("--grid-margin-mm", type=float, default=10.0,
                    help="auto-grid: how far beyond the core's wander range to extend")
    ap.add_argument("--limit", type=int, default=0, help="only run first N candidates (debug)")
    ap.add_argument("--radius-probe", type=float, nargs="+", default=None,
                    help="instead of the grid, ablate a single point at each of these radii "
                         "(um) -- for calibrating lesion size")
    ap.add_argument("--at-xy", type=float, nargs=2, default=None,
                    help="mm x y -- override the lesion centre (default: phase-singularity "
                         "trajectory mean). Use the functional-core centroid here.")
    ap.add_argument("--tag", default=None, help="label for the output csv (default: mode-derived)")
    ap.add_argument("--keep-vm", action="store_true", help="don't delete vm.igb after scoring")
    ap.add_argument("--only", default=None,
                    help="comma-separated candidate names to run (subset of the grid, e.g. "
                         "'core,g_20_22,g_30_27')")
    args = ap.parse_args()

    base = f"{REPO}/opencarp/runs/phase3/{args.site}"
    man = read_manifest(f"{base}/manifest.txt")
    job, sim, mesh = man["job"], man["sim"], man["mesh"]
    pts, elems, centroids = load_mesh(mesh)

    # reentry timing -> checkpoint file + absolute times
    node, s2, nbeat, bcl = np.loadtxt(f"{job}/reentries.txt").tolist() \
        if os.path.getsize(f"{job}/reentries.txt") else (0, 0, 0, 0)
    t_last_beat = s2 + bcl * nbeat - 10.0            # RP_B: start_S2 + bcl*n_beat - step
    branch_abs = t_last_beat + args.branch_ms
    statef = f"{sim}/state.{branch_abs:.1f}"
    if not os.path.isfile(statef + ".roe"):
        sys.exit(f"no checkpoint at {statef}.roe -- have: "
                 + ", ".join(sorted(os.path.basename(p) for p in
                              __import__('glob').glob(f'{sim}/state.*.roe'))))
    tend = branch_abs + args.run_ms
    base_cmd = build_free_run_cmd(f"{sim}/freerun_cmd.txt")

    # locate the core over the post-lesion window: frame 0 of vm == free-run start, and the
    # branch is `branch_ms` into that, so the window is traj frames [branch_ms, branch_ms+run_ms]
    gt = np.load(f"{sim}/phase_singularity_results.npz")
    tf, tx, ty = gt["traj_frame"], gt["traj_x"], gt["traj_y"]
    win = (tf >= args.branch_ms) & (tf <= args.branch_ms + args.run_ms)
    if win.sum() < 5:
        win = tf >= args.branch_ms
    if args.at_xy:
        core = (args.at_xy[0] * 1000.0, args.at_xy[1] * 1000.0)
        print(f"lesion centre overridden to ({args.at_xy[0]:.1f},{args.at_xy[1]:.1f}) mm")
    else:
        core = (float(np.mean(tx[win])), float(np.mean(ty[win])))
        core_span = (float(np.ptp(tx[win])), float(np.ptp(ty[win])))
        print(f"core over post-lesion window: centroid "
              f"({core[0]/1000:.1f},{core[1]/1000:.1f})mm, "
              f"span {core_span[0]/1000:.1f}x{core_span[1]/1000:.1f}mm")

    if args.radius_probe:
        cands = [(f"r{int(r)}", core[0], core[1], r) for r in args.radius_probe]
    else:
        if args.grid_mm:
            x0, x1, y0, y1 = args.grid_mm
        else:  # auto: centre on the core-path centroid, cover its range + margin
            cx, cy = np.mean(tx) / 1000, np.mean(ty) / 1000
            hx = np.ptp(tx) / 2000 + args.grid_margin_mm
            hy = np.ptp(ty) / 2000 + args.grid_margin_mm
            x0, x1 = round(cx - hx), round(cx + hx)
            y0, y1 = round(cy - hy), round(cy + hy)
            # clamp inside the 50 mm sheet with a small border
            x0, y0 = max(x0, 4), max(y0, 4)
            x1, y1 = min(x1, 46), min(y1, 46)
            print(f"auto grid: x [{x0},{x1}] y [{y0},{y1}] mm, step {args.grid_step_mm}")
        xs = np.arange(x0, x1 + 1e-6, args.grid_step_mm)
        ys = np.arange(y0, y1 + 1e-6, args.grid_step_mm)
        cands = [("core", core[0], core[1], args.lesion_radius_um)]
        for yy in ys:
            for xx in xs:
                cands.append((f"g_{xx:.0f}_{yy:.0f}", xx * 1000, yy * 1000, args.lesion_radius_um))
    if args.only:
        want = set(args.only.split(","))
        cands = [c for c in cands if c[0] in want]
        missing = want - {c[0] for c in cands}
        if missing:
            sys.exit(f"--only names not in the grid: {sorted(missing)}")
    if args.limit:
        cands = cands[:args.limit]

    outdir = f"{base}/sweep"
    os.makedirs(outdir, exist_ok=True)
    tag = args.tag or ("radius_probe" if args.radius_probe
                       else f"grid_r{int(args.lesion_radius_um)}")
    csvp = f"{base}/sweep_{tag}.csv"
    fieldnames = ["name", "x_mm", "y_mm", "r_mm", "n_elems", "status", "terminated", "t_term_ms"]

    # resume: keep rows already scored in a previous (interrupted) run
    rows = []
    done_names = set()
    if os.path.isfile(csvp):
        with open(csvp) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done_names.add(row["name"])
        print(f"resuming: {len(done_names)} candidates already scored in {csvp}")

    def flush_csv():
        with open(csvp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"[{args.site}] branch @ {branch_abs:.0f} ms  (state {os.path.basename(statef)}), "
          f"lesion r={args.lesion_radius_um:.0f} um, {len(cands)} candidates, "
          f"post-lesion run {args.run_ms:.0f} ms -> tend {tend:.0f}")

    for k, (name, xu, yu, r_um) in enumerate(cands):
        if name in done_names:
            print(f"  [{k+1}/{len(cands)}] {name} ... already done, skipping")
            continue
        cdir = f"{outdir}/{name}"
        os.makedirs(cdir, exist_ok=True)
        lesion_elems = elems_within(centroids, xu, yu, r_um)
        scalef = f"{cdir}/lesion_gi_scale.dat"
        write_scale_vec(scalef, len(elems), lesion_elems)

        cmd = ["openCARP"] + base_cmd + [
            "-simID", cdir,
            "-start_statef", statef,
            "-tend", f"{tend:.1f}",
            "-meshname", mesh,
            "-gi_scale_vec", scalef,
        ]
        print(f"  [{k+1}/{len(cands)}] {name}  ({xu/1000:.1f},{yu/1000:.1f})mm r={r_um/1000:.1f}  "
              f"{len(lesion_elems)} elems ... ", end="", flush=True)
        log = open(f"{cdir}/carp.log", "w")
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(mesh))
        if r.returncode != 0:
            print("CARP FAILED (see carp.log)")
            rows.append(dict(name=name, x_mm=xu / 1000, y_mm=yu / 1000, r_mm=r_um / 1000,
                             n_elems=len(lesion_elems), status="carp_error",
                             terminated="", t_term_ms=""))
            flush_csv()
            continue

        vmigb = f"{cdir}/vm.igb"
        # termination decided purely from the voltage field (tracker-independent)
        from carputils.carpio import igb
        data, _, _ = igb.read(vmigb)
        vm = data if data.shape[0] == pts.shape[0] else data.T
        t_term = termination_time_ms(vm)
        terminated = t_term is not None
        print(f"{'TERMINATED @ %dms' % t_term if terminated else 'sustained'}")
        rows.append(dict(name=name, x_mm=xu / 1000, y_mm=yu / 1000, r_mm=r_um / 1000,
                         n_elems=len(lesion_elems), status="ok",
                         terminated=int(terminated),
                         t_term_ms=("" if t_term is None else t_term)))
        flush_csv()
        # keep disk sane: drop the big vm.igb once scored
        if not args.keep_vm:
            try:
                os.remove(vmigb)
            except OSError:
                pass

    flush_csv()
    n_term = sum(int(r["terminated"]) == 1 for r in rows if r["terminated"] != "")
    print(f"\n[{args.site}] done: {n_term}/{len(rows)} candidates terminated the rotor")
    print(f"wrote {csvp}")


if __name__ == "__main__":
    main()
