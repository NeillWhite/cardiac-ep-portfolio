#!/usr/bin/env python3
"""
Export one rotor's full data (mesh, raw Vm, phase-singularity ground truth) into a single
portable .npz that plain numpy can load -- no carputils/Docker needed. This is what
opencarp/feature_exploration.ipynb reads, so the notebook can run in the normal project venv.

Run inside the openCARP container (needs carputils' IGB reader):
    docker run --rm -v $(pwd)/runs:/shared -v $(pwd):/opencarp_repo \
        docker.opencarp.org/opencarp/opencarp:latest \
        python3 /opencarp_repo/export_for_notebook.py <sim_dir> <mesh_prefix> --out <out.npz>
"""
import argparse

import numpy as np

from carputils.carpio import igb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("mesh_prefix")
    ap.add_argument("--vm-file", default="vm.igb")
    ap.add_argument("--results", default="phase_singularity_results.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pts = np.loadtxt(args.mesh_prefix + ".pts", skiprows=1)
    tris = []
    tags = []
    with open(args.mesh_prefix + ".elem") as f:
        n = int(f.readline())
        for _ in range(n):
            parts = f.readline().split()
            tris.append([int(parts[1]), int(parts[2]), int(parts[3])])
            tags.append(int(parts[4]))
    tris = np.array(tris)
    tags = np.array(tags)

    data, header, t = igb.read(f"{args.sim_dir}/{args.vm_file}")
    vm = (data if data.shape[0] == pts.shape[0] else data.T).astype(np.float32)

    results = np.load(f"{args.sim_dir}/{args.results}")

    np.savez_compressed(
        args.out,
        pts=pts, tris=tris, tags=tags, t=t, vm=vm,
        traj_frame=results["traj_frame"], traj_x=results["traj_x"],
        traj_y=results["traj_y"], traj_charge=results["traj_charge"],
        ablation_target=results["ablation_target"], radius_um=results["radius_um"],
    )
    print(f"Wrote {args.out}: vm {vm.shape}, {pts.shape[0]} nodes, "
          f"{results['ablation_target'].sum()} ablation-target nodes")


if __name__ == "__main__":
    main()
