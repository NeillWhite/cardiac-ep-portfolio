#!/usr/bin/env python3
"""Phase 4 latency benchmark: the surrogate path vs the biophysics pipeline it replaces.
Runs inside the container."""
import time
import sys
import numpy as np
sys.path.insert(0, "/repo/opencarp/phase4")
from extract_features import load, per_electrode, add_spatial_context, functional_core

cfg = sys.argv[1] if len(sys.argv) > 1 else "/repo/opencarp/runs/phase4/d10"
vm, pts, dt, man = load(cfg)
x0 = vm.shape[1] // 2

reps = 5
t = time.perf_counter()
for _ in range(reps):
    add_spatial_context(per_electrode(vm, pts, dt, x0))
t_feat = (time.perf_counter() - t) / reps

t = time.perf_counter()
for _ in range(reps):
    functional_core(vm, pts)
t_fc = (time.perf_counter() - t) / reps

print(f"config {cfg.split('/')[-1]}: Vm field {vm.shape}")
print(f"  surrogate: feature extraction from the 600 ms window, 676 electrodes : {t_feat*1000:7.0f} ms")
print(f"             + GBT inference (from train.py)                            :     ~2 ms")
print(f"  replaces:  functional-core analysis over the full ~3 s record          : {t_fc*1000:7.0f} ms")
print(f"             + the openCARP simulation to produce that record            : ~86000 ms")
