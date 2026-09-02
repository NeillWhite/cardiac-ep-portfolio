#!/bin/bash
# Phase 4 dataset: 32 PSD rotors on varied fibrotic substrates. Runs inside the container
# (launched detached by gen_dataset_launch.sh). Each config is ~90 s; resumable
# (gen_one.sh skips configs with a status=ok manifest).
set -euo pipefail
REPO=/repo
#      name  fib_cx fib_cy fib_r fib_seed fib_frac  rot_x rot_y  chir
CFG=(
 "d00  19 19 15 10 0.30  21 21 1"
 "d01  19 19 13 11 0.35  21 17 -1"
 "d02  19 23 11 12 0.25  17 25 -1"
 "d03  19 23 15 13 0.25  19 25 -1"
 "d04  19 27 11 14 0.35  17 27 -1"
 "d05  19 27 15 15 0.25  19 25 1"
 "d06  19 31 14 16 0.25  21 31 -1"
 "d07  19 31 13 17 0.30  19 31 -1"
 "d08  23 19 13 18 0.35  25 21 -1"
 "d09  23 19 14 19 0.30  23 21 -1"
 "d10  23 23 13 20 0.25  25 21 1"
 "d11  23 23 15 21 0.30  21 21 -1"
 "d12  23 27 13 22 0.25  21 27 -1"
 "d13  23 27 15 23 0.30  25 29 -1"
 "d14  23 31 15 24 0.30  23 31 -1"
 "d15  23 31 12 25 0.30  23 29 1"
 "d16  27 19 15 26 0.25  25 17 -1"
 "d17  27 19 15 27 0.35  29 17 -1"
 "d18  27 23 14 28 0.30  27 21 -1"
 "d19  27 23 14 29 0.35  27 21 -1"
 "d20  27 27 13 30 0.25  29 29 1"
 "d21  27 27 11 31 0.30  29 29 -1"
 "d22  27 31 14 32 0.30  25 33 -1"
 "d23  27 31 13 33 0.25  25 31 -1"
 "d24  31 19 14 34 0.30  31 21 -1"
 "d25  31 19 14 35 0.30  31 19 1"
 "d26  31 23 11 36 0.25  31 23 -1"
 "d27  31 23 13 37 0.25  29 21 -1"
 "d28  31 27 12 38 0.35  31 27 -1"
 "d29  31 27 13 39 0.35  31 27 -1"
 "d30  31 31 13 40 0.30  31 31 1"
 "d31  31 31 13 41 0.35  33 29 -1"
)
for c in "${CFG[@]}"; do
  bash "$REPO/opencarp/phase4/gen_one.sh" $c || echo "  (config errored, continuing)"
done
echo "=== gen_dataset done  $(date -u +%FT%TZ) ==="
grep -h '^status' "$REPO"/opencarp/runs/phase4/d*/manifest.txt | sort | uniq -c
