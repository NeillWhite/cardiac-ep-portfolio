#!/bin/bash
# Phase 4 Gate 4d: for a handful of configs, re-run PSD (late state + cmd dump), then
# branch the established rotor 3 ways (none / lesion@predicted / lesion@control).
set -euo pipefail
REPO=/repo
# name  fib_cx fib_cy fib_r fib_seed fib_frac  rot_x rot_y chir   pred_x pred_y   ctrl_x ctrl_y
CFG=(
 "c00  19 19 15 10 0.30  21 21  1   17.7 16.8   30.7 16.8"
 "c07  19 31 13 17 0.30  19 31 -1   18.9 29.3   31.9 29.3"
 "c18  27 23 14 28 0.30  27 21 -1   23.9 21.8   37.0 21.8"
 "c10  23 23 13 20 0.25  25 21  1   20.6 23.8   33.5 23.8"
 "c26  31 23 11 36 0.25  31 23 -1   32.1 23.0   19.1 23.0"
 "c30  31 31 13 40 0.30  31 31  1   33.6 34.2   20.6 34.2"
)
for row in "${CFG[@]}"; do
  set -- $row
  name=$1; shift
  bash "$REPO/opencarp/phase4/gen_one.sh" "$name" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" \
    || { echo "$name gen failed"; continue; }
  grep -q "^status=ok" "$REPO/opencarp/runs/phase4/$name/manifest.txt" || { echo "$name no sustained rotor"; continue; }
  python3 "$REPO/opencarp/phase4/closed_loop2.py" "$REPO/opencarp/runs/phase4/$name" "$9" "${10}" "${11}" "${12}"
done
echo "=== closed_loop2 done $(date -u +%FT%TZ) ==="
