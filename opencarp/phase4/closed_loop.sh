#!/bin/bash
# Phase 4 Gate 4d -- closed-loop check: for 8 held-out configs, re-run the rotor with a
# 5 mm non-conductive lesion baked in at (a) the SURROGATE's predicted functional core
# and (b) a CONTROL point ~13 mm away. If the surrogate is right, _pred should NOT
# sustain a rotor and _ctrl should.  Runs inside the container; detached via launcher.
set -euo pipefail
REPO=/repo
CFG=(
 "d00_pred  19 19 15 110 0.30  21 21 1  17.7 16.8 5"
 "d00_ctrl  19 19 15 110 0.30  21 21 1  30.7 16.8 5"
 "d07_pred  19 31 13 117 0.30  19 31 -1  18.9 29.3 5"
 "d07_ctrl  19 31 13 117 0.30  19 31 -1  31.9 29.3 5"
 "d04_pred  19 27 11 114 0.35  17 27 -1  18.3 25.7 5"
 "d04_ctrl  19 27 11 114 0.35  17 27 -1  31.3 25.7 5"
 "d18_pred  27 23 14 128 0.30  27 21 -1  23.9 21.8 5"
 "d18_ctrl  27 23 14 128 0.30  27 21 -1  37.0 21.8 5"
 "d10_pred  23 23 13 120 0.25  25 21 1  20.6 23.8 5"
 "d10_ctrl  23 23 13 120 0.25  25 21 1  33.5 23.8 5"
 "d01_pred  19 19 13 111 0.35  21 17 -1  15.8 18.9 5"
 "d01_ctrl  19 19 13 111 0.35  21 17 -1  28.8 18.9 5"
 "d26_pred  31 23 11 136 0.25  31 23 -1  32.1 23.0 5"
 "d26_ctrl  31 23 11 136 0.25  31 23 -1  19.1 23.0 5"
 "d30_pred  31 31 13 140 0.30  31 31 1  33.6 34.2 5"
 "d30_ctrl  31 31 13 140 0.30  31 31 1  20.6 34.2 5"
)
for c in "${CFG[@]}"; do
  bash "$REPO/opencarp/phase4/gen_one.sh" $c || echo "  (errored, continuing)"
done
echo "=== closed_loop done  $(date -u +%FT%TZ) ==="
for d in "$REPO"/opencarp/runs/phase4/d*_pred "$REPO"/opencarp/runs/phase4/d*_ctrl; do
  [ -f "$d/manifest.txt" ] && echo "$(basename $d): $(grep -h '^status' $d/manifest.txt)"
done
