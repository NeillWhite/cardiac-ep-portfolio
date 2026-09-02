#!/bin/bash
# Phase 4 Gate 4d (baked-in variant -- avoids the gi_scale_vec-on-restore issue):
# for a few configs, re-seed the rotor with a non-conductive lesion of radius {5,8,12} mm
# baked into the substrate at the TRUE functional core, and separately at a control point.
# gen_one.sh's status tells us whether a sustained rotor forms.
set -euo pipefail
REPO=/repo
# name       fib_cx fib_cy fib_r fib_seed fib_frac  rot_x rot_y chir   fc_x fc_y   ctrl_x ctrl_y
BASE=(
 "b00  19 19 15 210 0.30  25 25  1   18.3 17.4   31 33"
 "b18  27 23 14 228 0.30  33 30 -1   24.1 23.3   14 33"
 "b26  31 23 11 236 0.25  25 30 -1   31.6 22.8   15 30"
)
for row in "${BASE[@]}"; do
  set -- $row; nm=$1; shift
  fcx=$1 fcy=$2 fr=$3 fs=$4 ff=$5 rx=$6 ry=$7 ch=$8 tx=$9 ty=${10} cx=${11} cy=${12}
  for r in 5 8 12; do
    bash "$REPO/opencarp/phase4/gen_one.sh" "${nm}_fc${r}"  $fcx $fcy $fr $fs $ff $rx $ry $ch  $tx $ty $r || true
    bash "$REPO/opencarp/phase4/gen_one.sh" "${nm}_ct${r}"  $fcx $fcy $fr $fs $ff $rx $ry $ch  $cx $cy $r || true
  done
done
echo "=== closed_loop3 done $(date -u +%FT%TZ) ==="
for d in "$REPO"/opencarp/runs/phase4/b??_??*; do
  [ -f "$d/manifest.txt" ] && echo "$(basename $d): $(grep -h '^status' $d/manifest.txt)"
done
