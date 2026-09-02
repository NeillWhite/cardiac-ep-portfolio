#!/bin/bash
# Re-run the lesion-size probe AT THE FUNCTIONAL-CORE CENTROID (not the phase-singularity
# pivot, which is what the original probes used) for rotors A, B, C.
# Detached + resumable via g3b-style launcher fc_probe_launch.sh.
set -euo pipefail
REPO=/repo
for S in A B C; do
  XY=$(python3 "$REPO/opencarp/phase3/functional_core.py" "$S")
  echo "######## rotor $S -- functional-core centroid = $XY mm  $(date -u +%FT%TZ)"
  python3 "$REPO/opencarp/phase3/lesion_sweep.py" "$S" \
    --at-xy $XY --radius-probe 3000 4000 5000 6000 8000 10000 \
    --run-ms 2000 --tag radius_probe_fc
done
echo "######## fc_probe done  $(date -u +%FT%TZ)"
