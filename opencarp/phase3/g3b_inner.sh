#!/bin/bash
# G3b: induce rotors B and C, calibrate lesion size, and run the position sweep for each.
# Runs INSIDE the openCARP container (launched detached by g3b.sh). Every step is
# resumable -- re-running skips inductions whose manifest exists and sweep candidates
# already in the CSV.
set -euo pipefail
REPO=/repo

for S in B C; do
  echo "######################## rotor $S ########################  $(date -u +%FT%TZ)"

  if [ -f "$REPO/opencarp/runs/phase3/$S/manifest.txt" ]; then
    echo "induction for $S already done -- skipping"
  else
    bash "$REPO/opencarp/phase3/induce.sh" "$S"
  fi

  echo "--- rotor $S: lesion-size probe (at meander centroid) ---"
  python3 "$REPO/opencarp/phase3/lesion_sweep.py" "$S" \
    --radius-probe 4000 6000 8000 10000 12000 --run-ms 2000 --tag radius_probe

  echo "--- rotor $S: position sweep (auto grid, r=6mm, 2.5s post-lesion) ---"
  python3 "$REPO/opencarp/phase3/lesion_sweep.py" "$S" \
    --lesion-radius-um 6000 --run-ms 2500 --grid-step-mm 5 --tag grid_r6000 --keep-vm
done

echo "######################## G3b done  $(date -u +%FT%TZ) ########################"
