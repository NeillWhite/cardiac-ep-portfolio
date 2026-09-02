#!/bin/bash
# Phase 3 rotor-A position sweep, r=6 mm, as a DETACHED container that survives a
# terminal/Cursor close. Safe to re-run: lesion_sweep.py resumes from the CSV, skipping
# candidates already scored. A machine reboot kills the container -- just run this again.
#   opencarp/phase3/run_grid_A.sh          # start / resume
#   docker logs -f pfa_phase3_grid_A       # follow
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker rm -f pfa_phase3_grid_A 2>/dev/null || true
docker run -d --name pfa_phase3_grid_A -v "$REPO:/repo" -w /repo/opencarp \
  docker.opencarp.org/opencarp/opencarp:latest \
  python3 /repo/opencarp/phase3/lesion_sweep.py A \
    --lesion-radius-um 6000 --run-ms 1500 \
    --grid-mm 10 35 12 37 --grid-step-mm 5 --tag grid_r6000_v2 --keep-vm
echo "started pfa_phase3_grid_A -- follow with: docker logs -f pfa_phase3_grid_A"
echo "results: $REPO/opencarp/runs/phase3/A/sweep_grid_r6000_v2.csv"
