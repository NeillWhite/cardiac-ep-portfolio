#!/bin/bash
# Launch G3b (rotors B and C: induce + radius probe + position sweep) as a DETACHED
# container that survives a terminal/Cursor close. Safe to re-run -- every step resumes.
#   opencarp/phase3/g3b.sh                 # start / resume
#   docker logs -f pfa_phase3_g3b          # follow
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker rm -f pfa_phase3_g3b 2>/dev/null || true
docker run -d --name pfa_phase3_g3b -v "$REPO:/repo" -w /repo/opencarp \
  docker.opencarp.org/opencarp/opencarp:latest \
  bash /repo/opencarp/phase3/g3b_inner.sh
echo "started pfa_phase3_g3b -- follow with: docker logs -f pfa_phase3_g3b"
echo "results: $REPO/opencarp/runs/phase3/{B,C}/sweep_grid_r6000.csv"
