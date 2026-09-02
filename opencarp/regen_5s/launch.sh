#!/bin/bash
# Host-side launcher for the 5 s rotor regeneration. Run from the repo root:
#   opencarp/regen_5s/launch.sh A        # one site
#   opencarp/regen_5s/launch.sh A B C    # all three, sequential
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMG=docker.opencarp.org/opencarp/opencarp:latest

for SITE in "$@"; do
  echo ">>> launching site $SITE"
  docker run --rm \
    -v "$REPO":/repo \
    -w /repo/opencarp \
    "$IMG" \
    bash /repo/opencarp/regen_5s/run_one.sh "$SITE" \
    2>&1 | tee "$REPO/opencarp/runs/regen_5s_${SITE}.log"
done
echo ">>> all requested sites done"
