#!/bin/bash
# Launch fc_probe.sh (lesion-size probe at the functional-core centroid, rotors A/B/C)
# as a detached, resumable container.
#   opencarp/phase3/fc_probe_launch.sh        # start / resume
#   docker logs -f pfa_phase3_fc_probe
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker rm -f pfa_phase3_fc_probe 2>/dev/null || true
docker run -d --name pfa_phase3_fc_probe -v "$REPO:/repo" -w /repo/opencarp \
  docker.opencarp.org/opencarp/opencarp:latest \
  bash /repo/opencarp/phase3/fc_probe.sh
echo "started pfa_phase3_fc_probe -- docker logs -f pfa_phase3_fc_probe"
echo "results: $REPO/opencarp/runs/phase3/{A,B,C}/sweep_radius_probe_fc.csv"
