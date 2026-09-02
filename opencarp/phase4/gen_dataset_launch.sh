#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker rm -f pfa_phase4_gen 2>/dev/null || true
docker run -d --name pfa_phase4_gen -v "$REPO:/repo" -w /repo/opencarp \
  docker.opencarp.org/opencarp/opencarp:latest bash /repo/opencarp/phase4/gen_dataset.sh
echo "started pfa_phase4_gen -- docker logs -f pfa_phase4_gen"
