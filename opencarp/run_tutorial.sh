#!/bin/bash
# Run the reentry induction example inside the openCARP Docker container.
# Verified path against live docs 2026-08-18 -- see README.md in this folder.
# Run this INSIDE the container (docker run -it -v $(pwd):/shared docker.opencarp.org/opencarp/opencarp:latest),
# not on the host.
set -e

cd /openCARP/examples/02_EP_tissue/21_reentry_induction
./run.py --np 2 --protocol RP_E --start_bcl 200 --end_bcl 130 --max_n_beats_RP 1 --visualize
