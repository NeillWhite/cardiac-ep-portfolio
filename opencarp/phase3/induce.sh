#!/bin/bash
# Phase 3 step 1: re-induce one rotor, KEEPING the full working tree (mesh, parameters,
# state checkpoints) so lesion counterfactuals can branch off it. Same induction as
# regen_5s, but ex/ is preserved and the exact free-run openCARP command is dumped to
# freerun_cmd.txt for lesion_sweep.py to reuse.
#   site A: run.py default   |   B: mirror side   |   C: below the patch
set -euo pipefail
SITE="${1:?usage: induce.sh <A|B|C>}"
REPO=/repo
OUT="$REPO/opencarp/runs/phase3/$SITE"
EX="$OUT/ex"
PREPACE_SRC="$REPO/opencarp/runs/reentry_induction/prepace_block_50000.0um_resolution_400.0um_cv_0.3_with_4_beats_at_500.0_bcl_lump_1"
RPB=/usr/local/lib/python3.10/dist-packages/carputils/model/protocols/RP_B_algorithm.py

echo "=== phase3 induce: site $SITE  $(date -u +%FT%TZ) ==="
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r /openCARP/examples/02_EP_tissue/21_reentry_induction "$EX"
[ -d "$PREPACE_SRC" ] && cp -r "$PREPACE_SRC" "$EX/" && echo "reused cached prepace state"

case "$SITE" in
  A) : ;;
  B) sed -i 's|args.slabsize/7\.|args.slabsize*6/7.|g' "$EX/run.py" ;;
  C) sed -i 's|args.slabsize/7\., args.slabsize/2\.|args.slabsize/2., args.slabsize/7.|g' "$EX/run.py" ;;
esac

# extend the post-induction free-run + save checkpoints + dump the exact carp command
python3 - "$RPB" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = ("                    # Run simulation\n"
       "                    job.carp(cmd)\n"
       "\n"
       "                    extract_last_ms_from_igb(job, nodes_to_check_str, simid, 100)")
new = ("                    cmd += ['-tend', t_last_beat + 4000,\n"
       "                            '-num_tsav', 4,\n"
       "                            '-tsav[0]', t_last_beat + 1000,\n"
       "                            '-tsav[1]', t_last_beat + 1500,\n"
       "                            '-tsav[2]', t_last_beat + 2000,\n"
       "                            '-tsav[3]', t_last_beat + 3000]\n"
       "                    import json as _json\n"
       "                    open(simid + '/freerun_cmd.txt', 'w').write('\\n'.join(str(x) for x in cmd))\n"
       "                    # Run simulation\n"
       "                    job.carp(cmd)\n"
       "\n"
       "                    extract_last_ms_from_igb(job, nodes_to_check_str, simid, 100)")
if "freerun_cmd.txt" in s:
    print("RP_B_algorithm already patched -- skipping")
else:
    # the free-run block sets '-tend', t_last_beat + 600 already; strip that so ours wins
    s2 = s.replace("                            '-tend', t_last_beat + 600,\n", "", 1)
    assert old in s2, "patch anchor not found"
    open(p, "w").write(s2.replace(old, new, 1))
    print("patched RP_B_algorithm: free-run 4000ms, 4 checkpoints, cmd dump")
PY

cd "$EX"
./run.py --np 8 --protocol RP_B --start_bcl 200 --end_bcl 100 --step 10 \
    --max_n_beats_RP 1 --overwrite-behaviour overwrite

JOB=$(find "$EX" -maxdepth 1 -type d -name '*_RP_B*' | sort | tail -1)
NODE=$(awk 'NR==1{print $1}' "$JOB/reentries.txt")
SIM=$(dirname "$(find "$JOB" -name vm.igb | head -1)")
echo "job=$JOB  node=$NODE  sim=$SIM"
cat "$JOB/reentries.txt"
ls "$SIM"/state.*.roe
python3 "$REPO/opencarp/phase_singularity.py" "$SIM" "$JOB/block_i" --out "$SIM" --radius-um 3000
python3 "$REPO/opencarp/export_for_notebook.py" "$SIM" "$JOB/block_i" --out "$OUT/rotor${SITE}_phase3.npz"
# manifest for lesion_sweep.py
{
  echo "site=$SITE"
  echo "job=$JOB"
  echo "sim=$SIM"
  echo "mesh=$JOB/block_i"
  echo "reentry_node=$NODE"
} > "$OUT/manifest.txt"
cat "$OUT/manifest.txt"
echo "=== phase3 induce $SITE done  $(date -u +%FT%TZ) ==="
