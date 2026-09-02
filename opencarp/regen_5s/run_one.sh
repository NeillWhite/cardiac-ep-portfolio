#!/bin/bash
# Re-induce one reference rotor and record a ~5 s free-running window (vs. the ~400 ms of
# the original Phase 2 runs), plus intermediate state checkpoints for Phase 3 lesion
# branching. Runs INSIDE the openCARP container. Invoked by regen_5s/launch.sh.
#
#   site A: original stimulus site (run.py default)
#   site B: mirror-opposite side      (README §2a patch)
#   site C: below the fibrotic patch  (README §2a patch)
set -euo pipefail

SITE="${1:?usage: run_one.sh <A|B|C>}"
REPO=/repo
OUT="$REPO/opencarp/runs/regen_5s/$SITE"
EX="$OUT/ex"
PREPACE_SRC="$REPO/opencarp/runs/reentry_induction/prepace_block_50000.0um_resolution_400.0um_cv_0.3_with_4_beats_at_500.0_bcl_lump_1"
RPB=/usr/local/lib/python3.10/dist-packages/carputils/model/protocols/RP_B_algorithm.py

echo "=== regen 5s: site $SITE  $(date -u +%FT%TZ) ==="
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r /openCARP/examples/02_EP_tissue/21_reentry_induction "$EX"

# reuse the freshly-regenerated prepace steady state (substrate-only, independent of stim site)
if [ -d "$PREPACE_SRC" ]; then
  cp -r "$PREPACE_SRC" "$EX/"
  echo "reused cached prepace steady state"
else
  echo "WARNING: no cached prepace state, run.py will regenerate it (~minutes)"
fi

# stimulus-site patch (README §2a)
case "$SITE" in
  A) : ;;
  B) sed -i 's|args.slabsize/7\.|args.slabsize*6/7.|g' "$EX/run.py" ;;
  C) sed -i 's|args.slabsize/7\., args.slabsize/2\.|args.slabsize/2., args.slabsize/7.|g' "$EX/run.py" ;;
  *) echo "bad site $SITE"; exit 2 ;;
esac

# extend the post-induction free-run 600 ms -> 5000 ms, and save 5 state checkpoints
python3 - "$RPB" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = ("                            '-tend', t_last_beat + 600,\n"
       "                            '-meshname', meshname]")
new = ("                            '-tend', t_last_beat + 5000,\n"
       "                            '-num_tsav', 5,\n"
       "                            '-tsav[0]', t_last_beat + 1000,\n"
       "                            '-tsav[1]', t_last_beat + 2000,\n"
       "                            '-tsav[2]', t_last_beat + 3000,\n"
       "                            '-tsav[3]', t_last_beat + 4000,\n"
       "                            '-tsav[4]', t_last_beat + 4900,\n"
       "                            '-meshname', meshname]")
assert old in s, "RP_B_algorithm.py free-run patch target not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched RP_B_algorithm.py: free-run -> 5000 ms + 5 checkpoints")
PY

cd "$EX"
echo "--- run.py RP_B (site $SITE) ---"
./run.py --np 8 --protocol RP_B --start_bcl 200 --end_bcl 100 --step 10 \
    --max_n_beats_RP 1 --overwrite-behaviour overwrite

# locate the job dir, reentry node, and vm.igb
JOB=$(find "$EX" -maxdepth 1 -type d -name '*_RP_B*' | sort | tail -1)
echo "job dir: $JOB"
cat "$JOB/reentries.txt" 2>/dev/null || echo "(no reentries.txt — rotor may not have been confirmed)"
NODE=$(awk 'NR==1{print $1}' "$JOB/reentries.txt" 2>/dev/null || true)
VM=$(find "$JOB" -name vm.igb | head -1)
[ -z "$VM" ] && VM=$(find "$JOB" -name 'vm_saved.igb' -o -name 'vm_no_reentry*.igb' | head -1)
SIMDIR=$(dirname "$VM")
MESH="$JOB/block_i"
echo "vm: $VM   simdir: $SIMDIR   node: ${NODE:-unknown}"

igbhead "$VM" || true

# ground truth + portable export
echo "--- phase_singularity.py ---"
python3 "$REPO/opencarp/phase_singularity.py" "$SIMDIR" "$MESH" \
    --out "$SIMDIR" --radius-um 3000
echo "--- export_for_notebook.py ---"
python3 "$REPO/opencarp/export_for_notebook.py" "$SIMDIR" "$MESH" \
    --out "$OUT/rotor${SITE}_5s.npz"

# keep the checkpoints + a copy of the key artifacts where Phase 3 can find them
mkdir -p "$OUT/keep"
cp "$VM" "$OUT/keep/" 2>/dev/null || true
cp "$SIMDIR"/state.* "$OUT/keep/" 2>/dev/null || true
cp "$SIMDIR"/phase_singularity_results.npz "$OUT/keep/" 2>/dev/null || true
cp "$MESH".{pts,elem} "$OUT/keep/" 2>/dev/null || true
ls -la "$OUT/keep/"
echo "=== site $SITE done  $(date -u +%FT%TZ) ==="
