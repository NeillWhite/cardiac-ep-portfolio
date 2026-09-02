#!/bin/bash
# Phase 4 dataset: place ONE rotor (PSD / Eikonal init -- no prepace, no bench) on a
# VARIED fibrotic substrate, free-run 3 s, and export what the surrogate needs
# (Vm, functional core, phase-singularity path, electrode features).
# Runs INSIDE the openCARP container.
#
#   gen_one.sh <name> <fib_cx> <fib_cy> <fib_r> <fib_seed> <fib_frac> <rot_x> <rot_y> <chirality>
#   (all mm; chirality -1=CCW 1=CW)
set -euo pipefail
NAME="${1:?}"; FCX="${2:?}"; FCY="${3:?}"; FR="${4:?}"; FSEED="${5:?}"; FFRAC="${6:?}"
RX="${7:?}"; RY="${8:?}"; CHIR="${9:?}"
LX="${10:-}"; LY="${11:-}"; LR="${12:-5}"   # optional non-conductive lesion (mm) baked in from t=0
REPO=/repo
OUT="$REPO/opencarp/runs/phase4/$NAME"
EX="$OUT/ex"
PSD=/usr/local/lib/python3.10/dist-packages/carputils/model/protocols/PSD_algorithm.py

echo "=== phase4 gen $NAME | fib=($FCX,$FCY) r=$FR seed=$FSEED frac=$FFRAC | rotor=($RX,$RY) chir=$CHIR | $(date -u +%FT%TZ)"
if [ -f "$OUT/manifest.txt" ] && grep -q "^status=ok" "$OUT/manifest.txt"; then
  echo "already done, skipping"; exit 0
fi
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r /openCARP/examples/02_EP_tissue/21_reentry_induction "$EX"
rm -rf "$EX"/fibrosis_block_* "$EX"/prepace_block_*

# patch run.py: fibrosis block + PSD rotor placement (+ optional baked-in lesion)
python3 - "$EX/run.py" "$FCX" "$FCY" "$FR" "$FSEED" "$FFRAC" "$RX" "$RY" "$LX" "$LY" "$LR" <<'PY'
import sys
p, fcx, fcy, fr, fseed, ffrac, rx, ry, lx, ly, lr = sys.argv[1:]
s = open(p).read()
s = s.replace("        random.seed(1)\n", f"        random.seed({int(fseed)})\n", 1)
s = s.replace("        centre = np.asarray([args.slabsize/2, args.slabsize/2, 0.])\n",
              f"        centre = np.asarray([{float(fcx)*1000}, {float(fcy)*1000}, 0.])\n", 1)
s = s.replace("tree.query_ball_point(centre, args.slabsize/3.5)",
              f"tree.query_ball_point(centre, {float(fr)*1000})", 1)
s = s.replace("int(len(elements_in_fibrotic_reg)*0.3)",
              f"int(len(elements_in_fibrotic_reg)*{float(ffrac)})", 1)
if lx:  # add a solid non-conductive disc (the "ablation lesion") to the dead-element set
    s = s.replace(
        "        elements_in_fibrotic_reg = set(elements_in_fibrotic_reg) - elems_not_conductive\n",
        "        elements_in_fibrotic_reg = set(elements_in_fibrotic_reg) - elems_not_conductive\n"
        f"        elems_not_conductive |= set(tree.query_ball_point([{float(lx)*1000}, {float(ly)*1000}, 0.], {float(lr)*1000}))\n"
        "        elements_in_fibrotic_reg = set(elements_in_fibrotic_reg) - elems_not_conductive\n", 1)
s = s.replace("        centre = np.asarray([args.slabsize/2., args.slabsize/2., 0.])\n",
              f"        centre = np.asarray([{float(rx)*1000}, {float(ry)*1000}, 0.])\n", 1)
open(p, "w").write(s)
print(f"  patched run.py" + (f" (+lesion at ({lx},{ly}) r={lr})" if lx else ""))
PY

# patch PSD_algorithm.py: 500ms -> 3000ms free-run, save a late state, dump the carp cmd
python3 - "$PSD" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
if "freerun_cmd_psd.txt" in s:
    print("  PSD_algorithm already patched"); raise SystemExit
assert "'-tend', 500.1," in s
s = s.replace("            '-tend', 500.1,\n            '-simID', job.ID,\n",
              "            '-tend', 3000.1,\n            '-simID', job.ID,\n"
              "            '-num_tsav', 2, '-tsav[1]', 2500,\n", 1)
s = s.replace("    # Run simulations\n    job.carp(cmd)\n",
              "    open(job.ID + '/freerun_cmd_psd.txt','w').write('\\n'.join(str(x) for x in cmd))\n"
              "    # Run simulations\n    job.carp(cmd)\n", 1)
open(p, "w").write(s)
print("  patched PSD_algorithm: 3000ms, state@2500, cmd dump")
PY

cd "$EX"
set +e
./run.py --np 8 --protocol PSD --PSD_bcl 160 --chirality "$CHIR" --radius 8000 \
    --overwrite-behaviour overwrite
rc=$?
set -e

JOB=$(find "$EX" -maxdepth 1 -type d -name '2*' ! -name 'meshes' | sort | tail -1)
VM="$JOB/vm.igb"
if [ $rc -ne 0 ] || [ ! -f "$VM" ]; then
  echo "  PSD FAILED (rc=$rc) for $NAME"
  { echo "status=psd_failed"; echo "config=$FCX $FCY $FR $FSEED $FFRAC $RX $RY $CHIR"; } > "$OUT/manifest.txt"
  exit 0
fi

# sustained-rotor check: any tissue still swinging >40 mV in the last 250 ms?
ALIVE=$(python3 - "$VM" "$JOB/block_i" <<'PY'
import sys, numpy as np
from carputils.carpio import igb
d,_,_ = igb.read(sys.argv[1]); pts = np.loadtxt(sys.argv[2]+".pts", skiprows=1)
vm = d if d.shape[0]==pts.shape[0] else d.T
tail = vm[:, -250:]
print(1 if float(np.max(tail.max(1)-tail.min(1))) > 40 else 0)
PY
)
if [ "$ALIVE" != "1" ]; then
  echo "  rotor died / not sustained for $NAME -- discarding"
  { echo "status=not_sustained"; echo "config=$FCX $FCY $FR $FSEED $FFRAC $RX $RY $CHIR"; } > "$OUT/manifest.txt"
  exit 0
fi

python3 "$REPO/opencarp/phase_singularity.py" "$JOB" "$JOB/block_i" --out "$JOB" --radius-um 3000
python3 "$REPO/opencarp/extract_electrode_features.py" "$JOB" "$JOB/block_i" \
    --out "$OUT/electrode_features.csv" --stride 5
python3 - "$JOB" "$JOB/block_i" "$OUT" <<'PY'
import sys, numpy as np
from scipy.ndimage import uniform_filter
from carputils.carpio import igb
job, mesh, out = sys.argv[1:]
d,_,_ = igb.read(f"{job}/vm.igb")
pts = np.loadtxt(mesh+".pts", skiprows=1)/1000.0
vm = d if d.shape[0]==pts.shape[0] else d.T
nx=len(np.unique(pts[:,0])); ny=len(np.unique(pts[:,1]))
X=pts[:,0].reshape(ny,nx); Y=pts[:,1].reshape(ny,nx)
far=int(np.argmax((vm.max(1)-vm.min(1))*(np.hypot(pts[:,0]-25,pts[:,1]-25)>18)))
tr=vm[far]>-20; a=np.where((~tr[:-1])&(tr[1:]))[0]
a=a[np.diff(np.concatenate([[-999],a]))>60]
cl=int(np.median(np.diff(a))) if len(a)>2 else 160
t0=vm.shape[1]//2; seg=vm[:,t0:t0+cl]
amp=(seg.max(1)-seg.min(1)).reshape(ny,nx); am=amp.copy(); am[am<12]=np.nan
m=~np.isnan(am)
sm=uniform_filter(np.nan_to_num(am),6)/np.maximum(uniform_filter(m.astype(float),6),1e-6)
w = sm < np.nanpercentile(sm,12)
wx,wy=float(X[w].mean()), float(Y[w].mean())
np.savez(f"{out}/ground_truth.npz", functional_core=np.array([wx,wy]),
         activation_strength=sm, X=X, Y=Y, cycle_ms=cl)
print(f"  functional core = ({wx:.1f},{wy:.1f}) mm  (cycle {cl} ms)")
PY

{
  echo "status=ok"
  echo "config=$FCX $FCY $FR $FSEED $FFRAC $RX $RY $CHIR"
  echo "job=$JOB"; echo "mesh=$JOB/block_i"
} > "$OUT/manifest.txt"
find "$EX" -name 'vm.igb' -size +40M -exec sh -c 'cp "$1" "$(dirname "$1")/../../vm_field.igb" 2>/dev/null; :' _ {} \; 2>/dev/null || true
echo "=== $NAME done  $(date -u +%FT%TZ)"
