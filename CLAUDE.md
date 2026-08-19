# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A portfolio project for a Staff Scientist (ML & Simulations) role at a cardiac
electrophysiology / pulsed-field-ablation (PFA) company. It has two mostly-independent
tracks that eventually connect:

1. **ECG classification** (`scripts/`) — PyTorch baseline on real PTB-XL data.
2. **openCARP simulation** (`opencarp/`) — biophysics simulation of cardiac tissue,
   used to generate ground-truth "ablation target" labels (rotor/phase-singularity
   tracking) and electrogram features for a second ML model.

**`docs/IMPLEMENTATION_PLAN.md` is the authoritative project plan** — phases,
review gates, the ground-truth definition for Phase 2, and known risks to flag rather than
silently resolve. Read it before starting work in any phase. It supersedes any stale detail
in `README.md` (e.g. the macro F1 target differs slightly between the two — the plan's
0.70 is the current bar).

The project is built in explicit phases with **review gates between them** — do not jump
ahead to the next phase's code without the human confirming the previous gate's results
(real metrics/plots, not "it ran"). See `docs/IMPLEMENTATION_PLAN.md` §8 for the gate summary.

`docs/MODEL_ARCHITECTURE.md` documents the Phase 1 CNN's architecture and design rationale
in detail — read it before changing `scripts/model.py`.

## Commands

Phase 1 (ECG), run from repo root, `data/`, `models/`, `results/` are gitignored:

```bash
pip install -r requirements.txt
python scripts/download_ptbxl.py --output data/ptbxl        # pulls PTB-XL from PhysioNet via wfdb, ~1.7GB at 100Hz
python scripts/preprocess.py --input data/ptbxl --output data/processed
python scripts/train_baseline.py --data data/processed --epochs 30
python scripts/evaluate.py --data data/processed --checkpoint models/baseline.pt
```

`train_baseline.py` and `evaluate.py` import `dataset` and `model` as bare local modules
(no package structure), so they must be run with `scripts/` as the working directory *or*
rely on the scripts' own path resolution — check each script before assuming which; e.g.
`train_baseline.py`/`evaluate.py` resolve `models/`/`results/` relative to repo root
regardless of cwd, but the `from dataset import ...` / `from model import ...` imports
require `scripts/` to be on `sys.path` (i.e. run `python scripts/train_baseline.py` from
repo root, not `cd scripts && python train_baseline.py` unless scripts/ is added to path —
verify actual behavior before changing this).

No test suite, linter, or CI config exists yet.

Phase 2 (openCARP) runs in Docker locally, not CI — see `opencarp/README.md`. The exact
image name / tutorial paths in that README are **unverified guesses from a sandboxed
build environment** and must be checked against live docs
(https://opencarp.org/download/installation, https://opencarp.org/documentation) before
relying on them.

## Architecture notes

**Phase 1 data flow:** `download_ptbxl.py` → raw PhysioNet records in `data/ptbxl/`
(`ptbxl_database.csv` for labels, `scp_statements.csv` for diagnostic-code→superclass
mapping) → `preprocess.py` maps SCP diagnostic codes to the 5 PTB-XL superclasses (NORM,
MI, STTC, CD, HYP), **drops any record with more than one superclass label** (multi-label
records are excluded, not multi-hot encoded — report the dropped fraction per the plan),
and writes fixed `X_{split}.npy` / `y_{split}.npy` arrays using PTB-XL's own recommended
fold split (folds 1-8 train, 9 val, 10 test) → `train_baseline.py` trains `ECGConvNet`
(`model.py`, a small 1D-CNN over 12-lead signals, deliberately simple/explainable rather
than SOTA) with class-weighted loss (superclasses are imbalanced, NORM dominates) →
`evaluate.py` writes `results/metrics.json` (macro F1, full classification report,
confusion matrix) and prints a plaintext confusion matrix.

Per the implementation plan, still missing from this flow as of the current scaffold: fixed
random seeding, early stopping, a `--seed` CLI arg, and a plotted (not just printed)
confusion matrix — these are Phase 1 TODOs, not yet implemented.

**Phase 2 ground truth is the hard part of this project** — nothing labels "optimal
ablation site" the way PTB-XL labels arrhythmia class. The plan's approach: induce a
reentrant rotor in simulated tissue (S1-S2 stimulation near a low-conductivity patch) →
Hilbert-transform the transmembrane voltage time series at each mesh point → detect phase
singularities (rotor core) → label nearby mesh points as ablation targets → train a
classifier that predicts rotor-adjacency from *local electrogram features only*
(bipolar amplitude, fractionation, local activation time, dominant frequency), so
inference doesn't require the full phase-mapping computation. This is new code on top of
openCARP, not something the simulator gives you natively. See `docs/IMPLEMENTATION_PLAN.md`
§4 for the full reasoning and why a simple interpretable model (GBT/small MLP) is
preferred over deep learning on raw waveforms for this phase.

**Phase 3** (real-time optimization) and **Phase 4** (PFA electrostatics angle, optional)
only start once the prior phase's pipeline is real and working — see the plan for details;
no code for either exists yet.

## Documentation convention

Every simplifying assumption (single-label ECG filtering, 2D vs 3D mesh, electrostatics
approximation standing in for full electroporation physics, etc.) is meant to get one
explicit sentence in the README's Limitations section as it's introduced — this is a
stated project convention (`docs/IMPLEMENTATION_PLAN.md` §7), not just a suggestion. Update the
README's "Verification / Results" section with real numbers as each phase completes rather
than batching it at the end.
