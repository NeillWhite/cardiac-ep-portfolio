# Cardiac EP Simulation + ML Portfolio Project

A two-part project built for cardiac electrophysiology / ablation-focused ML roles:

1. **Real ECG signal classification** (PTB-XL) — arrhythmia detection baseline. *(Phase 1 — start here)*
2. **Biophysics-based EP simulation** (openCARP) — synthetic tissue/electrogram generation,
   eventually extended toward pulsed-field-ablation-relevant lesion modeling. *(Phase 2+)*

This README is written like a lightweight design document, mirroring how a medical device
company documents requirements → verification → results, since that's the environment
this project targets.

---

## 1. Problem Statement

Cardiac ablation systems need to (a) interpret complex cardiac signals (ECG, unipolar/bipolar
electrograms) and (b) run biophysics simulations fast enough to give near-real-time feedback
during a procedure. This project builds toward both: a signal-classification baseline on real
clinical ECGs, and a simulation pipeline that can later be optimized for speed with ML surrogates.

## 2. Design Requirements

| Requirement | Target | Rationale |
|---|---|---|
| Classification accuracy (macro F1) | ≥ 0.75 on PTB-XL superclass labels | Baseline competitive with published PTB-XL results |
| Inference latency | < 50ms per 10s 12-lead ECG | ✅ verified 0.339ms CPU / 0.151ms GPU, see `docs/MODEL_ARCHITECTURE.md` §2 — ~100x headroom, was never the bottleneck |
| Simulation reproducibility | Deterministic given fixed seed/mesh | Required for any verification claim |
| Code portability | Runs on CPU-only machine | Reviewers shouldn't need a GPU to check your work |

## 3. Repository Structure

```
cardiac-ep-portfolio/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── docs/
│   ├── IMPLEMENTATION_PLAN.md  # authoritative project plan, phases, review gates
│   ├── MODEL_ARCHITECTURE.md   # Phase 1 CNN architecture + design rationale
│   └── ep-primer.html          # cardiac EP field-guide reference doc
├── data/                   # not committed — see download instructions below
├── scripts/
│   ├── download_ptbxl.py   # pulls PTB-XL from PhysioNet via wfdb
│   ├── preprocess.py       # filtering, resampling, train/val/test split
│   ├── dataset.py          # PyTorch Dataset for ECG windows
│   ├── model.py            # 1D-CNN baseline classifier
│   ├── train_baseline.py   # training loop
│   ├── evaluate.py         # metrics + confusion matrix
│   └── export_viewer_data.py  # test-set predictions -> JSON for the interactive viewer (§7)
├── models/                 # saved checkpoints (not committed)
├── results/                # metrics.json, plots
└── opencarp/
    ├── README.md            # Docker setup + tutorial run instructions
    └── run_tutorial.sh
```

## 4. Getting Started (Phase 1 — ECG baseline)

This part needs to run on your own machine since it requires downloading PhysioNet data
(this sandbox can't reach physionet.org).

```bash
pip install -r requirements.txt
python scripts/download_ptbxl.py --output data/ptbxl
python scripts/preprocess.py --input data/ptbxl --output data/processed
python scripts/train_baseline.py --data data/processed --epochs 30
python scripts/evaluate.py --data data/processed --checkpoint models/baseline.pt
```

Results (accuracy, F1, confusion matrix) get written to `results/`.

**Note on labels:** PTB-XL ships with a diagnostic hierarchy (superclass → subclass →
individual statement). Start with the 5 superclasses (NORM, MI, STTC, CD, HYP) — that's the
standard baseline task in the literature and keeps the label space clean while you get the
pipeline working.

## 5. Phase 2 — openCARP simulation

See `opencarp/README.md`. This requires Docker and is meant to run locally, not in a CI
sandbox — EP simulations are compute-heavy.

## 6. Verification / Results

See `docs/MODEL_ARCHITECTURE.md` for the full architecture writeup, design rationale, and
model-specific limitations (as distinct from the project-level ones in §8 below).

### Phase 1 — ECG baseline (runs 2026-08-18, seed 42)

Trained on real PTB-XL (100Hz, PhysioNet release 1.0.3), 5-superclass task
(NORM/MI/STTC/CD/HYP), standard fold split (1-8 train / 9 val / 10 test).

**Current result — macro F1 (test): 0.6080** — below the plan's 0.70 target.
Early stopping at epoch 14 (patience=5), best checkpoint epoch 9
(val_macro_f1=0.6002).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORM | 0.81 | 0.91 | 0.86 | 912 |
| MI | 0.74 | 0.54 | 0.62 | 256 |
| STTC | 0.68 | 0.73 | 0.70 | 242 |
| CD | 0.80 | 0.68 | 0.74 | 184 |
| HYP | 0.19 | 0.09 | 0.12 | 56 |

*(This specific checkpoint is a 2026-08-19 re-run of the identical seed/config described
below — see the reproducibility note after the Decision paragraph for why the number moved
from 0.6028 to 0.6080 despite nothing being changed.)*

![Confusion matrix](results/confusion_matrix.png)
![Example ECG traces with predicted vs. true label](results/example_traces.png)

**What changed and what it did:** the first run (macro F1 0.6105) used raw
inverse-frequency class weights and early-stopped on val_loss at epoch 15
(patience=5). HYP (the smallest class, 3.3% of records) had precision 0.12 —
heavy false-positive rate from NORM. Hypothesis: the aggressive weighting was
overcorrecting, and val_loss (computed under those same weights) was cutting
training short before macro-F1 actually plateaued. Two fixes: switched to
sqrt(1/count) class weights, and switched both LR scheduling and early
stopping to track val macro-F1 directly instead of val_loss.

**Result: this did not help.** Macro F1 is flat-to-slightly-worse (0.6028 vs.
0.6105), and training ran 10 epochs longer with visibly noisy val_macro_f1
(oscillating 0.55-0.62 across epochs) rather than converging smoothly — a sign
of real variance, not just a stopped-too-early artifact. HYP precision/recall
moved (0.12→0.13 / 0.32→0.21) but stayed bad under both weighting schemes; CD
improved (F1 0.70→0.72, precision 0.66→0.90) at STTC's expense (precision
0.64→0.54, more NORM/CD points now misrouted through STTC instead of HYP).
Errors shifted around the confusion matrix rather than net-decreasing.

**Revised diagnosis:** this looks less like a hyperparameter artifact and more
like a genuine data-scarcity / model-capacity limit for HYP specifically — 535
total single-label examples (~427 in train after the fold split) is thin for a
class whose distinguishing ECG features (voltage/QRS-morphology criteria for
hypertrophy) are reportedly subtle even for a small 3-layer CNN with global
average pooling. NORM/MI/STTC/CD are consistently in the 0.66-0.81 F1 range
across both runs and in line with published PTB-XL baselines — the macro
average is being pulled down by one class, not a broad failure.
Not yet attempted: oversampling/augmenting HYP specifically, a
focal-loss-style objective, or accepting a 4-class macro F1 as a secondary
metric alongside the 5-class one — flagged here rather than iterated on
further without review.

**Decision:** accepted as the Phase 1 baseline as-is (2026-08-18) — two
independent fixes showed no net improvement, so further iteration here has
diminishing returns relative to Phase 2 (the openCARP simulation work, which
is the harder and higher-priority part of this project per the plan). Revisit
HYP-specific interventions (oversampling, focal loss) if time remains after
Phase 2-3 are solid.

**v2 architecture experiment (2026-08-19, attempted and reverted):** tried a
dilated final conv layer (wider receptive field, 340ms→600ms) plus
concatenated avg+max pooling (instead of avg alone), aimed at the same
"global average pooling washes out localized abnormalities" concern. Result:
raw accuracy improved (0.72→0.75) but macro F1 got worse (0.6028→0.5992) and
HYP F1 got worse too (0.16→0.11) — more model capacity gave the model more
ways to fit the well-represented classes without any more HYP data to anchor
that class's boundary. Reverted; full writeup and the before/after table are
in `docs/MODEL_ARCHITECTURE.md` §5, since `scripts/model.py` itself no longer
shows any trace of the attempt.

**Reproducibility note (2026-08-19):** regenerating the reverted checkpoint —
identical architecture, seed (42), and hyperparameters as the run that
produced the documented 0.6028 — actually produced macro F1 0.6080, a
different result. `train_baseline.py`'s `set_seed()` seeds numpy/torch in the
main process, but the `DataLoader` uses `num_workers=2`; the worker
subprocesses' own RNG state isn't covered by that single seed call, so batch
ordering/timing isn't actually fixed run-to-run despite "using a fixed seed."
Treat 0.6028 and 0.6080 as two draws from the same config's inherent
run-to-run variance (~0.005 macro F1), not two different models — the
architecture and training recipe are identical. Not fixed yet; a real fix
would mean `num_workers=0` (slower) or a properly seeded per-worker
`worker_init_fn` + `torch.Generator`, plus pinning
`torch.backends.cudnn.deterministic=True`. Flagged rather than silently
patched over.

### Phase 2 — openCARP simulation

*(Not yet started — see plan `docs/IMPLEMENTATION_PLAN.md` §4.)*

## 7. Interactive Viewer

A browser-based viewer over the Phase 1 test-set predictions — self-contained single HTML
file, no server, works fully offline (aside from optionally loading its Google Fonts).

**Primary copy:** `results/ecg_viewer.html` in this repo. Explicitly gitignored (unlike the
PNG plots elsewhere in `results/`) — at 5.6MB and fully regenerable from a checkpoint, it's
not worth tracking, so it only exists locally after being generated. If you're working over
SSH with no local browser, pull it to your own machine, e.g.:
```bash
scp <user>@<host>:/path/to/pfa/results/ecg_viewer.html ~/Downloads/
```
then open the downloaded file directly.

**How to use it:** pick a diagnostic superclass in the left rail (NORM/MI/STTC/CD/HYP), then
an individual test-set sample from the list below it — each row is marked ✓ or ✗ against the
true label. Selecting a sample shows the model's full confidence breakdown across all five
classes (the true class gets an outlined bar when the model got it wrong, so you can see how
close the model came, not just that it missed), a large primary-lead trace with a hover
crosshair for exact time/amplitude, and the full 12-lead grid in standard clinical layout —
click any of the 12 small panels to make it the primary trace.

**Reusable across models by construction:** the viewer reads a fixed JSON schema (id / true
label / predicted label / per-class confidence / per-lead waveform) rather than anything
specific to `ECGConvNet`. To showcase a future, more complex model, regenerate the data:

```bash
python scripts/export_viewer_data.py --checkpoint models/<new_checkpoint>.pt \
    --model-name "<model description>" --output results/viewer_data.json
```

then hand the resulting `results/viewer_data.json` to Claude Code to splice into the viewer
template and rewrite `results/ecg_viewer.html`. The header's model name and export date
update automatically from the JSON — no other viewer changes needed.

## 8. Limitations & Future Work

- **Multi-label records dropped, not multi-hot encoded:** PTB-XL records with
  more than one diagnostic superclass are excluded from training entirely for a
  clean single-label baseline. This drops 25.5% of all records (21,799 total →
  16,244 single-label). Meaningful chunk of the data, worth revisiting if the
  single-label ceiling turns out to be the real bottleneck rather than class
  weighting.
- Baseline uses a simple 1D-CNN; a resnet-style or transformer architecture would likely
  improve macro F1 but adds complexity not needed for a first pass.
- No handling yet of lead-missing or noisy real-world recordings (relevant to Job 2 — see
  the separate PulseDB project).
- openCARP integration currently generates synthetic electrograms only; extending the
  electrical model toward pulsed-field-ablation-style lesion prediction (electric field
  magnitude as a lesion-likelihood proxy) is the natural next step, given the target
  company's PFA-based platform.
- No latency optimization / ML surrogate yet — that's Phase 3 of the overall plan.

## 9. Why This Project

Built to demonstrate the specific overlap of skills requested in cardiac EP / ablation ML
roles: interpreting real cardiac signals (ECG, electrograms), hands-on use of a cardiac
simulation tool (openCARP), and an understanding of the real-time compute constraints that
matter in an actual mapping/ablation system.
