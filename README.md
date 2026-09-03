# Cardiac EP Simulation + ML

Machine learning and biophysics simulation applied to cardiac ablation targeting. The
project works through one question — *where do you ablate to stop a rotor?* — in four
phases:

1. **ECG signal classification** (PTB-XL) — arrhythmia-detection baseline on real clinical data.
2. **Rotor simulation + ground truth** (openCARP) — induce a reentrant rotor in simulated
   tissue, track its core, and train an electrogram classifier for core-adjacency.
3. **Simulate the ablation** — drop lesions at candidate sites and measure what terminates
   the rotor. Finding: the rotor's pivot (phase singularity) is the *wrong* target; the
   nearby **functional core** is the right one.
4. **Real-time surrogate** — a fast ML model that predicts the functional core from a
   600 ms electrogram window, ~1600× faster than the simulation.

The **[rendered docs site](https://neillwhite.github.io/cardiac-ep-portfolio/)** has the
two theory tutorials (bidomain/monodomain EP, the openCARP pipeline) in readable form. This
README is the design-document view: requirements → verification → results, in the style a
medical-device team documents its work.

---

## 1. Problem Statement

Cardiac ablation systems need to (a) interpret complex cardiac signals (ECG, unipolar/bipolar
electrograms) and (b) run biophysics simulations fast enough to give near-real-time feedback
during a procedure. This project builds toward both: a signal-classification baseline on real
clinical ECGs, and a simulation pipeline that can later be optimized for speed with ML surrogates.

## 2. Design Requirements

| Requirement | Target | Rationale |
|---|---|---|
| Classification accuracy (macro F1) | ≥ 0.70 on PTB-XL superclass labels | Baseline competitive with published PTB-XL results — ⚠️ current 0.61, gap is entirely the HYP class (see §6) |
| Inference latency | < 50ms per 10s 12-lead ECG | ✅ verified 0.339ms CPU / 0.151ms GPU, see `docs/MODEL_ARCHITECTURE.md` §2 — ~100x headroom, was never the bottleneck |
| Simulation reproducibility | Deterministic given fixed seed/mesh | Required for any verification claim |
| Code portability | Runs on CPU-only machine | Reviewers shouldn't need a GPU to check your work |

## 3. Repository Structure

```
cardiac-ep-portfolio/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── docs/                       # also served as a site via GitHub Pages
│   ├── index.html              # landing page for the Pages site
│   ├── IMPLEMENTATION_PLAN.md   # project plan: phases, review gates, ground-truth definition
│   ├── MODEL_ARCHITECTURE.md    # Phase 1 CNN architecture + design rationale
│   ├── PHASE2_METHODOLOGY.md    # Phase 2 narrative walkthrough (words + code)
│   ├── PHASE3_FINDINGS.md       # Phase 3 ablation-outcome results
│   ├── PHASE3_4_PLAN.md         # Phase 3 + 4 plan
│   ├── PHASE4_PLAN.md           # Phase 4 plan
│   ├── PHASE4_FINDINGS.md       # Phase 4 surrogate results
│   ├── BIDOMAIN_MONODOMAIN_TUTORIAL.html  # the EP theory the simulator solves
│   ├── OPENCARP_SIMULATION_PRIMER.html    # how the rotor pipeline works
│   └── figures/                 # figures embedded in the README and HTML docs
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
    ├── README.md                    # Docker setup + verified run instructions/gotchas
    ├── run_tutorial.sh
    ├── phase_singularity.py         # Hilbert phase -> PS detection -> ablation-target labels
    ├── plot_phase_singularity.py    # review-gate figures from phase_singularity.py output
    ├── render_vm_frames.py          # per-frame PNGs for the tracked-rotor video
    ├── render_full_story.py         # per-frame PNGs: stimulus through established rotor
    ├── render_pacing_train.py       # per-frame PNGs: all 6 pacing beats through established rotor
    ├── extract_electrode_features.py  # single-bipole electrode features (first attempt, ROC-AUC 0.61)
    ├── extract_cluster_features.py    # local-electrode-cluster features, swept over k=1..8
    ├── train_electrogram_classifier.py  # GBT classifier, single k (single-bipole or cluster CSV)
    ├── train_cluster_sweep.py         # trains the classifier at every k, produces the sweep plot
    ├── train_cross_rotor.py           # pairwise train-on-A/test-on-B cross-rotor check
    ├── train_leave_one_rotor_out.py   # leave-one-rotor-out CV across N independent rotors
    ├── export_for_notebook.py         # dumps one rotor's mesh+Vm+labels to a portable .npz
    ├── feature_exploration.ipynb      # hands-on EDA/feature-engineering/model-comparison notebook
    ├── regen_5s/                    # re-induce the 3 reference rotors with 5 s windows + checkpoints
    ├── phase3/                      # Phase 3: lesion sweep, functional-core detection, efficacy maps
    │   ├── lesion_sweep.py            # branch a rotor, drop a lesion at each grid site, score termination
    │   ├── functional_core.py         # weak-activation centroid = the ablation target
    │   ├── plot_efficacy.py           # efficacy maps + radius curves
    │   └── illustrate*.py             # the Phase 3 figures
    ├── phase4/                      # Phase 4: real-time surrogate
    │   ├── gen_one.sh / gen_dataset.sh  # PSD-protocol varied-substrate rotor generator (32 configs)
    │   ├── extract_features.py        # per-electrode features from a 600 ms window
    │   ├── train.py                   # HistGBT, leave-one-config-out, localisation error
    │   ├── benchmark.py               # surrogate latency vs. the biophysics pipeline
    │   └── closed_loop*.sh            # ablate the predicted site, check the effect
    ├── notebook_data/               # not committed — regenerable via export_for_notebook.py
    └── runs/                        # not committed — raw sim output (meshes, IGB, checkpoints)
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

### Phase 2 — openCARP simulation (ground-truth generation, run 2026-08-20)

**Substrate:** 5cm×5cm 2D tissue patch, 400µm resolution (126×126 regular grid, 15,876
nodes), Courtemanche + AF-remodeling ionic model, circular fibrotic patch (radius 1.42cm,
centered) reusing openCARP's `21_reentry_induction` example (`opencarp/README.md`).

**Induction:** the documented `RP_E`/`PEERP` invocations from `opencarp/README.md`
completed the setup and stimulus train but then crashed inside carputils' own
`extract_last_ms_from_igb` helper (`FileNotFoundError` for a `vm.igb` that was never
written) — traced to a **stale bundled prepace checkpoint**: the Docker image ships a
pre-tuned steady-state (`prepace_block_.../state.2000.0.roe`) generated in Dec 2020 by an
older openCARP build (checkpoint format v1), and restoring it into the current build's
solver crashes silently (process exits 0/1 with no diagnostic, right after "Restoring...").
Deleting that stale directory so `run.py` regenerates the prepace steady-state against the
*current* binary fixed it. `RP_B` (rapid pacing, BCL 200→100ms in 10ms steps, arrhythmia
check after every beat) then successfully **induced a sustained reentrant rotor** via an S2
stimulus at mesh node 7830 (coords (7.2, 24.8)mm) at a coupling interval of 150ms; the
rotor core itself settles at the fibrotic patch's edge, trajectory centroid ≈(28, 17)mm —
distinct from the stimulus site.

**Ground truth (new code, `opencarp/phase_singularity.py`):** Hilbert-transform each node's
mean-subtracted Vm → instantaneous phase; per-frame phase-singularity detection via
winding-number (phase-loop sum ≈±2π) over each mesh unit cell; greedy nearest-neighbour
linking into a single rotor-core trajectory; mesh nodes within 3mm of any trajectory point
labeled as ablation targets. Over the 400ms post-induction window: a phase singularity was
detected in **401/401 frames** (fully sustained, non-meandering-off rotor), tracked into one
continuous trajectory anchored at the patch border, labeling **823/15,876 nodes (5.2%)** as
ablation targets.

![Rotor trajectory and ablation-target labels](results/phase2_reentry_2026-08-20/rotor_trajectory.png)
![Vm snapshots with tracked rotor core](results/phase2_reentry_2026-08-20/vm_snapshots.png)

The snapshot grid shows a textbook spiral wave curling around the fibrotic patch, with the
detected phase-singularity marker (star) tracking the visual core of the spiral in every
panel — a useful sanity check independent of the numerical detection method.

**Mid-Phase-2 review gate:** cleared — the rotor trajectory and ablation-target labeling were
reviewed (including a full 6-beat pacing-train video showing the actual induction mechanism)
before building any classifier on top.

### Phase 2 — electrogram-feature classifier (rotor-adjacency prediction, run 2026-08-20)

**Virtual electrode sampling (`opencarp/extract_cluster_features.py`):** 576 sites on a 2mm
grid across the tissue (matching typical clinical mapping-catheter spacing; excludes
candidates within one neighbor-offset of the mesh boundary — see the bug note below).
Simplifying
assumption: a real catheter measures extracellular potential (`phi_e`), which this
monodomain-only simulation never computes (only `Vm`, the transmembrane difference — getting
a true `phi_e` needs either a full bidomain run or a separate lead-field forward-model
calculation on top of `Vm`, neither of which we did) — unipolar EGM is approximated as the
local `Vm(t)` directly instead (standard shortcut when that machinery is out of scope), and
bipolar EGM is the difference between two such signals ~2mm apart. That subtraction step
*is* exactly how real bipolar electrograms are derived from unipolar ones — but since our
"unipolar" inputs are themselves the `Vm` approximation, the resulting bipolar signal
inherits that same approximation; it doesn't escape it. Full explanation in
`docs/PHASE2_METHODOLOGY.md` §7.

**First attempt failed, honestly reported:** a single fixed-direction bipolar pair per site,
with the plan's four listed features (bipolar amplitude, fractionation, LAT, dominant
frequency — the last restricted to a 3-15Hz band, since an unrestricted FFT argmax on a
mostly non-periodic deflection picks up high-frequency edge artifacts, not real periodicity),
gave **ROC-AUC 0.61** — barely better than random. Per-class feature means were nearly
identical between ablation-target and background sites. Diagnosis: a phase singularity is a
*relational* concept (phase winds around a loop of neighboring points), so a single
electrode's own waveform is a weak proxy for it — using the full phase map directly would
just reconstruct the label's own definition (tautological 1.0 AUC, not a real prediction);
the actual task is deliberately harder, betting that a few *local* scalar features can stand
in for that expensive global computation.

**Fix — local electrode clusters + an electrode-count sweep:** instead of one fixed bipole,
each candidate site gets a small cluster of neighbor electrodes added one at a time (E, N, W,
S, then the 4 diagonals — up to 8, mimicking a small grid/basket catheter), with aggregate
features (min/mean/std of bipolar amplitude, mean/max fractionation, **spread and std of
local activation time across the cluster**, mean/std of dominant frequency) computed from the
first *k* neighbors. Training the same classifier at every k directly answers "how many local
electrodes does this need?":

![Electrode count sweep](results/phase2_reentry_2026-08-20/cluster_sweep.png)

ROC-AUC climbs from 0.67 (k=1) to 0.83-0.85 by k=3-5, up to 0.90 at k=8 — most of the
achievable signal comes from just a few local electrodes. Averaged over 10 spatial-split
repeats (checkerboard 5mm blocks, see below) for stability given the small positive count.

**Detailed result at k=4** (four cardinal neighbors — a physically realistic small
cross/grid catheter):

![Classifier results at k=4](results/phase2_reentry_2026-08-20/classifier_results.png)
![Spatial check: predictions vs. ground truth](results/phase2_reentry_2026-08-20/classifier_spatial_check.png)

**ROC-AUC 0.889, average precision 0.274** (base rate 5.7%) on a held-out **spatial**
checkerboard test set (5mm blocks — a naive random split would leak, since neighboring
electrodes are highly spatially correlated within one simulated rotor). At the default 0.5
probability threshold, hard predictions were spatially scattered despite good ranking
quality — a calibration artifact of the aggressive class-reweighting needed for ~5% positive
prevalence, not a real signal problem. Fixed by using a **prevalence-matched threshold**
(flag the top ~5% of test electrodes by predicted probability, the practically relevant
framing anyway — "check the top-N riskiest sites") instead of 0.5: precision 0.28, recall
0.31 for the ablation-target class. Feature importances (permutation-based, since
`HistGradientBoostingClassifier` has no built-in `feature_importances_`) rank `bipolar_amp_std`
and `dom_freq_std`/`dom_freq_mean` — cluster-*variability* in amplitude and frequency —
clearly highest; `lat_spread_ms`/`lat_std_ms` (the originally-hypothesized "timing spread near
a wavebreak" mechanism) turned out to carry almost no importance at all. See
`docs/PHASE2_METHODOLOGY.md` §9a for the diagnosis: LAT is computed as the single steepest
downstroke across the whole multi-beat window, and neighboring sites frequently pick a
*different* one of the ~2-3 available beats as "theirs," producing an apparent spread on the
order of a full cycle length almost everywhere (92% of a random 60-site background sample had
spread >100ms) — regardless of true core proximity. That's a genuine, diagnosed limitation of
the current LAT feature, not a project-wide failure — the amplitude/frequency cluster
features are still real signal.

**Two bugs found via hands-on exploration in `opencarp/feature_exploration.ipynb`,
fixed, and all above numbers re-verified against the fix:** (1) the LAT-timing issue above,
and (2) a genuine correctness bug in `extract_cluster_features.py`'s edge handling — for any
candidate site within one neighbor-offset of a mesh boundary (~15% of all sites, 100/676),
mirroring out-of-bounds directions could make two supposedly-different neighbors resolve to
the *same* mesh node, silently duplicating a "distinct electrode." Fixed by simply excluding
near-edge candidates from the sampling grid (576 sites now, also more physically realistic).
Re-running the full pipeline with the fix moved every headline number by roughly 0.01-0.03
AUC — small, and the qualitative story held — but worth catching and documenting rather than
assuming a bug that only touches background-class examples wouldn't matter.

**Cross-rotor validation (run 2026-08-21):** the k=1..8 sweep above was validated entirely
*within* rotor A — the spatial checkerboard split controls for nearby electrodes leaking
across train/test, but every electrode, train or test, still came from one single induced
rotor. To test real generalization, two more independent rotors were induced from different
stimulus sites on the same substrate (`opencarp/README.md` §"multi-rotor runs" — a patched
copy of openCARP's example script with the hardcoded stimulus location moved): rotor B from
the mirror-opposite side of the tissue, rotor C from below the patch. Both produced sustained
rotors (401/401 phase-singularity detection each) anchored at different points around the
same fibrotic patch — physically sensible, since each approaches from a different direction.

A first pairwise check (train on A, test on B, and vice versa) was **not** encouraging: mean
ROC-AUC peaked around k=2 (≈0.80) and *degraded* with more electrodes, dropping to ≈0.56
(near chance) at k=6 — suggesting the larger clusters' extra features were overfitting to
rotor-A-specific idiosyncrasies rather than learning transferable physiology.

With a third rotor, **leave-one-rotor-out** cross-validation (train on 2 pooled rotors, test
on the held-out third, rotated across all 3) told a more complete and encouraging story:

![Leave-one-rotor-out generalization](results/phase2_reentry_2026-08-20/leave_one_rotor_out.png)

Mean ROC-AUC stayed well above chance at every k (0.70-0.85), with **k=2 as a clear,
consistent peak across all three held-out rotors individually** (0.89 / 0.87 / 0.79). Unlike
the pairwise check, larger clusters (k=6-8) no longer collapsed toward chance once *two*
independent rotors' worth of data were pooled for training — pooling more independent
instances stabilizes the larger, more overfit-prone feature sets, exactly as you'd expect.
Practical read: **a 2-electrode local cluster gives the most reliable, generalizable signal**
found so far; larger clusters are plausible but need more independent training rotors than
we have to trust confidently.

**End-of-Phase-2 review gate:** per `docs/IMPLEMENTATION_PLAN.md` §8, this is the checkpoint
before Phase 3. Real, genuinely cross-rotor-validated signal exists (leave-one-out ROC-AUC
0.70-0.85 across 3 independent inductions) — a meaningfully stronger claim than a single
within-rotor number would support. Hard-decision numbers remain modest (small positive
counts per rotor, 22-33 sites each), and 3 rotors is still a small sample for fully trusting
the larger-k results, but the core finding (local cluster *variability* features carry real,
transferable signal about rotor-adjacency) held up under the most rigorous test we could run
at this scale — with the caveat that the LAT-timing half of the original mechanism turned out
not to hold up under scrutiny (see above and `docs/PHASE2_METHODOLOGY.md` §9a). Full
narrative walkthrough (words + code) in `docs/PHASE2_METHODOLOGY.md`; hands-on feature
exploration (raw-signal visualization, EDA, new feature scaffolding, model comparison) in
`opencarp/feature_exploration.ipynb`.

### Phase 3 — simulate ablation outcomes (2026-09-01)

After Phase 2 the project shifted to a **simulation-centered** framing — biophysics
simulation to predict optimal ablation sites, with ML scoped to making that prediction
fast. The phase-singularity trajectory tracker was rewritten (the old greedy linker froze on any
>3 mm frame jump — control rotor went from a 1-point "trajectory" to 99% frame coverage;
`docs/OPENCARP_SIMULATION_PRIMER.html` explains the sim layer). Three rotors were
regenerated with 5 s windows.

**Phase 3 (`docs/PHASE3_FINDINGS.md`, `opencarp/phase3/`):** branch a sustained rotor from a
mid-run state checkpoint, drop a circular non-conductive lesion (`gi_scale_vec`), and check
whether the rotor terminates. Across all three rotors:

- A 6 mm lesion on the tracked **phase singularity never terminates the rotor** — it
  re-forms its pivot a few mm away on the same circuit.
- The effective target is the **functional core** — the centroid of tissue that activates
  *weakly* over a cycle — which sits 7–8 mm from the phase singularity, near the anchoring
  fibrosis, and is the *same place* for all three rotors here.
- A ~5 mm-radius lesion at the functional core terminates each rotor in ~200 ms; targeting
  it instead of the pivot roughly **halves** the lesion size needed (rotor C: 12 mm → 5 mm)
  and removes the non-monotonic size behaviour (except for the strongly meandering rotor A).
- The functional core is **observable from electrogram amplitude** (corr ≈ 0.45–0.5),
  which is what Phase 4 predicts from.

![Phase 3 overview — the three rotors, their tracked cores, and the terminating lesion sites](docs/figures/fig_overview.png)

![Phase 3 target — functional core vs. phase singularity vs. which lesions actually terminate rotor A](docs/figures/fig_target.png)

More figures (`fig_mechanism_A.png`, `fig_circuit.png`, and the per-rotor
`efficacy_map_r6000.png` / `efficacy_radius_curve.png`) regenerate with
`opencarp/phase3/illustrate*.py` and `plot_efficacy.py` from the (gitignored) sim output.

### Phase 4 — real-time surrogate for the ablation target (`docs/PHASE4_FINDINGS.md`, run 2026-09-02)

A per-electrode gradient-boosted classifier predicts the functional-core ablation target
from a **600 ms window** of virtual-electrogram features, skipping the openCARP induction +
activation analysis. Trained on **32 rotors on varied fibrotic substrates** (generated via
the PSD protocol), evaluated leave-one-config-out:

- per-electrode ROC-AUC **0.970**
- **localisation error: median 1.8 mm** (vs. 4.2 mm for "ablate the lowest-voltage patch",
  6.5 mm for "always the centre") — inside the ~5 mm lesion radius that terminates a rotor
- holds on sparse grids: ~80 electrodes (mapping-catheter scale) → 2.8 mm
- **latency: ~53 ms vs. ~86 s** for the biophysics pipeline it replaces (~1600×)
- dominant feature: local mean unipolar amplitude (confirms the Phase 3 signal)
- closed-loop: ablating the predicted spot slows the (robust, PSD-seeded) rotors ~1.6×;
  clean termination shown for the Phase 3 rotors, transitively expected here

This is the "optimize the simulation compute … to provide real-time feedback" goal — a
model that reproduces the simulation's answer fast enough to use during a procedure.

## 7. Interactive Viewer

A browser-based viewer over the Phase 1 test-set predictions — self-contained single HTML
file, no server, works fully offline (aside from optionally loading its Google Fonts).

**Live:** https://neillwhite.github.io/cardiac-ep-portfolio/ecg_viewer.html
(served copy: `docs/ecg_viewer.html`). It's regenerated from a checkpoint +
`results/viewer_data.json`, so treat the committed copy as a build artifact rather than
hand-edited source.

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

then splice the resulting `results/viewer_data.json` into the viewer template and rewrite
`docs/ecg_viewer.html` (the served copy). The header's model name and export date update
automatically from the JSON — no other viewer changes needed.

## 8. Limitations & Future Work

- **Multi-label records dropped, not multi-hot encoded:** PTB-XL records with
  more than one diagnostic superclass are excluded from training entirely for a
  clean single-label baseline. This drops 25.5% of all records (21,799 total →
  16,244 single-label). Meaningful chunk of the data, worth revisiting if the
  single-label ceiling turns out to be the real bottleneck rather than class
  weighting.
- Baseline uses a simple 1D-CNN; a resnet-style or transformer architecture would likely
  improve macro F1 but adds complexity not needed for a first pass.
- No handling yet of lead-missing or noisy real-world recordings.
- **2D tissue patch, not 3D:** the reentry substrate is a flat 5cm×5cm sheet (openCARP's
  `21_reentry_induction` example), not a volumetric/anatomically-shaped chamber model —
  real atrial tissue has wall thickness and curvature that affect rotor stability.
- **Single induction attempt at one candidate site:** reentry was induced (and ground truth
  generated) from one S1-S2-style stimulus location near the fibrotic patch edge, not a
  systematic scan over multiple sites/timings — the induced rotor is a real, sustained
  result, but it's one realization, not a statistical characterization of where rotors
  tend to anchor on this substrate.
- **Phase computed via raw Hilbert transform** of mean-subtracted Vm, the simplest standard
  method — optical-mapping literature also uses a phase-space (Vm, dVm/dt) formulation,
  which can be more robust to noise; not needed here since simulated Vm is noise-free.
- **Fixed 3mm radius for ablation-target labeling** around the phase-singularity trajectory
  is a simplification standing in for any clinically-derived lesion-size rationale.
- **Unipolar EGM approximated as local Vm**, not a true bidomain/lead-field-derived
  extracellular potential — a monodomain-only simulation never computes the extracellular
  field (only `Vm`, the transmembrane difference); this is the standard workaround for
  feature-engineering studies where full forward-model electrograms are out of scope.
  Bipolar EGM (the difference of two such signals) uses exactly the standard real-world
  derivation procedure, but since it's built from the approximate unipolar signals above, the
  result inherits that same approximation rather than escaping it.
- **Classifier training data comes from one simulated rotor on one mesh** (33 positive
  electrode sites total) — the electrode-count sweep result (more local electrodes → better
  ROC-AUC) is a real, reproducible finding, but the classifier itself hasn't been validated
  against an independently induced rotor; that's the natural next experiment before trusting
  it beyond this one substrate.
- **Dominant-frequency search restricted to a 3-15Hz band**, the standard AF-literature
  convention — an unrestricted FFT peak search picks up high-frequency edge content from
  sharp, non-periodic deflections rather than genuine periodicity.
- openCARP integration currently generates synthetic electrograms only; extending the
  electrical model toward pulsed-field-ablation-style lesion prediction (electric field
  magnitude as a lesion-likelihood proxy) is the natural next step, given the target
  company's PFA-based platform.
- **Phase 3 lesions are instantaneous conductivity holes**, not modelled pulsed-field
  physics (field magnitude → electroporation threshold → lesion shape); 3 rotors, one
  fibrosis-model family, 2D.
- **Phase 4 rotors are seeded (PSD), not paced**, and the closed-loop `gi_scale_vec` lesion
  mechanism silently no-ops on PSD-seeded state restarts (cause unresolved; worked around by
  baking lesions into the substrate) — so Phase 4's *termination* claim rests transitively
  on Phase 3.

## 9. Why This Project

The hard, unsolved part of cardiac ablation is knowing *where* to ablate — and unlike
arrhythmia classification, nothing labels "optimal ablation site" directly. This project
takes that on end to end: interpret real cardiac signals (Phase 1), build a rotor in
simulation and define a defensible target (Phase 2), test that target by simulating the
ablation itself (Phase 3), then make the prediction fast enough to be useful in real time
(Phase 4) — documenting every simplification and reporting the honest numbers throughout.
