# Cardiac EP Simulation + ML — Implementation Plan

Scope: machine learning + biophysics simulation for cardiac electrophysiology mapping and
pulsed-field ablation (PFA). This plan is written to be handed to a coding agent (e.g.
Claude Code) running on a Linux server with Docker and GPU access.

Companion artifact: `cardiac-ep-portfolio.zip` (Phase 1 scaffold, already smoke-tested with
synthetic data — code compiles and the training/eval loop runs end to end. Real PTB-XL
download/training has NOT been run yet since the scaffolding was built in a sandbox without
network access to PhysioNet).

---

## 1. Goal & Success Criteria

The project sets out to demonstrate, in priority order:

1. Hands-on cardiac simulation tooling (openCARP)
2. ML applied to cardiac signals (ECG, unipolar/bipolar electrograms)
3. Simulation compute optimization via ML/statistics for near-real-time feedback
4. A defensible, literature-grounded definition of "optimal ablation site" — not just a
   model that outputs a number, but a documented rationale a reviewer with EP knowledge
   would find credible
5. (Stretch) A nod to pulsed-field-ablation-specific biophysics — PFA, not thermal RF
   ablation, is the modality of interest here

**Definition of done for the whole project:** a public repo with working code, a README
that reads like a design document, real (not synthetic) results with honest limitations,
and at least one artifact (plot, notebook, or writeup) per phase that can be walked
through and explained line by line.

---

## 2. Environment Prerequisites (Linux server)

Before the agent starts, confirm/have it verify:

- Docker installed and the user has permission to run it (`docker run hello-world`)
- Python 3.10+ with `venv` or `conda` available
- Disk space: reserve **~15GB** (PTB-XL 100Hz ~1.7GB, 500Hz ~5GB if you go there later,
  openCARP Docker image ~2-4GB, mesh/simulation output can balloon quickly — IGB files
  from even modest 3D meshes are often hundreds of MB per simulation run)
- GPU: check `nvidia-smi`. Not required for Phase 1 (the CNN baseline is small). Relevant
  for Phase 3 if you want the ML surrogate model to train fast, and openCARP itself can
  optionally offload linear algebra to GPU via Ginkgo — nice-to-have, not required
- RAM: PTB-XL 100Hz fully loaded into memory is roughly 1GB as float32 — fine on any modern
  server. Flag this as a non-issue unless the agent decides to use the 500Hz version, in
  which case reconsider (~5x the size) and prefer lazy/chunked loading instead of the
  current "load whole split into one array" approach in `preprocess.py`

**First agent task:** run a short environment check script (Docker present, Python version,
disk free, GPU present) and report back before doing anything else. Don't let it silently
proceed on missing prerequisites.

---

## 3. Phase 1 — ECG Baseline (refine existing scaffold, then run for real)

Scaffold already exists in `cardiac-ep-portfolio/`. Have the agent:

1. Extract the zip, create a venv, `pip install -r requirements.txt`
2. Run `download_ptbxl.py` for real. **Known risk:** `wfdb.dl_database` behavior/paths can
   drift between wfdb versions — if it fails, have the agent check the current PTB-XL page
   on PhysioNet directly rather than guessing at fixes
3. Run `preprocess.py`, then sanity-check class balance printed to console before training —
   PTB-XL superclasses are meaningfully imbalanced (NORM dominates), the weighted loss in
   `train_baseline.py` should compensate but confirm the printed class counts look sane
   (rough expectation: NORM is roughly half of single-label records, the rest split across
   MI/STTC/CD/HYP — treat this as a plausibility check, not a hard spec)
4. **Add before training, not currently in scaffold:**
   - Fixed random seed (numpy + torch) for reproducibility — add to `train_baseline.py`
   - Early stopping (patience on val loss) so it doesn't run all 30 epochs by default
   - A `--seed` CLI arg
5. Train, then evaluate. Target: macro F1 ≥ 0.70 is a reasonable bar for this baseline
   architecture on the 5-superclass task (published benchmarks on this exact task with
   simple CNNs tend to land in the 0.70-0.80 macro F1 range) — if it's far below that,
   something is likely wrong in preprocessing/label mapping, not the model, so debug there
   first
6. **Deliverable for this phase:** `results/metrics.json`, a confusion matrix plot (agent
   should add a matplotlib plot to `evaluate.py` — currently only prints the raw matrix),
   and 3-4 example ECG traces with predicted vs. true label saved as figures for the README

**Review gate:** stop here and show me results before Phase 2. I want to see the actual
macro F1 and confusion matrix, not just "it ran."

---

## 4. Phase 2 — openCARP Simulation (the important, underspecified part)

This is where the previous scaffold was thin. The hard problem isn't running openCARP —
it's **what ground truth "optimal ablation site" means** in a simulation, since nothing
labels that for you the way PTB-XL labels arrhythmia class.

### 4.1 Recommended ground-truth definition: rotor/phase-singularity tracking

This is a well-established technique in the EP literature (not something invented for this
project) and is tractable with openCARP + Python:

1. Set up a 2D tissue sheet (or simple 3D slab) with a region of reduced conductivity to
   represent scar/fibrosis — this creates the substrate for reentry
2. Use an S1-S2 cross-field stimulation protocol (standard in openCARP tutorials) to induce
   a reentrant rotor anchored near the low-conductivity region
3. Extract the transmembrane voltage field over time across the mesh
4. Compute **phase** at each point via Hilbert transform of the local voltage time series
   (standard technique for rotor visualization — search "phase singularity detection cardiac
   optical mapping" for the canonical method if the agent needs a reference implementation
   approach)
5. Identify **phase singularities** (points where phase wraps through all values around a
   small loop) frame by frame — these mark the rotor core
6. Label mesh points within a small radius of the phase singularity's trajectory over the
   simulation as the **ablation target ground truth**; everything else is background

This gives you a genuine spatial ground-truth label grounded in actual EP concepts (rotor
core ablation is a real clinical strategy for AF), not an arbitrary heuristic.

### 4.2 What the ML model predicts

Given this ground truth, frame the ML task as: **from a short window of simulated
unipolar/bipolar electrograms at candidate electrode sites, predict whether that site is
near a rotor core** (i.e., an ablation target), without needing the full phase-mapping
computation at inference time. This is the actual practical value proposition — phase
mapping needs the full time series and dense spatial sampling; a model that infers "this is
a rotor-adjacent site" from local electrogram features alone is the kind of shortcut that's
genuinely useful for real-time guidance. This maps directly to "hand-on experience with
cardiac simulation tools" + "ML in cardiac anatomy and electrophysiology, ECG, unipolar/
bipolar electrogram" — the standard clinical signals.

Feature candidates to extract per electrode site (standard EP signal features, keep it
interpretable rather than reaching for deep learning here first):
- Bipolar voltage amplitude
- Electrogram fractionation (number of deflections / spectral characteristics)
- Local activation time relative to a reference
- Dominant frequency (via FFT of the local electrogram)

A gradient-boosted tree or small MLP on these features is a defensible, explainable first
model — resist the urge to jump straight to a deep model on raw electrogram waveforms for
this phase; interpretability matters more than squeezing out accuracy here, and it's easier
to defend.

### 4.3 Concrete openCARP steps

1. Docker setup (see previous `opencarp/README.md` in the scaffold) — **agent must verify
   current image name, tutorial paths, and CLI against live docs at
   https://opencarp.org/download/installation and https://opencarp.org/documentation**,
   since my earlier scaffold guessed at paths that may not match the current release
2. Generate/use a simple 2D mesh via `meshtool`
3. Define tissue properties: normal myocardium conductivity + a lower-conductivity patch
4. Write the S1-S2 stimulation protocol (openCARP tutorials have a canonical example of
   this — reuse/adapt rather than writing from scratch)
5. Run the simulation, export transmembrane voltage over time (IGB format — openCARP/
   carputils has Python readers for this, use those rather than hand-parsing IGB)
6. Post-process: Hilbert transform → phase → phase singularity detection → ablation target
   labels (this is new code, not something openCARP gives you natively)
7. Sample virtual electrodes across the mesh, extract electrogram features at each,
   join with the ablation-target labels from step 6
8. Train the classifier from 4.2

**Review gate:** stop after step 6 (ground truth generation) and show me a visualization of
the rotor trajectory and labeled ablation-target region before building the classifier on
top of it. This is the part most likely to need iteration, and it's cheap to check before
you build ML on top of it.

---

## 5. Phase 3 — Real-Time Optimization

Only start once Phase 2 has a working simulation → feature → prediction pipeline.

1. Profile the Phase 2 pipeline end to end (Python `cProfile` or simple timing) and identify
   the actual bottleneck — likely the PDE solve in openCARP itself, or the Hilbert
   transform/phase-singularity detection if done naively over a fine mesh
2. Define a concrete latency target and justify it (e.g., "clinical mapping systems update
   in the sub-second range" — cite a real source for whatever number you pick rather than
   inventing one)
3. Build an ML surrogate that approximates the expensive step:
   - If the bottleneck is the full-mesh PDE solve: train a model that predicts
     rotor-likelihood or electrogram features directly from tissue/geometry properties,
     skipping full simulation for a given configuration
   - If the bottleneck is phase-singularity detection: this step is already fast, so the
     surrogate framing may not apply — be honest about this in the writeup rather than
     forcing an ML surrogate where a simple algorithmic optimization is the real answer
4. Benchmark before/after with real wall-clock numbers on the actual server, not estimates
5. Optional but adds credibility (and exercises C++ alongside the Python): reimplement the
   single hottest inner loop in C++ (pybind11 or a simple CLI called from Python) and
   report the speedup. Don't rewrite the whole pipeline in C++ — one well-chosen hot path
   is a stronger, more honest signal than an unfinished full rewrite

**Review gate:** show me the profiling results before deciding what to optimize — don't
assume where the bottleneck is.

---

## 6. Phase 4 — PFA-Specific Angle

Lowest priority, do only if time remains after Phases 1-3 are solid.

1. Literature pass first, no code: search "irreversible electroporation cardiac tissue
   model," "pulsed field ablation lesion simulation," "electroporation threshold
   myocardium." Summarize the core physics in a few paragraphs before writing any code —
   this is a credibility check on yourself as much as prep for implementation
2. Simplified implementation approach: rather than a full electroporation model (which is
   its own research area), compute the electric field magnitude distribution from applied
   electrode voltages using a simple electrostatics approximation on the existing mesh
   (this is a different physics problem than the bidomain electrophysiology model used in
   Phases 1-3 — be clear in the writeup that this is a simplification layered on top, not
   an integrated multi-physics model)
3. Treat a field-strength threshold as a proxy for "irreversible electroporation occurred
   here" and compare the predicted lesion footprint against the rotor-core ablation targets
   from Phase 2 — do they overlap? This comparison, done honestly with its limitations
   stated, is more valuable than a polished-looking but physically unjustified model

**This phase is explicitly optional** — flag to the agent that if time runs short, a
well-documented Phase 1-3 plus a clearly written "here's how I'd extend this to PFA
biophysics" section in the README is a perfectly credible stopping point.

---

## 7. Documentation Requirements (apply to every phase)

- Update the root README's "Verification / Results" section with real numbers/figures as
  each phase completes — don't let this pile up to the end
- Every simplifying assumption (single-label ECG filtering, 2D vs 3D mesh, electrostatics
  approximation for PFA, etc.) gets one explicit sentence in the README's Limitations
  section. Reviewers in this field will immediately spot unstated simplifications; naming
  them yourself reads as competence, not weakness
- Keep a running `results/` directory with dated subfolders per experiment run so you can
  reference "run from Aug 20" if asked about methodology later

---

## 8. Milestones & Review Gates (summary)

| Gate | What I want to see before continuing |
|---|---|
| End of Phase 1 | Real PTB-XL macro F1 + confusion matrix |
| Mid Phase 2 | Rotor trajectory visualization + ablation-target labeling, before ML is layered on |
| End of Phase 2 | Classifier performance on rotor-adjacency prediction from electrogram features |
| Mid Phase 3 | Profiling results, before committing to what to optimize |
| End of Phase 3 | Before/after latency benchmark, real numbers |
| Phase 4 (optional) | Literature summary before any code |

---

## 9. Known Risks / Open Questions to Flag, Not Silently Resolve

- openCARP tutorial paths/CLI may have changed since my scaffold was written — agent should
  verify against live docs, not trust my guessed paths
- PTB-XL single-label filtering drops a meaningful fraction of records (some patients have
  multiple superclass diagnoses) — acceptable simplification for a baseline, but the agent
  should report what fraction got dropped so we can decide if it's acceptable
- The rotor/phase-singularity approach in Phase 2 requires an actual reentrant rotor to
  form in simulation — this may take some parameter tuning (conductivity values, stimulus
  timing) to reliably induce; budget iteration time here rather than expecting it to work
  first try
- No IRB/patient-data concerns since PTB-XL and openCARP outputs are public/synthetic
  respectively, but worth a one-line README note confirming this for a healthcare-adjacent
  employer

---

## 10. Agent Handoff Prompt

Paste this to the coding agent on the Linux server to kick off Phase 1:

```
I'm building a cardiac-electrophysiology simulation + ML project. Full plan is in
docs/IMPLEMENTATION_PLAN.md — read it in full before starting. Also extract
and review cardiac-ep-portfolio.zip, which has a Phase 1 scaffold already written and
smoke-tested with synthetic data (but never run against real PTB-XL data).

Start with Section 2 (environment check) and Section 3 (Phase 1) only. Do not proceed to
Phase 2 until I've reviewed real Phase 1 results — this is an explicit review gate in the
plan. Flag any assumptions you have to make, especially anywhere the scaffold's guessed
file paths or library behavior don't match what you find in practice.
```
