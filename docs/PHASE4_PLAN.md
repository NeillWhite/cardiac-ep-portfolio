# Phase 4 — ML surrogate for the ablation target

**Status:** started 2026-09-01, after **gate G3 closed** (see `PHASE3_FINDINGS.md`).
Supersedes the Phase 4 sketch in `PHASE3_4_PLAN.md`, which predicted "the efficacy map" —
G3 showed that target is sparse and rotor-specific, and that the real target is the
**functional core**.

**Role anchor:** Duty 1 (biophysics simulation → predict optimal ablation sites) + Duty 2
(optimize the simulation compute with ML for real-time feedback). No C++.

---

## 1. What G3 established, that Phase 4 builds on

- The ablation target is the **functional core** — the centroid of tissue that activates
  *weakly* over a rotor cycle (`opencarp/phase3/functional_core.py`). It is **not** the
  phase singularity (7–8 mm away) and it co-locates with the anchoring fibrosis.
- A ~5 mm-radius lesion there terminates the rotor in ~200 ms (functional-core size probe).
- **Electrogram / Vm amplitude tracks proximity to it** (corr ≈ 0.45–0.5)
  — dominant frequency and fractionation do not.
- The slow part of "find the target" is the openCARP induction + multi-second free-run +
  activation analysis (~minutes). That is what the surrogate replaces.

---

## 2. The surrogate task

**Per-electrode classification**, reusing the Phase 2 machinery with the corrected target:

- **Input:** local electrogram features at each of a grid of virtual electrodes — bipolar
  amplitude (primary), fractionation, LAT, dominant frequency, plus small-cluster
  aggregates (the Phase 2 feature set, `opencarp/extract_electrode_features.py` /
  `extract_cluster_features.py`), from a short recording window.
- **Label:** is this electrode within R (≈ 5 mm) of the simulation's functional-core
  centroid?
- **Prediction → ablation target:** centroid of the positive-predicted electrodes
  (probability-weighted).
- **Metric of record:** geodesic distance between predicted and true functional core;
  fraction of cases where the predicted centroid is within one lesion radius; and — the
  closing check — does ablating the *predicted* spot terminate the rotor in a held-out
  openCARP run.

Per-electrode (not per-simulation (x,y) regression) because ~30 simulations × ~100
electrodes gives a usable training set; one (x,y) per sim does not.

---

## 3. Stages

### Stage 1 — dataset

**Infra note (2026-09-01):** varying the fibrosis via the RP_B induction path is blocked.
RP_B needs a prepace steady state that (a) cannot be shared across substrates (the "region
layout must be identical" checkpoint-restore check fails once node tags change) and
(b) segfaults when regenerated — `bench` (single-cell init) crashes with SIGSEGV in this
openCARP image. Every earlier success reused the shipped prepace and never hit this.

**Path forward — two tracks:**

1. **Bootstrap from existing data (now).** We already have ~120 full rotor `Vm` fields:
   the 3 reference rotors plus the ~85 *sustained* post-lesion efficacy sims (rotor still
   spinning, often relocated by the lesion — so the functional core sits in somewhat
   varied places). One substrate, but enough to build and shake out the whole
   feature → classifier → localisation pipeline.

2. **PSD-generated varied dataset (parallel).** The **PSD** protocol places rotors by
   Eikonal initialisation and **skips prepace/bench entirely** — no segfault, and rotor
   placement is deterministic. Vary: PSD seed location, fibrosis centre/radius/seed,
   chirality. Needs `PSD_algorithm.py`'s `-tend 500` patched to ~3000 ms and a seed-count
   that yields one clean rotor. `opencarp/phase4/gen_one.sh` (currently RP_B-based, to be
   switched to PSD).

**Gate 4a:** functional cores spread across the sheet; amplitude still tracks them.

### Stage 2 — the classifier *(user drives, I support)*

- HistGBT on per-electrode features (interpretable, fast) → probability of
  functional-core-adjacency. Small CNN on the electrode-grid feature image as an
  alternative if GBT plateaus.
- **Leave-one-substrate-out** cross-validation.
- Report: per-electrode AUC/AP, and the downstream centroid-localisation error.

### Stage 3 — the real-time / compute-optimisation story *(Duty 2)*

- **Profile** the full pipeline (openCARP induction + free-run + activation analysis)
  wall-clock vs. the surrogate (feature extraction + inference) on this machine.
- **Accuracy vs. input**: full field → sparse electrode grids (100 / 50 / 20) → shorter
  recording windows (3 s / 1 s / 0.5 s). The Pareto frontier.
- **Latency tricks**: feature-subset selection, model size, ONNX export, batching.
- **Headline:** "the biophysics pipeline localises the ablation target in *X* min; the
  surrogate does it within *Z* mm in *Y* ms."

### Stage 4 — closed-loop validation

Held-out substrates: take the surrogate's predicted target, ablate there in openCARP
(`lesion_sweep.py --at-xy`), confirm the rotor terminates. Report the hit rate and, for
meandering rotors, the lesion radius needed.

---

## 4. Caveats to carry

2D monodomain, isotropic; single fibrosis-model family; electrograms approximated as local
Vm (monodomain, no true extracellular potential — `PHASE2_METHODOLOGY.md` §7); functional
core defined over one auto-detected cycle; meandering rotors keep a radius-sensitive
outcome regardless of targeting.

---

## 5. Review gates

| Gate | Shows |
|---|---|
| **4a** | Dataset generated; functional cores spread across the sheet; amplitude still tracks them |
| **4b** | Per-electrode classifier CV performance + centroid-localisation error, leave-one-substrate-out |
| **4c** | Accuracy–latency Pareto; the headline latency/accuracy number |
| **4d** | Closed-loop: ablating the predicted spot terminates held-out rotors |
