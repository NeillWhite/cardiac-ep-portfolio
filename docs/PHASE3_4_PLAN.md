# Phases 3 & 4 — Simulate ablation outcomes, then an ML surrogate for real-time

**Status:** agreed 2026-08-28. This is the current plan for the project's back half. It
replaces the old `IMPLEMENTATION_PLAN.md` §5 ("Phase 3 — Real-Time Optimization"). A 3D
non-contact-mapping variant was considered and dropped.

**Framing:** the biophysics simulator (openCARP + the phase-singularity / ablation
pipeline) is the centre of gravity — *develop biophysics-based simulations to predict
optimal ablation sites*. The ML is scoped to *making that prediction fast enough to use
during a procedure* — a surrogate that reproduces the simulation's answer in milliseconds.
**No C++** (openCARP itself is compiled, so the latency comparison is honest).

---

## Phase 2′ — solidify the substrate + rotor simulation *(small, mostly plumbing)*

- **5 s regeneration** of the 3 reference rotors — re-induce with RP_B, free-run window
  patched 600 ms → 5000 ms, plus state checkpoints at +1/+2/+3/+4/+4.9 s for Phase 3
  branching. Script: `opencarp/regen_5s/`. Fixes the `lat_spread_ms` / `dom_freq_*`
  artifacts.
- Re-run `phase_singularity.py` + `export_for_notebook.py`; refresh the notebook's feature
  sections against a proper multi-beat window.
- **Risk:** the rotor may not sustain 5 s on the 5 cm patch. If it dies early, report the
  actual survival time and use what we get (still ≫ 400 ms); consider a larger sheet or
  stronger anchor only if it's badly short.
- The simulation layer is documented in `docs/OPENCARP_SIMULATION_PRIMER.html` before Phase 3 builds on it.

---

## Phase 3 — Simulate ablation outcomes → "optimal ablation site" *(the spine)*

Turn "rotor core = ablation target" (a proxy label) into an actual in-silico test of
ablation.

1. **Branch point.** From a sustained rotor, take a state checkpoint (rotor established,
   core position known from Phase 2's tracking).
2. **Candidate sites.** The tracked rotor core, plus a grid of alternatives across the sheet
   (e.g. 20–40 candidates, spaced ~4–6 mm).
3. **Counterfactual sims.** For each candidate: restart from the checkpoint (`-start_statef`),
   add a circular lesion — a non-conductive region, same mechanism as the fibrotic patch
   (`-tagreg` / a `.regele` element list) — at that site, run 1–2 s.
4. **Outcome metric.** Did the rotor terminate? Time-to-termination? Post-lesion activity
   level. Also record lesion size (radius).
5. **Deliverable.** An **ablation-efficacy map** over candidate sites for each rotor: which
   sites terminate the rotor, how fast, and at what lesion size. Optimal site = terminates
   with the smallest lesion. Visualize over the tissue with the rotor-core trajectory
   overlaid.

Design decisions to make early: lesion radius sweep (single fixed radius first, then vary);
candidate grid density; termination criterion; how long "terminated" has to hold.

**Gate G3a:** show the efficacy map for one rotor — does ablating the tracked core actually
terminate it, and are there other effective sites? — before scaling to all 3 rotors and
before building the surrogate.

---

## Phase 4 — ML surrogate for real-time

Phase 3 costs N counterfactual simulations per case (minutes–hours). Train a model that
predicts the ablation-efficacy map **without** running them.

1. **Dataset.** Phase 3 over the 3 reference rotors gives 3 × (20–40 candidates) labelled
   examples — thin. Extend: ~20–40 additional cheap 2 s inductions varying the fibrosis
   pattern / stimulus site, each with its own candidate sweep. Plumbing; I generate it.
   "I varied the scar geometry and pacing site for diverse rotors" is the one-line summary.
2. **Inputs.** Per candidate site: local electrogram features (the Phase 2 feature set,
   cleaned) and/or a short `Vm` window and/or the local substrate map. Decide after a quick
   feature-importance pass.
3. **Model.** GBT on features, or a small CNN on the local field. Keep it interpretable per
   project ethos.
4. **The inference-optimization story (the differentiator):**
   - profile end to end — where does the time actually go (openCARP solve vs. phase mapping
     vs. the counterfactual sweep)
   - accuracy vs. latency Pareto frontier
   - the "tricks": model size, feature-subset selection for latency, quantization,
     distillation, ONNX export, batching, caching
   - **headline:** *"the biophysics search takes X min; the surrogate reproduces its
     recommended site to within Z mm in Y ms."*

**Gate G4:** the Pareto figure + an honest statement of the speedup and the accuracy it
costs.

---

## Phase 5 — PFA-realistic lesion *(optional; connects to a PFA ablation platform)*

Replace Phase 3's non-conductive disk with a PFA lesion: applied electric field from an
electrode geometry (electrostatics solve on the mesh) → electroporation-threshold contour →
lesion shape. Compare the PFA-shaped lesion's efficacy to the idealized disk. Only if time
remains after 1–4 are solid.

---

## Limitations to document as they arise

2D monodomain; no fiber anisotropy; idealized circular substrate; single rotor per run;
lesion = instantaneous conductivity change (no acute electrophysiological effects of the
pulse itself in Phases 3–4); surrogate trained on simulation only, no real-data validation;
latency measured on one machine.

---

## Review gates summary

| Gate | Shows |
|---|---|
| Phase 2′ | 5 s rotors regenerated; feature artifacts resolved; survival time reported |
| G3a | Ablation-efficacy map for one rotor — does core ablation terminate it? |
| G3b | Efficacy maps for all 3 rotors |
| G4 | Surrogate accuracy–latency Pareto; speedup + accuracy cost stated honestly |
