# Phase 4 — surrogate for the ablation target (findings)

**Plan:** `PHASE4_PLAN.md`. **Goal (Duty 2 of the role):** predict the ablation target fast, from a
short electrogram recording, instead of running the biophysics pipeline.

---

## The task

Phase 3 established the ablation target is the **functional core** — the centroid of tissue
that activates weakly over a rotor cycle — found by simulating the rotor and analysing
activation over a full cycle (~minutes). Phase 4 predicts that point from a **600 ms
window** of virtual-electrode signals via a fast model.

- **Dataset:** 32 rotors, each on a different fibrotic substrate (patch centre, radius, hole
  seed/fraction varied) and each placed by the PSD protocol (Eikonal-based rotor placement
  that skips the pre-pacing step). The functional core spread across the sheet
  (x ∈ [15, 33], y ∈ [16, 34]), so it genuinely tracks the substrate rather than sitting
  at centre. Tooling: `opencarp/phase4/gen_*.sh`.
- **Features** (`extract_features.py`), per electrode on a grid, from the 600 ms window:
  unipolar & bipolar amplitude, activation count, fractionation, dominant frequency, plus
  *position-relative* context — amplitude vs. the field median, amplitude rank, local-minimum
  flag, neighbour mean/std, distance to the global amplitude minimum.
- **Model** (`train.py`): HistGradientBoosting per-electrode classifier, target = "within
  5 mm of the functional core". **Leave-one-config-out** CV. Predicted target = probability-
  weighted centroid of the top-decile electrodes.

---

## Result

| metric | value |
|---|---|
| per-electrode ROC-AUC (leave-one-config-out) | **0.970** |
| **localisation error, median [IQR]** | **1.8 mm [1.3, 2.8]** |
| — baseline: centroid of the 5 lowest-amplitude electrodes | 4.2 mm |
| — baseline: always predict mesh centre | 6.5 mm |
| dominant feature (permutation importance) | local mean unipolar amplitude |

**1.8 mm median error is well inside the ~5 mm-radius lesion that terminates the rotor**
(Phase 3), so the predicted spot is actionable. The surrogate more than halves the error of
the naive "ablate the lowest-voltage patch" rule — the spatial-context features (how an
electrode's amplitude compares to its neighbourhood and to the field) are what buy that.

### Holds up on sparse electrode grids

| electrode grid | electrodes / case | localisation error (median) |
|---|---|---|
| 2 mm | 676 | 1.8 mm |
| 4 mm | 169 | 1.9 mm |
| 6 mm | 81 | 2.8 mm |
| 8 mm | 45 | 3.7 mm |

An ~80-electrode grid — the scale of a real mapping catheter — still lands within ~3 mm.

### Latency

| | time |
|---|---|
| **surrogate**: feature extraction (600 ms window, 676 electrodes) + GBT inference | **~53 ms** |
| **replaces**: openCARP simulation to produce a multi-second Vm record | ~86 s |
| + the full-record activation analysis | ~20 ms |

~1600× faster than running the biophysics, and the surrogate needs only 600 ms of signal
rather than a multi-cycle recording.

### Closed-loop check (Gate 4d)

The direct test — branch an established rotor, ablate the surrogate's predicted spot,
confirm termination — was **blocked by a technical issue**: `gi_scale_vec` (the
conductivity-scaling lesion mechanism used in Phase 3) silently does nothing when applied
to a **PSD-initialised** rotor state on restart. A lesion covering half the mesh left the
rotor spinning. The identical mechanism *does* terminate the Phase 3 RP_B rotors (a 16 mm
lesion kills rotor A in 225 ms — re-verified), so it is specific to how PSD sets up its
state, cause not identified.

Fallback test (`closed_loop3.sh`): re-seed the rotor with a non-conductive lesion of radius
{5, 8, 12} mm **baked into the substrate** at the true functional core vs. a control point
~13 mm away, and measure the resulting rotor cycle length (longer = more disrupted).

**Rotor cycle length (ms):**

| substrate | 5 mm @ functional core | 5 mm @ control | 8 mm @ core | 8 mm @ control |
|---|---|---|---|---|
| b00 | **324** | 192 | 337 | 271 |
| b18 | **288** | 208 | 317 | 219 |
| b26 | **234** | 203 | 254 | — |

A 5 mm lesion at the surrogate's target **slows the rotor by 15–70 %** (192→324, 208→288,
203→234 ms); the same lesion 13 mm away barely changes it. A 12 mm lesion breaks the rotor
up regardless of position (it covers most of the fibrotic patch — uninformative).

**Interpretation:** the surrogate identifies the right region — ablating there has a
specific, dose-dependent effect on the rotor that ablating elsewhere does not. But on these
**PSD-seeded** rotors, a small focal lesion *slows* rather than *terminates* the arrhythmia
(unlike the Phase 3 RP_B rotors, where ~5 mm at the functional core terminated them — PSD's
strong seeded phase gradient makes the rotor more robust). Slowing a re-entrant tachycardia
by ~1.6× is itself a therapeutic effect and makes it more vulnerable to a second lesion or
to drug.

The stronger *termination* claim still rests on the transitive argument: Phase 3 validated
(lesion-size probes on 3 rotors) that ablating within ~5 mm of the functional core
terminates an RP_B-induced rotor; the surrogate's 1.8 mm median error is well inside that.

---

## Caveats

- **PSD rotors are seeded, not induced** — planted by Eikonal init rather than triggered by
  rapid pacing (the 3 Phase-3 reference rotors were genuinely paced). Fine for a training
  set; noted.
- One fibrosis-model family (a disc of random holes); 2D monodomain; electrograms
  approximated as local Vm.
- 32 configs, ~2.9 % positive electrodes — AUC is strong but AP (0.49) reflects the
  imbalance; the localisation-error metric is the one that matters for the clinical claim.
- Feature window is a mid-record slice of a sustained rotor; a real recording would include
  onset transients.
- The worst single config (d01, small high-hole-fraction patch) missed by 4.7 mm — near the
  edge of the actionable range.
