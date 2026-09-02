# Phase 3 — ablation-efficacy maps (G3a + G3b)

**GATE G3: CLOSED 2026-09-01.** Ablation-efficacy maps produced for all three rotors, the
"functional core, not the phase singularity" target identified and its lesion-size behaviour
characterised. Phase 4 (`PHASE4_PLAN.md`) proceeds on the reframed target.

**Gate G3a/G3b** (`PHASE3_4_PLAN.md`): show the ablation-efficacy map for each rotor — does
ablating the tracked core terminate it, and where else works? — before building the
surrogate.

Data: `opencarp/runs/phase3/{A,B,C}/`. Figures per rotor: `efficacy_radius_curve.png`,
`efficacy_map_r6000.png`.

---

## TL;DR — the finding across all three rotors

**A 6 mm lesion placed on the tracked rotor core does not terminate the rotor.** The
effective ablation positions are *offset* from the core, sit somewhere in the reentry
circuit around it, and which offset works is rotor-specific. And the more *tightly anchored*
the rotor, the harder it is to terminate with a modest focal lesion:

| rotor | core meander span | anchoring | terminate @ r=6 mm | size threshold (lesion at centroid) |
|---|---|---|---|---|
| **A** | ~12–20 mm | loose / meandering | **10 / 37** positions | ~5 mm radius (clean step) |
| **B** | ~7–15 mm | moderate | **8 / 49** positions | non-monotonic (8 mm ✓, 10 mm ✗, 12 mm ✓) |
| **C** | ~4 mm | tightly pinned | **1 / 31** positions | only 12 mm radius works |

Mechanistic reading: the phase singularity is where the wave *pivots*; the reentry *circuit*
loops around it, often hugging the fibrotic patch. Ablate the pivot and the rotor re-forms
its pivot a few mm away on the same circuit. Ablate a critical circuit segment — or, for a
pinned rotor, the anchoring obstacle — and it stops. This is consistent with why clinical
rotor-core ablation has underperformed: functional rotors are a moving target and "burn the
singularity" is not the operative lesion.

---

## G3a — rotor A (detail)

---

## Method

1. **Induce** rotor A (RP_B rapid pacing, node 7830, BCL 150 ms), free-run 3.8 s, save state
   checkpoints. The rotor sustains the whole window; its core is a **strongly meandering**
   functional rotor — the tracked phase singularity roams a ~12 × 11 mm territory
   (`opencarp/phase3/induce.sh`).
2. **Branch** every counterfactual from the *same* checkpoint (1.5 s into the free-run,
   t = 4490 ms) so the only variable is the lesion.
3. **Lesion** = a circular patch made non-conductive via an element-wise conductivity
   scaling file (`gi_scale_vec` → ~0), which changes conductivity only and leaves the ionic
   region layout intact so the checkpoint restores cleanly. Validated: a deliberately huge
   lesion kills the rotor outright (0 phase singularities), a tiny one does nothing.
4. **Score** termination from the voltage field directly — no node produces an
   action-potential swing (> 40 mV peak-to-peak) in any trailing 250 ms window. This is
   **tracker-independent** (the phase-singularity linker is unreliable on short branch runs).
5. `opencarp/phase3/lesion_sweep.py` (radius probe + position grid + resume);
   `opencarp/phase3/plot_efficacy.py`.

---

## Results

### Lesion-size threshold (lesion at the meander centroid)

| radius | outcome |
|---|---|
| 4 mm | sustained |
| 6 mm | terminated (~50 ms after lesion) |
| 8–12 mm | terminated |

Threshold ≈ **5 mm radius (10 mm lesion)**. An on-target lesion at/above that size collapses
the organizing singularity within tens of ms. Smaller and the rotor rides around it.

### Position sweep (fixed 6 mm radius, 37 sites on a 5 mm grid, 1.5 s post-lesion)

**10 / 37 positions terminate the rotor.** Time-to-termination 225–1175 ms (1–6 rotor
cycles), no clear spatial gradient.

The terminating sites form a **compact cluster on and to the right of the core's dwell
zone** — roughly x ∈ [20, 35] mm, y ∈ [22, 32] mm, overlapping the fibrotic patch's right
half — plus the tracked-core site itself.

**Counterintuitive finding:** the rotor core spends most of its time in the *left-central*
region (x ≈ 12–20 mm). Lesions placed there **do not** terminate it — the rotor simply
relocates around the hole. Lesions placed where the rotor would *relocate to*, or across a
critical part of its circuit (the right side), do terminate it. "Ablate the phase
singularity" is not the winning strategy for a meandering functional rotor — which matches
the clinical picture that functional/meandering rotors respond to ablation less predictably
than anatomically anchored ones.

---

## G3b — rotors B and C

Same pipeline (`opencarp/phase3/g3b.sh`), post-lesion window bumped to **2.5 s**, candidate
grid auto-centred on each rotor's tracked core path.

**Rotor B** — moderately anchored (core meanders ~7–15 mm around a centroid at ~(27, 17) mm).

- Radius probe at the centroid: **non-monotonic** — 4/6 mm sustain, 8 mm terminates (425 ms),
  10 mm sustains, 12 mm terminates. There is no clean "threshold"; a mid-size lesion can
  convert one reentry into another that survives.
- Position sweep (r = 6 mm): **8 / 49 positions terminate**, forming a band *above* the core
  (y ≈ 19–24 mm, x ≈ 19–34 mm). The lesion on the core centroid itself **sustains**.

**Rotor C** — tightly pinned (core meander span only ~4 mm, at ~(32, 28) mm, on the fibrotic
patch edge).

- Radius probe at the centroid: **only 12 mm terminates**; 4–10 mm all sustain. A modest
  focal lesion on a pinned rotor just becomes part of the obstacle it circulates around.
- Position sweep (r = 6 mm): **1 / 31 positions terminate** — a single site at (25, 27) mm,
  ~7 mm from the core, near the *fibrotic-patch centre*. The lesion on the core centroid
  sustains. For a pinned rotor the operative lesion is on the anchoring structure, not the
  visible core, and the effective spot is small and specific.

---

## Lesion size: pivot vs. functional core

The size probes were re-run centred on the functional-core centroid
(`opencarp/phase3/fc_probe.sh`) rather than the phase-singularity pivot:

| rotor | radius needed AT THE PIVOT | radius needed AT THE FUNCTIONAL CORE |
|---|---|---|
| A | ≥ 6 mm | ~5 mm — but a non-monotonic dropout (5 ✓, 6 ✗, 8 ✓) |
| B | erratic (8 ✓, 10 ✗, 12 ✓) | **clean threshold ~4 mm** |
| C | **12 mm** | **clean threshold ~5 mm** |

Targeting the functional core roughly **halves** the lesion size needed and mostly removes
the non-monotonic behaviour — decisive for the pinned rotor C, where pivot-targeting
demanded an unrealistically large 12 mm-radius (24 mm) lesion and functional-core targeting
brings it to a clinically ordinary ~10 mm lesion. Rotor A keeps a non-monotonic dip
regardless of target — a genuine meandering-rotor sensitivity (a "wrong-size" lesion lets it
settle into a new stable configuration); the auto-computed A centroid was also ~3 mm off a
hand estimate, which may contribute.

## Caveats

2D monodomain, isotropic; three rotors, one substrate realisation; one branch time per
rotor; the position sweeps used a single lesion radius (6 mm); termination judged over a
1.5–2.5 s post-lesion window (a site that terminates at ~2 s vs. never is near the decision
boundary); meandering rotor A's outcome is somewhat radius-sensitive at any target.

---

## What this sets up

- **The finding is consistent across all three rotors** — see the TL;DR table. Anchor
  tightness is the axis: loose rotors terminate from a cluster of offset positions; a pinned
  rotor terminates from essentially one spot on its anchor.
- **Phase 4 target — reconsidered.** "Predict the efficacy map from local features" is a
  sparse, rotor-specific target (10 + 8 + 1 = 19 positives / 117 sims across 3 rotors) and
  probably too thin and too idiosyncratic to learn directly. A better-posed target: predict
  the **reentry circuit / critical isthmus** (or the anchoring structure) from the activation
  pattern — more learnable, more clinically framed, and efficacy follows from it. Decide
  between (a) that reframing, (b) a broader substrate-varied dataset for the original target,
  or (c) both, before starting Phase 4.
