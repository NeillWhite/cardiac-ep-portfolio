# Phase 2: openCARP Simulation Setup

Runs on your own machine (needs Docker). Verified against the live docs at
https://opencarp.org/download/installation/install-docker and
https://opencarp.org/documentation/examples on 2026-08-18 — the scaffold's original
guesses (`opencarp/opencarp` on Docker Hub, generic tutorial paths) were wrong; corrected
below.

## 1. Install via Docker

```bash
docker pull docker.opencarp.org/opencarp/opencarp:latest
docker run -it -v $(pwd):/shared docker.opencarp.org/opencarp/opencarp:latest
```

Note the registry: it's `docker.opencarp.org`, not Docker Hub. The image includes
openCARP and `carputils`; **meshalyzer is not included** and needs a separate local
install if you want its GUI for visualization (not required — the examples below produce
PNG/GIF output via carputils' own plotting, and IGB files can be read directly in Python).

Once inside the container, examples live at `/openCARP/examples`.

## 2. Reentry induction example — the actual ground-truth source

`/openCARP/examples/02_EP_tissue/21_reentry_induction/run.py` is a near-exact match for
what the plan (`../docs/IMPLEMENTATION_PLAN.md` §4.1) asks for, already built and maintained
upstream rather than something to write from scratch:

- **Tissue:** 2D patch, 5cm × 5cm, ~0.4mm average edge length
- **Substrate:** a circular fibrotic region (radius 1.42cm) at the center — 30%
  non-conductive elements, 70% with rescaled ionic conductances (Courtemanche model +
  AF remodeling, 0.3 m/s conduction velocity) — this *is* the "low-conductivity patch"
  from the plan
- **Induction protocols** (the plan's "S1-S2 cross-field stimulation" maps to one of
  several offered here — RP_E/RP_B are the closest fit, PEERP and PSD are alternatives):
  - `prepace` — fixed-BCL stimulation to reach steady state before induction
  - `RP_E` / `RP_B` — decreasing-coupling-interval rapid pacing trains (RP_E checks for
    arrhythmia only at the end, RP_B after each beat)
  - `PSD` — manual phase-singularity placement via Eikonal initialization
  - `PEERP` — pacing timed to the effective refractory period for ectopic-beat induction

Example invocations:
```bash
./run.py --np 2 --protocol prepace --prepace_bcl 500 --prebeats 4 --visualize
./run.py --np 2 --protocol RP_E --start_bcl 200 --end_bcl 130 --max_n_beats_RP 1 --visualize
./run.py --np 2 --protocol PEERP --max_n_beats_PEERP 2 --visualize
```

Output: activation maps, transmembrane voltage animations, phase maps, and last-activation-
time (LAT) maps, i.e. most of the raw material Phase 2 needs already gets produced by the
example — the new work is the Hilbert-transform/phase-singularity-tracking-to-ablation-
label pipeline (plan §4.1 steps 3-6), which openCARP does not provide natively.

**Next step before writing any of that new code:** run this example inside the container,
confirm it actually sustains a reentrant rotor (not guaranteed on the first parameter
choice — flagged as a known risk in the plan, §9), and inspect what `run.py` actually
writes to disk (exact IGB paths, mesh files) before building the post-processing pipeline
against it.

## 3. Basic tissue EP example (simpler reference, not the primary path)

`/openCARP/examples/02_EP_tissue/01_basic_usage/run.py` — single stimulus, thin monolayer,
no reentry. Useful as a sanity check that the container/environment works before touching
the reentry example, not as the actual ground-truth source.

```bash
./run.py --duration 20 --S1-strength 20. --S1-dur 15 --visualize
```

## 4. What to extract for the ML side

Once a working reentry simulation exists:

- **Transmembrane voltage over time** at each mesh point — input to the Hilbert
  transform / phase singularity detection (new code, plan §4.1 steps 3-6)
- **Activation times** at each mesh point
- **Unipolar/bipolar electrograms** at virtual electrode locations — the signals directly
  analogous to what a real mapping catheter records, and the actual ML input (plan §4.2)
- **Conduction velocity fields** — useful for identifying slow-conduction zones

## 5. Toward Phase 3 (real-time optimization)

Once there's a working simulation → feature → prediction pipeline, profile it (usually the
PDE solve dominates) and train a fast ML surrogate — see main plan §5.

## 6. Toward the PFA angle

Optional, lowest priority — see main plan §6. Search terms if/when this is picked up:
"irreversible electroporation cardiac tissue model", "pulsed field ablation lesion
simulation", "electroporation threshold myocardium".
