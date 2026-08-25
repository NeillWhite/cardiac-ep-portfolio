# Phase 2 Methodology, Explained in Full

This document exists so you can explain this project to someone else without me in the room.
It walks through every stage of the Phase 2 pipeline twice: first in plain words (what are we
doing, and why), then in the actual code (how it's implemented). Code excerpts are pulled
directly from the real files in `opencarp/` — nothing here is simplified or fabricated for
the writeup.

If you only read one section carefully, read §3 (the Hilbert transform) and §4 (phase
singularities) — those two ideas are the intellectual core of the whole ground-truth
approach, and everything downstream (electrodes, classifier, sweeps) is "how do we get
useful information out of the thing §3-4 gave us, cheaply."

---

## 1. The problem, stated plainly

We simulated a small patch of heart tissue and made it go into a specific kind of
arrhythmia: a **rotor** — a spiral wave of electrical activation that spins in place instead
of dying out, the mechanism believed to sustain atrial fibrillation. Clinically, if you can
find where a rotor's *core* is (the point it spins around), ablating that spot can terminate
the arrhythmia. The problem: finding the core requires seeing the *whole* electrical picture
of the chamber over time, which real catheters can't cheaply do everywhere at once.

So Phase 2 has two halves:

1. **Ground truth.** Given the full simulated data (which we *do* have, because it's a
   simulation), compute exactly where the rotor's core is at every moment. This becomes the
   label: "this point in tissue is near the core" / "this point isn't."
2. **The actual ML question.** Pretend you *don't* have the full picture — you only have a
   handful of electrodes recording locally. Can you predict "near the core" from just their
   local signals? That's the thing that would actually be useful at the bedside, because it
   doesn't require the expensive full-field computation from step 1.

Everything below is: how step 1 works, then how step 2 tries to get away without it.

---

## 2. Inducing the rotor (not our code — openCARP's)

Before any of our own code runs, openCARP's bundled `21_reentry_induction` example (not
something we wrote) sets up a 5cm×5cm sheet of tissue with a circular patch of
low-conductivity "fibrotic" tissue in the middle, then fires a train of pacing stimuli at one
site, each one closer together in time than the last (basic cycle length, or **BCL**, 200ms
down to 100ms in 10ms steps). The idea: a normal beat, given enough recovery time, propagates
as a smooth expanding circle. A beat fired *too soon* catches part of the tissue still
"refractory" (not yet recovered from the last beat) — that part can't conduct, so the wave
splits, and the piece that can still propagate curls around the blocked region instead of
just continuing outward. If the timing and the tissue geometry line up right, that curl
never straightens back out — it becomes a permanent, self-sustaining spiral. See
`opencarp/README.md` for exactly which stimulus succeeded and why the first two protocols we
tried (`RP_E`, `PEERP`) crashed on a stale bundled checkpoint before we found the fix.

This step's output, for our purposes, is one file: `vm.igb`, the transmembrane voltage
(`Vm`, in millivolts) at all 15,876 mesh points, once per millisecond, for the ~400ms after
the triggering stimulus. That's the raw material everything else works from.

---

## 3. The Hilbert transform — turning voltage into an angle

### In words

Pick any single one of those 15,876 points and just look at its `Vm` over time. During a
sustained rotor, that trace oscillates: it swings up to about +20mV (depolarized, "firing")
each time the spinning wave sweeps past that point, then back down to about -80mV (resting)
until the wave comes around again.

Here's the problem with using that raw number directly: **voltage alone doesn't tell you
"how far around its cycle" that point currently is.** Imagine a Ferris wheel, and you're only
told the *height* of one seat. A seat at the "10 o'clock" position and a seat at "2 o'clock"
can be at nearly the same height — height alone is ambiguous about position. To know the true
angle around the wheel, you need a second number, 90° out of phase with height (say, the
seat's horizontal position), and then `angle = atan2(horizontal, height)` gives you the exact
angle.

The **Hilbert transform** is a standard signal-processing operation that manufactures exactly
that missing second number — directly from the one real signal you already have, no second
measurement needed. Feed it a real oscillating signal, and it hands back a new signal that is
the original one, shifted 90° in phase (a sine becomes a cosine, roughly speaking). Pair the
original and its Hilbert-shifted twin together as a complex number
`z(t) = Vm(t) + i · Hilbert(Vm)(t)`, and the angle of that complex number,
`phase(t) = atan2(Hilbert(Vm)(t), Vm(t))`, is a genuine "position in the cycle" — a number
that sweeps smoothly from -π to +π once per oscillation, exactly analogous to the Ferris
wheel's true angle.

Do this independently at all 15,876 mesh points, and the "movie" of raw voltage becomes a
"movie" of phase angle — same spatial layout, same timesteps, but every number is now
"how far around its own local cycle is this point right now," not "what's its voltage."

### In code

From `opencarp/phase_singularity.py`:

```python
def compute_phase(vm, detrend=True):
    """vm: (n_nodes, n_time) -> phase: (n_nodes, n_time) in (-pi, pi]."""
    x = vm - vm.mean(axis=1, keepdims=True) if detrend else vm
    analytic = hilbert(x, axis=1)
    return np.angle(analytic)
```

- `vm` is the full array loaded from `vm.igb`: one row per mesh node (15,876 of them), one
  column per millisecond (401 of them).
- `x = vm - vm.mean(...)`: subtract each node's own average voltage first. The Hilbert
  transform assumes the signal oscillates *around* zero; a signal that's mostly sitting at
  -80mV with occasional spikes to +20mV isn't centered, so we center it per-node before
  transforming. (`detrend=True` is the default and what we actually use.)
- `hilbert(x, axis=1)` is `scipy.signal.hilbert`. Despite the name, it doesn't return "the
  Hilbert transform" by itself — it returns the whole complex **analytic signal**
  `x(t) + i·Hilbert(x)(t)` in one call, `axis=1` telling it to do this along the time axis
  independently for every row (every mesh node) at once.
- `np.angle(analytic)` takes that complex number at every (node, timestep) and returns its
  angle — exactly the `atan2` step from the Ferris-wheel analogy, vectorized over the whole
  mesh and the whole time window in one call.

The result, `phase`, is the same shape as `vm` (15,876 × 401), but every entry is now an
angle in `(-π, π]` instead of a voltage in mV.

---

## 4. Phase singularities — finding the actual pivot point

### In words

Now for the payoff of having phase instead of voltage: **a rotor's core has a specific,
clean mathematical signature in the phase field, and nowhere else does.**

Picture standing at some point in the tissue, away from the core, and walking in a tiny
circle around your starting point, checking the phase at each step. Elsewhere in a smoothly
propagating wave, phase changes gradually and consistently as you walk — and by the time you
get back to where you started, the phase has returned to very close to its starting value
(net change ≈ 0). Nothing surprising.

Now do the same tiny walk, but centered *on the rotor's core*. Because every direction around
the core is, in a sense, "a different point in the cycle" (the spiral wraps all the way
around it), walking one full loop around the core means the phase you measure sweeps through
its *entire range* — all the way from -π to +π — exactly once, before you return to your
starting point. That's not a gradual local change like everywhere else; it's a full 360°
"winding" packed into an infinitesimally small loop. That winding is the phase singularity,
and mathematically it can only happen at the exact point the spiral pivots around — the
core.

This is precisely why we needed *phase* rather than raw voltage for this step: winding only
makes sense for a genuinely cyclic quantity (something that "wraps around," like an angle).
Voltage doesn't wrap around — there's no meaningful sense in which -80mV connects back to
+20mV the way -π connects to +π. Phase, by construction from the Hilbert transform, does
wrap around cleanly, which is what makes the winding-number test below well-defined and
robust.

### In code

```python
def wrap(phase_diff):
    return (phase_diff + np.pi) % (2 * np.pi) - np.pi


def detect_ps_frame(phase_grid, charge_tol=0.15, border=2):
    """phase_grid: (ny, nx) phase at one instant. Returns list of (ix, iy, charge)
    for unit cells whose corner phases wind by ~+-2*pi (a phase singularity),
    excluding a border margin to avoid mesh-edge artifacts."""
    p00 = phase_grid[:-1, :-1]
    p01 = phase_grid[:-1, 1:]
    p11 = phase_grid[1:, 1:]
    p10 = phase_grid[1:, :-1]

    loop = wrap(p01 - p00) + wrap(p11 - p01) + wrap(p10 - p11) + wrap(p00 - p10)
    charge = loop / (2 * np.pi)

    ny_cells, nx_cells = charge.shape
    hits = []
    for iy in range(border, ny_cells - border):
        for ix in range(border, nx_cells - border):
            c = charge[iy, ix]
            if abs(abs(c) - 1.0) < charge_tol:
                hits.append((ix, iy, np.sign(c)))
    return hits
```

Since the mesh is a regular 126×126 grid, "walk a tiny loop" becomes concrete: take every
elementary 2×2 block of neighboring mesh points (`p00, p01, p11, p10` — the four corners of
one small square, visited in order going around it), and sum up the phase change from corner
to corner around that square:

- `wrap(p01 - p00)`: phase change from bottom-left to bottom-right corner.
- `wrap(p11 - p01)`: bottom-right to top-right.
- `wrap(p10 - p11)`: top-right to top-left.
- `wrap(p00 - p10)`: top-left back to bottom-left, closing the loop.

Each individual step's phase difference could naively be, say, `3.0 - (-3.0) = 6.0`, which is
actually a *small* real change (just past the ±π wraparound point) — so `wrap()` first forces
every step's difference back into `(-π, π]` before adding it in, the same "roll over" logic
a clock uses (11 o'clock to 1 o'clock is a 2-hour step, not a 10-hour one). Sum the four
(correctly wrapped) steps, divide by 2π, and you get the **topological charge**: a number
that should come out to almost exactly 0 for an ordinary square (phase returns to where it
started), or almost exactly ±1 for a square that happens to contain a phase singularity
(one full winding, direction — clockwise or counterclockwise — given by the sign).
`charge_tol=0.15` allows for the fact that the mesh is discrete, not perfectly continuous, so
the winding won't compute out to *exactly* 1.0 in practice. `border=2` just skips checking
squares right at the mesh edge, where boundary effects can produce spurious windings that
aren't real singularities.

Run this once per millisecond (401 times) and you get, for each frame, the list of small
squares that currently contain a phase singularity — i.e., where the rotor's core is *right
now*.

---

## 5. Tracking the core's path over time

### In words

The core doesn't sit still — real (and simulated) rotors "meander," tracing a small loop or
rosette pattern rather than pinning to one exact spot, especially near a substrate feature
like our fibrotic patch edge. Frame-by-frame detection (§4) gives us, at most, one core
location per millisecond; stitching those 401 detections into one continuous path just means
picking, at each frame, whichever detected core position is closest to where the core was in
the *previous* frame (since it can't have physically teleported). If a frame's nearest
candidate is implausibly far from the last known position, we skip that frame rather than
draw a nonsensical jump.

### In code

```python
def track_trajectory(ps_by_frame, xs, ys, max_jump_um=3000.0):
    """Greedy nearest-neighbour linking of one PS per frame into a single trajectory."""
    traj = []
    prev_xy = None
    for frame_idx, hits in ps_by_frame:
        if not hits:
            continue
        candidates = [
            (0.5 * (xs[ix] + xs[ix + 1]), 0.5 * (ys[iy] + ys[iy + 1]), charge)
            for (ix, iy, charge) in hits
        ]
        if prev_xy is None:
            x, y, charge = candidates[0]
        else:
            dists = [np.hypot(x - prev_xy[0], y - prev_xy[1]) for (x, y, _) in candidates]
            best = int(np.argmin(dists))
            if dists[best] > max_jump_um:
                continue  # discontinuity -- skip rather than jump implausibly far
            x, y, charge = candidates[best]
        traj.append({"frame": frame_idx, "x": x, "y": y, "charge": charge})
        prev_xy = (x, y)
    return traj
```

Each detected square from §4 gets converted to a physical (x, y) coordinate (the midpoint of
its four corners). The very first frame with any detection just takes the first candidate
(nothing to compare against yet); every frame after that measures the distance from every
candidate in the current frame to the *previous* accepted position, and keeps the closest one
— as long as it's within `max_jump_um` (3mm) of the last position. Both of our rotors turned
out to have exactly one detection per frame throughout (401/401), so this greedy linking
barely had to do any real work — but it's what makes the code robust to messier cases (a
different substrate, a noisier signal) where more than one candidate per frame is common.

---

## 6. Labeling ablation targets

### In words

This is the actual ground-truth label the classifier will later try to predict: given the
full trajectory the core traced out over the whole 400ms window, any mesh point that ever
came within a small radius (3mm — a simplification standing in for a real clinically-derived
lesion-size number) of *any* point on that trajectory is labeled an ablation target. Every
other point is background. Clinically, this corresponds to: "if you knew the rotor's full
path, ablating the region it swept through should stop it."

### In code

```python
def label_ablation_targets(pts, traj, radius_um=3000.0):
    if not traj:
        return np.zeros(pts.shape[0], dtype=bool)
    traj_xy = np.array([[p["x"], p["y"]] for p in traj])
    labels = np.zeros(pts.shape[0], dtype=bool)
    for i, (x, y, _) in enumerate(pts):
        d = np.hypot(traj_xy[:, 0] - x, traj_xy[:, 1] - y)
        labels[i] = d.min() <= radius_um
    return labels
```

For every one of the 15,876 mesh points, compute its distance to *every* point on the
trajectory, take the minimum (distance to the nearest point the core ever visited), and label
it `True` if that minimum is within 3mm. Straightforward, if a little brute-force (15,876
points × ~400 trajectory points — still fast, a fraction of a second).

This is the point where it's worth being explicit about something important: **using the
full phase map (§3-4) to predict this label would be circular.** The label *is* "distance to
the phase-singularity trajectory" — if you fed that same distance back in as a predictor,
you'd score a perfect, meaningless 1.0, because you'd be using the label's own definition to
predict itself. The entire motivation for what comes next (§7 onward) is that computing the
phase map requires simultaneous dense recording across the *whole* mesh, which a real
catheter doesn't have — so the actual, non-trivial question is whether a handful of *local*
electrode measurements can stand in for that expensive global computation.

---

## 7. Virtual electrodes and the "unipolar EGM" approximation

### In words

A real catheter doesn't read transmembrane voltage (`Vm`) directly — `Vm` is the voltage
*across the cell membrane*, not measurable from outside the cell. What a catheter actually
measures is **extracellular potential** (`phi_e`): the voltage in the fluid just outside the
cells, which is what genuinely "unipolar" and "bipolar" electrograms are built from.

The heart's tissue has two physically coupled electrical compartments: the *intracellular*
space (inside cells, connected cell-to-cell via gap junctions) and the *extracellular* space
(the fluid bathing them). As a cell depolarizes, current crosses the membrane between the two
and flows back through the extracellular fluid to complete the circuit. There are two ways to
get a genuine `phi_e` out of a simulation:

1. **Run a bidomain simulation.** This solves for both compartments explicitly — two coupled
   equations tracking intracellular potential and `phi_e` as *separate* quantities everywhere
   in the tissue (and any surrounding bath/blood pool). `phi_e` is directly in the output, so
   you can read off "what would an electrode at this exact spot measure."
2. **Run monodomain (cheaper — what we did), then a separate forward-model calculation** on
   top of its `Vm` output (a "pseudo-ECG"/lead-field integral) to compute what `phi_e` *would
   have been* at any point, without rerunning the main simulation.

Monodomain is a mathematical shortcut valid under a simplifying assumption (intracellular and
extracellular conductivity proportional to each other); under that assumption, the two
coupled bidomain equations collapse into *one* equation for `Vm` — the *difference* between
the two potentials, not either one individually. That's cheaper to solve (the reason it's the
common default, including in the openCARP example we used — its `parameters.par` explicitly
sets `bidomain = 0`), but the cost is real: a monodomain simulation's output simply never
contains `phi_e`. It isn't hidden in there waiting to be extracted; the math only ever tracked
the difference.

**We did neither of the two genuine options above.** We took a cheaper shortcut than both:
**approximate the "unipolar EGM" at a site as just the local `Vm(t)`** — not a derivation of
extracellular potential by any method, just reusing the transmembrane voltage number directly
as a stand-in. This is a real, named limitation (documented in the README), not something to
gloss over.

One more thing worth being precise about, since it's easy to overstate: the **bipolar EGM**
computation itself — `bipolar(t) = unipolar_A(t) - unipolar_B(t)` — genuinely *is* the
standard, correct way real bipolar electrograms are always derived from two unipolar ones,
regardless of what the unipolar inputs actually are. But that procedure is only as good as
its inputs, and ours are the `Vm`-as-stand-in shortcut above, not true `phi_e` — so the
resulting bipolar signal inherits that same approximation. It doesn't escape it just because
the subtraction step is standard.

### In code

From `opencarp/extract_cluster_features.py`:

```python
def bipolar_amplitude(unipolar_a, unipolar_b):
    bp = unipolar_a - unipolar_b
    return float(bp.max() - bp.min()), bp
```

`unipolar_a` and `unipolar_b` are each just a row of the `vm` array — the raw `Vm(t)` trace
at one mesh node, treated as that site's unipolar EGM. Their difference is the bipolar signal;
its peak-to-peak range (`max - min`) is the bipolar amplitude, the simplest and most
clinically standard EGM feature (low amplitude is a classic marker of scar/fibrosis or a
rotor core's "voltage dropout").

---

## 8. Electrogram features: fractionation, LAT, dominant frequency

### In words

Beyond amplitude, three more standard EP signal features, each capturing a different aspect
of "what does this site's local activity look like":

- **Fractionation**: how many separate up-and-down deflections does the bipolar signal have,
  above some noise floor? A clean single beat has one clear deflection; a site with
  chaotic, colliding wavefronts (as you'd expect very close to a rotor core, where wavefronts
  arrive from multiple directions) tends to show several smaller, ragged deflections instead
  of one clean one.
- **Local activation time (LAT)**: when, within the recorded window, did this site actually
  "fire"? The standard definition is the moment of the steepest downstroke (most negative
  `dVm/dt`) — the sharpest, fastest part of the depolarization.
- **Dominant frequency (DF)**: treat the signal as approximately periodic (each rotor
  rotation looks similar) and ask: what's the strongest repeating frequency, via a Fourier
  transform? For our two rotors (BCL ≈150-200ms per rotation), that's roughly 5-6.7 Hz.

One real wrinkle we hit and fixed: naively taking "the frequency with the biggest FFT peak"
sometimes picked frequencies over 400Hz — nonsense for a 150-200ms rotor. The cause: a signal
with one or two sharp, mostly non-repeating deflections (not a clean periodic oscillation)
has a *broadband* spectrum dominated by the sharp edges themselves, not by any genuine
periodicity. The standard fix in the AF-mapping literature is to restrict the search to a
physiologically plausible band (we used 3-15Hz) before picking the peak — which is exactly
what real dominant-frequency analysis tools do, not a shortcut invented for this project.

### In code

```python
FRACTIONATION_DERIV_THRESHOLD_FRAC = 0.05
DF_BAND_HZ = (3.0, 15.0)

def fractionation(bp):
    thresh = FRACTIONATION_DERIV_THRESHOLD_FRAC * (bp.max() - bp.min() + 1e-9)
    d = np.diff(bp)
    signs = np.sign(d)
    signs[np.abs(d) < thresh] = 0
    nz = signs[signs != 0]
    return int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0


def dominant_freq(bp, fs_hz):
    n = len(bp)
    spectrum = np.abs(np.fft.rfft(bp - bp.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    band = (freqs >= DF_BAND_HZ[0]) & (freqs <= DF_BAND_HZ[1])
    return float(freqs[band][np.argmax(spectrum[band])]) if band.any() else 0.0
```

- **`fractionation`**: `d = np.diff(bp)` is the signal's slope at every timestep.
  `signs = np.sign(d)` turns each slope into +1 (rising), -1 (falling), or would be 0 (flat).
  Any slope too small to matter (below 5% of the signal's own range) gets forced to exactly 0
  and dropped (`signs[signs != 0]` filters it out) — this is the noise floor, so tiny wiggles
  don't count as real deflections. `np.diff(nz) != 0` then counts how many times the
  direction actually flips (rising-to-falling or vice versa) among the remaining real
  movements — each flip is one "deflection."
- **LAT** (computed in `opencarp/extract_cluster_features.py`'s main loop, not shown as a
  standalone function): `lat_idx = np.argmin(np.diff(unipolar))` — the timestep where the
  slope is *most negative*, i.e. the steepest downstroke, converted to milliseconds.
- **`dominant_freq`**: `np.fft.rfft` computes the frequency spectrum of the (mean-subtracted)
  bipolar signal; `np.fft.rfftfreq` gives the actual Hz value for each entry of that
  spectrum. `band = ...` masks out everything outside 3-15Hz, and `np.argmax(spectrum[band])`
  finds the strongest frequency *within* that restricted range only.

---

## 9. Why single-electrode features weren't enough

### In words

The first, simplest version of this pipeline (`opencarp/extract_electrode_features.py`) used
exactly one fixed-direction bipolar pair per candidate site (always paired with the neighbor
2mm to its east) and the four features from §7-8 directly. Result: **ROC-AUC 0.61**, barely
better than a coin flip, and per-class feature averages (ablation-target sites vs.
background) were nearly identical.

The diagnosis, and it's the single most important conceptual point in this whole writeup:
**a phase singularity is fundamentally a *relational* concept — defined by how phase changes
across a small loop of *neighboring* points (§4) — not a property visible in any one point's
own isolated signal.** A single electrode's waveform shape only indirectly, weakly reflects
that it happens to sit near such a loop. Asking one electrode's own amplitude/fractionation/
frequency to reveal "am I near a topological winding among my neighbors" is asking a
fundamentally local, single-point measurement to answer a fundamentally multi-point,
relational question. That mismatch — not a coding bug — is why the first attempt was weak.

### In code

The fix, in `opencarp/extract_cluster_features.py`: give every candidate site a small
*cluster* of nearby electrodes (2mm spacing, added one direction at a time — east, north,
west, south, then the four diagonals, up to 8 total, mimicking a small grid/basket mapping
catheter) and compute features that are explicitly *relational* across that cluster, not just
per-point:

```python
DIRECTIONS = [
    (OFFSET_NODES, 0), (0, OFFSET_NODES), (-OFFSET_NODES, 0), (0, -OFFSET_NODES),
    (OFFSET_NODES, OFFSET_NODES), (-OFFSET_NODES, OFFSET_NODES),
    (-OFFSET_NODES, -OFFSET_NODES), (OFFSET_NODES, -OFFSET_NODES),
]
```
```python
lats_arr = np.array(lats)   # this candidate's LAT plus every neighbor's LAT so far
rows.append({
    ...
    "bipolar_amp_min": float(np.min(bp_amps)),
    "bipolar_amp_mean": float(np.mean(bp_amps)),
    "bipolar_amp_std": float(np.std(bp_amps)),
    "fractionation_mean": float(np.mean(fracs)),
    "fractionation_max": float(np.max(fracs)),
    "lat_spread_ms": float(lats_arr.max() - lats_arr.min()),
    "lat_std_ms": float(lats_arr.std()),
    "dom_freq_mean": float(np.mean(dfs)),
    "dom_freq_std": float(np.std(dfs)),
    ...
})
```

`lat_spread_ms` was *intended* as the key relational feature: max LAT minus min LAT *among
this candidate site and its neighbors*, on the theory that a smoothly propagating wavefront
activates a small local cluster at almost the same time (small spread), while a site near a
wavebreak/core sees genuinely inconsistent local timing (large spread). **That theory turned
out to be substantially wrong in practice — see §9a below**, where hands-on exploration in
`opencarp/feature_exploration.ipynb` found the actual top features are `bipolar_amp_std` and
`dom_freq_std`/`dom_freq_mean` (cluster *variability* in amplitude and frequency, not LAT
timing). The `_std`/`_min`/`_mean` variants of amplitude, fractionation, and frequency share
the same underlying idea — not "what does one point look like," but "how much does this
small neighborhood *disagree with itself*" — and that part of the hypothesis held up; the LAT
half specifically didn't, for a concrete, diagnosable reason.

This is computed once per candidate site *for every k from 1 to 8* (adding one more neighbor
each time), which is what makes the electrode-count sweep (§11) possible — same feature
*definitions*, just built from a growing amount of local spatial information.

### 9a. Correction, found via hands-on exploration: `lat_spread_ms` doesn't actually separate the classes

Opening the raw signals up in a notebook (exactly the kind of check a data-science-first
approach catches that a "the AUC looked fine" pass doesn't) found this directly: **median
`lat_spread_ms` is 221ms for ablation-target sites and 219.5ms for background sites** —
essentially identical. The permutation-importance chart
(`results/phase2_reentry_2026-08-20/classifier_results.png`) already showed this, correctly —
`lat_spread_ms` and `lat_std_ms` sit near zero importance, `bipolar_amp_std` dominates — but
an earlier draft of this document claimed the opposite, misattributing the "biggest single
win" to the wrong feature without checking the chart it cited. That's now fixed here.

**Why the LAT-spread idea fails in practice, diagnosed concretely:** `compute_lat` (§8) finds
the single steepest downstroke *anywhere in the whole 400ms window* — but that window
contains roughly 2-3 separate beats (rotor cycle length ≈150-200ms). Nothing forces
neighboring sites to pick the *same* beat as each other's steepest one; cycle-to-cycle
variation in upstroke sharpness means two sites 2mm apart can easily each have a different
beat register as "their" steepest downstroke, producing an apparent spread on the order of an
entire cycle length (100-200+ms) — regardless of whether those two sites are actually near a
core or not. Sampling 60 random background clusters confirmed this isn't a rare edge case:
**55/60 (92%) had a spread over 100ms**, with the same right-skewed, cycle-length-scale
distribution as the ablation-target sites. The feature isn't measuring "local timing
disagreement near a core" at all, for the most part — it's measuring "which of ~2-3 candidate
beats each site's independent argmin happened to land on," which is close to a coin flip
almost everywhere.

This doesn't undermine the *cluster-vs-single-site* finding overall — `bipolar_amp_std` and
`dom_freq_std`/`dom_freq_mean` are still cluster-relational features in the same family, and
they're the ones actually carrying the signal in the electrode-count sweep (§11) and the
cross-rotor validation (§12). It does mean the specific "wavebreak → inconsistent timing"
story for *why* clustering helps needs a caveat: the amplitude- and frequency-based cluster
features are doing the real work; the LAT-spread mechanism, as currently computed, isn't. A
concrete, scoped fix — restricting LAT to a single reference beat instead of a global argmin
over the whole multi-beat window — is left as a scaffolded exercise in the notebook (§5,
`local_conduction_velocity`) rather than done here, since it changes a core function used
throughout the pipeline and deserves its own dedicated validation pass before trusting it.

---

## 10. The classifier

### In words

A gradient-boosted tree ensemble (`sklearn.ensemble.HistGradientBoostingClassifier`) — a
sequence of small decision trees, each one trained to correct the previous ones' mistakes.
Deliberately not a deep model on raw waveforms, per the project's stated philosophy: with
only ~9 hand-engineered, physically meaningful features, a small tree ensemble is both
plenty expressive enough and — crucially — the kind of model whose feature importances a
domain expert (or a data scientist willing to check the underlying signals, per §9a) can
actually sanity-check: `bipolar_amp_std` and `dom_freq_std`/`dom_freq_mean` dominate, and that
claim is verifiable against the raw traces, unlike "the deep net learned something."

Two practical issues had to be handled explicitly, both because ablation-target sites are
rare (~5% of all sites):

1. **Class imbalance**: without correction, a classifier can get 95% accuracy by just
   predicting "background" for everything. Every training example is weighted by the
   inverse of its class's frequency, so the rare positive examples count proportionally
   more during training.
2. **Spatial autocorrelation**: all the training data comes from one continuous simulated
   field — electrodes 2mm apart are highly correlated (very likely to share a label and have
   similar feature values). A plain random train/test split would let near-duplicate
   neighbors leak across the split, inflating the test score. Instead, the domain is divided
   into 5mm × 5mm blocks, and blocks are assigned to train or test in a checkerboard pattern
   — so no test point's immediate neighborhood is present in the training set.

### In code

```python
def spatial_split(df):
    bx = (df["x_mm"] // BLOCK_SIZE_MM).astype(int)
    by = (df["y_mm"] // BLOCK_SIZE_MM).astype(int)
    is_test = (bx + by) % 2 == 0
    return ~is_test, is_test
```
```python
pos_rate = y_train.mean()
sample_weight = np.where(y_train == 1, 1.0 / pos_rate, 1.0 / (1 - pos_rate))

clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
clf.fit(X_train, y_train, sample_weight=sample_weight)
```

`bx, by` bucket every site's (x, y) position into which 5mm block it falls in.
`(bx + by) % 2 == 0` is the checkerboard pattern (like the black/white squares on a chess
board, but at 5mm resolution) — adjacent blocks always land on opposite sides of the split.
`sample_weight` gives every positive (ablation-target) example a weight of
`1 / pos_rate` (large, since `pos_rate` is small — about 20× for a 5% positive rate) and
every negative example a weight of `1 / (1 - pos_rate)` (close to 1), so the training
objective doesn't just learn to always predict "background."

One more real issue, caught rather than glossed over: the default 0.5 probability threshold
(`clf.predict(...)`) gave spatially nonsensical predictions — scattered essentially randomly
across the whole domain — even though the underlying probabilities *ranked* sites sensibly
(good ROC-AUC). The aggressive reweighting above shifts probability calibration, so 0.5 isn't
a meaningful cutoff anymore. Fix: pick a threshold matched to the known prevalence instead —
flag the top ~5% of sites by predicted probability, which is also the practically relevant
framing anyway ("check the top-N riskiest sites first"):

```python
threshold = np.quantile(y_prob, 1 - pos_rate)
y_pred = (y_prob >= threshold).astype(int)
```

---

## 11. The electrode-count sweep

### In words

Your idea, and it turned out to be the single most useful experiment in Phase 2: instead of
picking one fixed cluster size, train and evaluate the *same* classifier separately at every
k from 1 to 8, and plot performance against k. This directly answers "how many local
electrodes does this task actually need?" — a question with real practical weight, since more
electrodes means a more elaborate (slower, more expensive) catheter.

Within one rotor, this gave a clean, encouraging climb: ROC-AUC 0.68 at k=1 up to 0.85 by
k=3, then a slower rise to 0.91 at k=8 — most of the achievable signal from just 3-4 local
electrodes, matching the "relational signature" story from §9 directly.

### In code

```python
for k in ks:
    df = full[full["k"] == k].reset_index(drop=True)
    aucs, aps = [], []
    for rep in range(args.n_repeats):
        # jitter the checkerboard phase per repeat for a rough stability estimate
        ...
        clf = HistGradientBoostingClassifier(...)
        clf.fit(X_train, y_train, sample_weight=sw)
        y_prob = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
        aps.append(average_precision_score(y_test, y_prob))
```

Same classifier, same spatial-split logic as §10, just looped over every value of k and
repeated 10 times with a randomly shifted checkerboard phase each time (since with only ~33
positive sites total, a single fixed split is a noisy point estimate — averaging over several
phase-shifted splits gives a more honest mean ± spread).

---

## 12. Cross-rotor validation — first a sobering result, then a more complete one

### In words

Everything in §9-11 was validated *within one simulated rotor* — the spatial checkerboard
split (§10) controls for nearby electrodes leaking across train/test, but every electrode,
train or test, still comes from the exact same single induced rotor. That leaves open a real
question: does any of this generalize to a *different* rotor, or did the classifier just
learn to interpolate within one specific instance's particular quirks?

To test this properly, we induced a second, genuinely independent rotor: same substrate, but
a stimulus fired from the mirror-opposite side of the tissue (a patched copy of openCARP's
example script, changing the hardcoded stimulus location). It worked — a second sustained
rotor, anchored on the opposite edge of the same fibrotic patch (physically sensible: the
wave now approaches from the other direction, so it breaks and curls on the near side instead
of the far side).

Then: train the classifier on rotor A's data, evaluate on rotor B's — and the reverse — at
every k. The result was **not** what the within-rotor sweep predicted. Instead of a clean
climb, cross-rotor performance peaked around **k=2 (mean ROC-AUC ≈0.80)** and then got
*noisier and generally worse* with more electrodes, dropping to **0.52 (chance level) at
k=6**. The likely explanation: the larger clusters' extra features let the model fit
rotor-A-specific idiosyncrasies (its particular meander shape and timing) that don't carry
over to rotor B, rather than learning genuinely transferable physiology. If anything, k=2
looks like the more honest, generalizable choice — small enough to resist overfitting to one
instance, still a real improvement over k=1.

This is exactly the kind of thing this project's documentation convention exists for:
report it plainly rather than quietly keep the more flattering within-rotor number. Two
rotors is itself a small sample for judging generalization, so this result is suggestive,
not final — which is part of why a third rotor is the natural next step.

### In code

```python
def fit_eval(train_df, test_df, k, seed):
    tr = train_df[train_df["k"] == k]
    te = test_df[test_df["k"] == k]
    X_train, y_train = tr[FEATURES].values, tr["ablation_target"].values.astype(int)
    X_test, y_test = te[FEATURES].values, te["ablation_target"].values.astype(int)
    ...
    clf.fit(X_train, y_train, sample_weight=sw)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), average_precision_score(y_test, y_prob)

...
auc_ab, ap_ab = fit_eval(df_a, df_b, k, args.seed)  # train A, test B
auc_ba, ap_ba = fit_eval(df_b, df_a, k, args.seed)  # train B, test A
```

Same `fit_eval` logic as §10-11, just called twice per k with the train/test roles swapped
between the two independently-induced rotors' full datasets (no within-rotor spatial split
needed here — the two rotors are already spatially/temporally independent of each other by
construction).

### A third rotor, and leave-one-out — the fuller picture

Two rotors gives you exactly one train/test pair per direction — not much to judge
generalization from. A third rotor (induced from below the fibrotic patch, the same
site-patching trick applied a second time) enables **leave-one-rotor-out** cross-validation:
train on the *pooled* data from two rotors, test on the third, rotate which one is held out,
and average.

```python
for held_out in names:
    train_df = pd.concat([df for n, df in rotors.items() if n != held_out], ignore_index=True)
    test_df = rotors[held_out]
    auc, ap = fit_eval(train_df, test_df, k, args.seed)
```

This told a meaningfully different story than the pairwise check: mean ROC-AUC stayed well
above chance at *every* k (0.70-0.84, versus the pairwise check's collapse to 0.52 at k=6),
with k=2 as a clear, consistent peak across all three individually-held-out rotors (0.87 /
0.86 / 0.79 — notably low spread, unlike the noisier larger-k folds). The interpretation:
pooling **two** independent rotors' worth of training data (rather than just one) gives the
larger, more overfit-prone cluster features (k=6-8) enough varied examples to avoid latching
onto any single rotor's idiosyncrasies — exactly the effect you'd expect from more training
diversity, and a good illustration of why "validate on a second held-out instance" and
"validate on enough held-out instances to pool a stable training set" can give genuinely
different answers. Three rotors is still a small sample, but it's a substantially more
trustworthy claim than either the single-rotor sweep or the two-rotor pairwise check alone.

---

## 13. Quick-reference cheat sheet

If you need to explain this in 60 seconds:

1. We made a simulated heart-tissue patch go into a rotor (spinning electrical wave) using a
   standard "pace it faster and faster until part of the tissue can't keep up" protocol.
2. To know *exactly* where the rotor's core is (our ground truth), we convert each point's
   voltage trace into a **phase** (via the Hilbert transform — a way to turn "how far along
   its own cycle is this point" into a real, measurable angle), then find the one spot where
   walking a tiny loop makes that angle sweep a full 360° — the mathematical signature of the
   spiral's pivot point.
3. The actual ML question is harder and more useful: can a **few local electrodes**, without
   that expensive full-field computation, predict "near the core"? Single electrodes
   couldn't (ROC-AUC 0.61) — the core's signature is inherently about *disagreement between
   neighboring points*, not any one point's own signal. In practice that showed up as
   cluster-level *amplitude and frequency* variability (`bipolar_amp_std`, `dom_freq_std`),
   not the activation-timing spread the theory predicted (§9a) — timing spread turned out to
   be swamped by a separate multi-beat measurement artifact, a good example of a plausible
   mechanism not surviving contact with the actual data.
4. Small local clusters of electrodes fixed that within one rotor (up to ROC-AUC 0.91), but a
   pairwise check against a **second, independently induced rotor** showed that gain doesn't
   fully hold up alone — performance peaked around 2 local electrodes and got noisier with
   more. A **third** rotor, and leave-one-rotor-out cross-validation (train on 2 pooled
   rotors, test on the third), told a fuller story: mean ROC-AUC stayed well above chance at
   every cluster size (0.70-0.84), with 2 local electrodes as a clear, low-variance peak
   across all three — pooling more independent training instances stabilizes the larger
   cluster sizes too, rather than them being fundamentally non-generalizable.
