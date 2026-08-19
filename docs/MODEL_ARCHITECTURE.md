# Phase 1 Model: Architecture & Design Rationale

Covers `scripts/model.py` (`ECGConvNet`) as trained and evaluated for Phase 1 — see
`README.md` §6 for the actual macro F1 / confusion matrix results this architecture
produced, and `IMPLEMENTATION_PLAN.md` §3 for how it fits the overall project. This document
answers a narrower question: *why this architecture, specifically*, and what it can't do as
a result.

---

## 1. Task framing

- **Input:** one 12-lead ECG recording, 10 seconds at 100Hz → tensor shape `(12, 1000)`
  (leads, time), per-lead z-score normalized at the `Dataset` level (`dataset.py`).
- **Output:** a 5-way softmax over PTB-XL diagnostic superclasses — NORM, MI, STTC, CD, HYP.
- **Framing:** single-label multiclass classification, not multi-label — see README §8 for
  why (multi-label records are dropped upstream, not modeled).

## 2. Architecture

```
Input (12, 1000)
   │
   ├─ Conv1d(12→32, k=7, pad=3)  →  BatchNorm1d(32)  →  ReLU  →  MaxPool1d(2)     (32, 500)
   │
   ├─ Conv1d(32→64, k=5, pad=2)  →  BatchNorm1d(64)  →  ReLU  →  MaxPool1d(2)     (64, 250)
   │
   ├─ Conv1d(64→128, k=5, pad=2) →  BatchNorm1d(128) →  ReLU  →  AdaptiveAvgPool1d(1)  (128, 1)
   │
   ├─ Flatten → Dropout(0.3) → Linear(128→5)
   │
Output (5,) logits
```

Verified layer-by-layer shape trace (input `(1, 12, 1000)`):

| Stage | Output shape | Params |
|---|---|---|
| Conv1d(12→32, k=7) + BN | (32, 1000) | 2,720 + 64 |
| MaxPool1d(2) | (32, 500) | 0 |
| Conv1d(32→64, k=5) + BN | (64, 500) | 10,304 + 128 |
| MaxPool1d(2) | (64, 250) | 0 |
| Conv1d(64→128, k=5) + BN | (128, 250) | 41,088 + 256 |
| AdaptiveAvgPool1d(1) | (128, 1) | 0 |
| Dropout(0.3) + Linear(128→5) | (5,) | 645 |
| **Total** | | **55,205** |

Measured on this server (2026-08-19), not estimated:

| | Latency |
|---|---|
| CPU, single sample | 0.339 ms |
| CPU, batch of 64 | 0.184 ms/sample |
| GPU (RTX 3070), single sample | 0.151 ms |

The README's original design requirement (§2: "< 50ms per 10s 12-lead ECG") was an
unverified aspirational target when written — it's now confirmed with roughly 100x headroom
on CPU alone. At 55K parameters this model is small enough that inference cost was never
going to be the bottleneck in this pipeline; simulation compute (Phase 2/3) is where latency
actually matters.

### Receptive field

The conv/pool stack alone (before the final pooling) has an effective receptive field of
**34 samples = 340ms at 100Hz** — computed via the standard RF recurrence
(`RF_out = RF_in + (k-1)·jump_in`, `jump_out = jump_in·stride`) through the three conv
layers and two intervening max-pools. 340ms roughly spans a QRS complex plus part of the
ST-T segment — the right order of magnitude for the kind of localized morphology that
distinguishes these superclasses (e.g. STTC is defined by ST/T-wave shape).

`AdaptiveAvgPool1d(1)` then averages that 34-sample-wide local pattern detector across all
250 remaining time steps — i.e. across the full 10-second recording. The practical
consequence: the model is a **local-pattern detector pooled globally**, not a beat-by-beat
or whole-cycle model. Each of the 128 output channels reports "how strongly did my ~340ms
pattern fire, averaged over the whole recording" — with no explicit notion of beat order,
heart rate, or where in the recording something happened. This is a real capability
boundary, not just a training artifact — see Limitations below.

## 3. Design decisions

**1D convolution over raw waveform, not RNN/Transformer.** A QRS complex looks similar
regardless of where it falls in the 10-second window — that's a translation-invariance
assumption 1D convs encode directly and cheaply, without needing a recurrent or attention
mechanism to learn it from data. Consistent with this project's stated preference
(`IMPLEMENTATION_PLAN.md` §4.2, applied to the Phase 2 model choice, but the reasoning
applies here too) for the simplest architecture that plausibly fits the task over a more
powerful one that's harder to justify in review.

**Shallow and narrow (3 conv blocks, 32→64→128 channels).** Chosen as a fast, cheaply
trained, easily explained baseline (`model.py`'s own docstring calls this out explicitly) —
not tuned via architecture search. Doubling channels while halving temporal resolution at
each stage is the standard convnet pattern (trade spatial/temporal resolution for channel
depth as features get more abstract); the specific widths (32/64/128) were a reasonable
starting guess, not a searched hyperparameter.

**Kernel sizes 7, 5, 5 (not increasing with depth).** The first layer uses a wider kernel
(7 taps = 70ms at 100Hz) to span QRS-scale features directly at raw input resolution, where
per-sample noise is highest and a wider filter averages over more of it. Later layers use
smaller kernels (5 taps) because they're already operating on downsampled, already-abstracted
feature maps where a wide kernel would waste capacity.

**BatchNorm after every conv.** Standard stabilization, lets a higher learning rate work
without divergence, and re-normalizes each conv's output regardless of amplitude variation
across the 12 leads (which differ systematically — limb leads are lower amplitude than
precordial leads).

**Global average pooling instead of flatten + dense.** Three consequences, in order of how
much they actually mattered here: (1) makes the classifier head input-length-agnostic — the
exact same architecture works unmodified on the 500Hz PTB-XL variant (5000 samples) without
redesigning anything, useful if that's revisited later; (2) keeps parameter count small — a
naive flatten of `(128, 250)` into even a modest dense hidden layer would cost tens of
millions of parameters versus this model's 55K total; (3) is itself a translation-invariant
"detector pooling" — if a diagnostic pattern occurs anywhere in the window, it's captured,
without needing to localize it.

**Dropout(0.3) immediately before the final Linear.** The only explicit regularization
besides BatchNorm and the small parameter count itself, placed where overfitting risk is
concentrated — the 128→5 compression is the only point where the model commits to a
prediction from a small feature vector.

**Class-weighted cross-entropy loss.** Required given NORM is ~56% of single-label records
(README §6). Two weighting schemes were tried in Phase 1 (raw inverse-frequency, then
sqrt inverse-frequency) — see README §6 for the honest result: neither fixed the HYP-class
collapse, because the underlying problem is data scarcity (535 total HYP examples), not loss
weighting.

**Adam + ReduceLROnPlateau, early-stopped on macro-F1 (not val_loss).** The scheduler and
early-stopping criterion were both switched from val_loss to macro-F1 partway through Phase
1 for a documented reason (README §6): val_loss is computed under the same class weights as
training, so it can plateau or drift independently of the actual target metric.

**Fixed seed (42).** Reproducibility, added per `IMPLEMENTATION_PLAN.md`'s explicit Phase 1
TODO list — not present in the original scaffold.

## 4. Limitations

- **HYP class collapse is a data problem, not an architecture problem.** F1 0.16 on HYP
  (535 total examples, ~427 after the train fold split) persisted across two different loss
  weighting schemes — see README §6 for the full before/after comparison. Nothing in this
  document's design choices addresses that; a fix would mean more HYP data (oversampling,
  augmentation) or accepting a narrower-scope metric, not a bigger model.
- **Global average pooling discards timing and order.** The model can't say *which* beat or
  *where* in the 10 seconds a diagnostic pattern occurred, and can't distinguish "this
  pattern happened once" from "this pattern happened throughout the recording" beyond a
  simple average. This also means no localization/explainability (e.g. Grad-CAM-style
  attribution back to a specific time window) has been attempted — though the architecture
  doesn't preclude it, since the last conv layer still has 250 temporal positions before
  pooling collapses them.
- **No lead-dropout or missing-lead robustness.** Trained and evaluated with all 12 leads
  always present. Behavior with a corrupted or missing lead — a realistic clinical scenario —
  is untested.
- **Per-lead z-score normalization erases absolute amplitude information.** `dataset.py`
  normalizes each lead to zero mean / unit variance per-record before the model ever sees
  it. This is standard practice, but it means genuinely low-voltage QRS complexes — a real
  diagnostic sign (the raw PTB-XL SCP codes include `LVOLT`, low-voltage, as seen in the raw
  metadata) — get rescaled away at the preprocessing stage. This is a limitation of the
  pipeline the model sits in, not the conv architecture itself, but it bounds what the model
  can possibly learn regardless of architecture changes.
- **Only validated at 100Hz.** The 500Hz PTB-XL variant is architecturally supported (global
  average pooling makes the head length-agnostic) but has never actually been run.
- **No architecture search.** Channel widths, kernel sizes, and depth were chosen by
  convention and the reasoning above, not tuned. This is the natural place to spend effort
  for a "more complex model" iteration — and exactly the scenario the interactive viewer
  (`results/ecg_viewer.html`, see README §7) was built to support without any viewer changes.
- **No confidence calibration check.** The interactive viewer surfaces raw softmax outputs
  as "confidence," but there's no reliability diagram or expected-calibration-error analysis
  behind that framing. A small model, trained with aggressive class weighting, on a class
  with 535 examples, is a plausible candidate for miscalibration — this hasn't been checked.
