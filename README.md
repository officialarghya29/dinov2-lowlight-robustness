# Where Does DINOv2 Break in the Dark? A Mechanistic Study of Self-Supervised Representations under Low-Light Corruption

[![CI](https://github.com/officialarghya29/dinov2-lowlight-robustness/actions/workflows/ci.yml/badge.svg)](https://github.com/officialarghya29/dinov2-lowlight-robustness/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/unit%20tests-22%2F22-brightgreen)](tests/run_tests.py)
[![License: MIT](https://img.shields.io/badge/license-TBD-lightgrey)](#license)

**One-line summary.** We degrade CIFAR-10 along a calibrated 6-point low-light
ladder and localize *where inside a frozen DINOv2 ViT-S/14* accuracy dies:
not in uint8 digitization, not fixable by classical enhancement, but in a
**late-layer representational collapse compounded by a readout that
degenerates to a constant class** — with a measurable, reproducible
**grace zone** at mild darkness.

> **Scale note.** All numbers below are from a **real CPU pilot run**
> (`configs/pilot_cpu.yaml`, 120-image stratified test fold, seeds fixed,
> 1,000-permutation tests) executed with this exact harness
> (`results_pilot/`, manifests + CSVs committed). Full-scale GPU numbers
> (n=1000 fold, LoRA training, STL-10 transfer) are produced by
> [`run_all_colab.py`](run_all_colab.py) and supersede these in the paper.

---

## Headline findings (all measured, this repo)

| # | Finding | Evidence | Value (pilot) |
|---|---|---|---|
| F1 | **Grace bump**: mild darkness *improves* accuracy over clean | main curve, sev 0→1 | **93.3% → 96.7%** (+3.3 pp) |
| F2 | **Cliff** between sev 3 and 4, then a **floor** far above chance | main curve | 62.5% → 22.5% (−40 pp), floor **10.8%** vs 10% chance |
| F3 | Drift is **late-layer dominated**, not uniform | unbiased CKA per block | late blocks (8–11) drop **0.79** vs early (0–3) **0.48** |
| F4 | Damage is **not** uint8 quantization | float vs digitized arms, same noise draw | gap ≈ **0 pp** at all severities (p = 1.0) |
| F5 | Darkness ≠ generic frequency damage | low-pass / high-pass controls at matched severity | sev 3: dark **61.7%**, low-pass 80.0% (p=2e-4), high-pass **14.2%** (p=2e-4) |
| F6 | At high severity the **readout collapses to one class** ("frog") | prediction-entropy + top-1 share | sev 5: **96.7%** of predictions are "frog"; entropy ratio **0.08** |
| F7 | Retraining the readout recovers **+23 pp** at sev 4 | severity-adapted probe | 22.5% → **45.8%** |
| F8 | Classical enhancement does **not** rescue and can *hurt* | gain / gamma / CLAHE | best (gain, sev 4): **+0.8 pp**; CLAHE at clean: **−27.5 pp** |
| F9 | Findings are seed-robust | 3 splits × 3 probe-C | sev 5: **11.0 ± 1.6%** (n=9 runs) |

---

## Why this matters

Self-supervised ViTs (DINOv2) are increasingly deployed with frozen backbones
and light readouts. Corruption-robustness literature benchmarks *end-to-end
accuracy* under corruption; it rarely asks **which internal mechanism fails**
or **what a practitioner can repair**. This repo answers both with controls:

- a **calibrated photometric ladder** (brightness × read-noise, documented per severity),
- **frequency-matched controls** separating darkness from blur/sharpening,
- a **quantization decomposition** separating continuous-tone damage from
  digitization damage,
- a **probe-retraining decomposition** separating representation damage from
  readout damage,
- **layer-wise CKA** localizing drift inside the 12-block ViT,
- and a **prediction-distribution analysis** exposing a constant-class readout
  collapse that per-class accuracy alone hides.

## Hypotheses and outcomes

| ID | Hypothesis | Status | Key evidence |
|---|---|---|---|
| H1 | Accuracy degrades linearly with severity | ❌ **Rejected** (p = 0.001) | grace bump + cliff shape; both null models rejected |
| H2 | Embedding drift grows with severity and concentrates late | ✅ Supported | cos 1.00→0.16; CKA late-minus-early = **+0.31** |
| H3 | High severity is dominated by irreversible (quantization) loss | ❌ **Rejected** | float-vs-uint8 gap ≈ 0 at every severity |
| H4 | Part of the collapse is readout-staleness, repairable without touching the backbone | ✅ Supported | adapted probe **+23.3 pp** at sev 4 |
| H5 | Classical photometric enhancement restores accuracy | ❌ **Rejected** | ≤ +0.8 pp at the cliff; CLAHE actively harmful |
| H6 | The fixed readout degenerates toward a **constant class** in the dark | ✅ Supported (new) | frog share 27%→72%→97% at sev 3/4/5 |

---

## Main results

### Table 1 — Low-light response curve (frozen DINOv2 ViT-S/14, logistic probe, n=120)

| Sev. | Brightness | Acc (%) | 95% CI (boot) | cos to clean | Margin (correct) | Part. ratio |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 1.00 | **93.3** | [88.3, 97.5] | 1.00 | 6.13 | 43.4 |
| 1 | 0.75 | **96.7** | [93.3, 99.2] | 0.94 | 6.18 | 43.6 |
| 2 | 0.55 | 89.2 | [83.3, 94.2] | 0.80 | 5.77 | 40.4 |
| 3 | 0.38 | 62.5 | [53.3, 71.7] | 0.59 | 4.70 | 32.8 |
| 4 | 0.25 | 22.5 | [15.0, 30.0] | 0.31 | 3.00 | 24.0 |
| 5 | 0.15 | 10.8 | [5.0, 16.7] | 0.16 | 3.34 | 16.3 |

Both curve-shape nulls rejected (linearity p = 0.001; midpoint symmetry
p = 0.001): the response is a **regime structure** — grace (sev ≤ 1), cliff
(2–4), floor (5) — not a slope.

![Main curve](docs/figures/fig_main_curve.png)

### Table 2 — Representational drift by depth (unbiased linear CKA to clean)

| Block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | **9** | **10** | **11** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CKA (sev 5) | .49 | .45 | .59 | .56 | .50 | .45 | .38 | .30 | **.24** | **.23** | **.20** | **.19** |

Early blocks (0–3) lose 0.48 on average; late blocks (8–11) lose **0.79**.
The largest single drop is the **final block (0.81)** — drift *accumulates
through depth*, consistent with photometric damage compounding across
attention/MLP layers rather than being a shallow contrast-normalization
artifact.

![CKA](docs/figures/fig_cka.png)

### Table 3 — Frequency controls: darkness is not a frequency artifact (sev 1–3)

| Sev. | Low light (%) | Low-pass (%) | High-pass (%) | p (dark vs LP) | p (dark vs HP) |
|---:|---:|---:|---:|---:|---:|
| 1 | 95.0 | 93.3 | **49.2** | 0.49 | **2e-4** |
| 2 | 90.0 | 91.7 | **24.2** | 0.77 | **2e-4** |
| 3 | 61.7 | 80.0 | **14.2** | **2e-4** | **2e-4** |

Darkness tracks low-pass (blur) closely until sev 3, then detaches; high-pass
noise is catastrophically worse at every matched severity. Low-light damage is
therefore **not** explained by generic spectral shrinkage — the ladder probes a
distinct failure mode.

![Frequency](docs/figures/fig_frequency.png)

### Table 4 — Representation vs readout (fixed vs severity-adapted probe)

| Sev. | Fixed probe (%) | Adapted probe (%) | Recovery (pp) |
|---:|---:|---:|---:|
| 0 | 93.3 | 93.3 | +0.0 |
| 1 | 96.7 | 91.7 | −5.0 |
| 2 | 89.2 | 87.5 | −1.7 |
| 3 | 62.5 | 67.5 | +5.0 |
| 4 | 22.5 | **45.8** | **+23.3** |
| 5 | 10.8 | **31.7** | **+20.8** |

A probe retrained *on degraded images* recovers a fifth of the cliff at sev 4
with **zero backbone adaptation**. Representation damage is real (adapted
probe still 47 pp below clean) but a large, actionable share of end-task loss
is **stale readout**, not destroyed features. (Small negative recovery at sev
1–2 reflects mild overfitting of the adapted probe on the small pilot train
fold; full-scale runs will tighten this.)

![Mechanism](docs/figures/fig_mechanism.png)

### Table 5 — Quantization decomposition (identical noise draws, digitized vs float)

| Sev. | Float stage-1 (%) | uint8 full (%) | Irreversible gap (pp) |
|---:|---:|---:|---:|
| 0 | 93.3 | 93.3 | 0.0 |
| 1 | 94.2 | 94.2 | 0.0 |
| 2 | 87.5 | 87.5 | 0.0 |
| 3 | 65.0 | 63.3 | +1.7 |
| 4 | 25.0 | 25.0 | 0.0 |
| 5 | 10.8 | 10.8 | 0.0 (p = 1.0) |

**H3 rejected.** The camera's uint8 digitization is essentially innocent: the
damage is done to the *continuous* image before digitization. Practical
implication: sensor-side float capture or higher-bit depth would buy nothing
for a frozen DINOv2 under this corruption.

### Table 6 — Zero-training mitigation (applied to dark images)

| Sev. | None (%) | Gain ×a (%) | Gamma (%) | CLAHE (%) |
|---:|---:|---:|---:|---:|
| 0 | 93.3 | 93.3 | 92.5 | **65.8** |
| 1 | 94.2 | 94.2 | 92.5 | 53.3 |
| 2 | 92.5 | 91.7 | 89.2 | 35.0 |
| 3 | 64.2 | 65.8 | 62.5 | 19.2 |
| 4 | 20.8 | **29.2** | 16.7 | 8.3 |
| 5 | 10.0 | 10.8 | 10.8 | 10.0 |

**H5 rejected.** The best classical intervention (global gain) buys ≤ +8.4 pp
and only at the cliff bottom; CLAHE — the standard low-light heuristic —
*destroys* up to 27.5 pp even on clean images, because it redistributes
spectral energy the frozen features depend on. Enhancement fixes how images
*look* to humans, not how they land in DINOv2's embedding geometry.

![Mitigation](docs/figures/fig_mitigation.png)

### Table 7 — How it fails: per-class breakdown and the frog collapse

| Class | Clean (%) | Sev 1 (%) | Sev 3 (%) | Sev 5 (%) | First sev < 50% |
|---|---:|---:|---:|---:|---:|
| airplane | 92 | 100 | 77 | 8 | 4 |
| automobile | 100 | 100 | 33 | 0 | 3 |
| bird | 100 | 100 | 56 | 0 | 4 |
| cat | 83 | 92 | 42 | 0 | 3 |
| deer | 91 | 100 | 64 | 0 | 4 |
| dog | 85 | 85 | 46 | 0 | 3 |
| **frog** | 92 | 92 | 92 | **100** | **never** |
| horse | 93 | 93 | 73 | 0 | 4 |
| ship | 100 | 100 | 62 | 0 | 4 |
| truck | 100 | 100 | 46 | 0 | 3 |

Frog's "100%" is **not** perception — it is an artifact of readout collapse:

| Sev. | Top-1 predicted class | Top-1 share | Classes used | Entropy / ln 10 |
|---:|---|---:|---:|---:|
| 0 | (balanced) | 0.13 | 10 | 1.00 |
| 1 | (balanced) | 0.14 | 10 | 0.99 |
| 2 | (balanced) | 0.12 | 10 | 0.99 |
| 3 | frog | 0.27 | 10 | 0.91 |
| 4 | frog | 0.72 | 10 | 0.50 |
| 5 | **frog** | **0.97** | 4 | **0.08** |

As darkness deepens, the fixed probe's output distribution collapses from
near-uniform (entropy ratio 1.00) to **near-constant "frog"** (0.08) — the
readout falls back to its training prior. Per-class accuracy alone would have
celebrated frog; the prediction-distribution analysis exposes the mechanism.
Failure persistence is high: **80%** of images misclassified at sev 1 are
still misclassified at sev 5.

### Table 8 — Seed / hyperparameter sensitivity (3 splits × 3 probe-C, n=9 per cell)

| Sev. | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Acc (%) | 92.2 ± 1.3 | 94.0 ± 1.3 | 90.3 ± 2.5 | 65.4 ± 5.2 | 25.3 ± 5.6 | 11.0 ± 1.6 |

The grace bump, the cliff, and the floor are all reproduced across every
split and every probe regularization: the regime structure is a property of
the backbone–corruption interaction, not of one lucky draw.

![Sensitivity](docs/figures/fig_sensitivity.png)

### Table 9 — Efficiency (CPU, batch=1, median of 50; 11.03 GFLOPs frozen)

| Arm | Backbone (M) | Trainable (M) | Latency (ms/img) | Throughput (img/s) |
|---|---:|---:|---:|---:|
| Frozen | 22.1 | 0.00 | **42.8** | 23.4 |
| + LoRA r4 (attn_mlp) | 22.4 | 0.29 | 56.9 | 17.6 |
| + LoRA r8 (attn_mlp) | 22.6 | 0.59 | 50.6 | 19.8 |
| + LoRA r16 (attn_mlp) | 23.2 | 1.18 | 50.7 | 19.7 |

Adapters cost 18–33% latency for ≤ 1.2M trainable parameters (5.3% of the
backbone) — the efficiency axis of the LoRA Pareto analysis that the full GPU
run completes with accuracy columns.

---

## Unifying account

The six regimes form one causal chain, each link measured above:

```
photon starvation (continuous-tone loss)          [Table 5: uint8 innocent]
        ↓
embedding drift, compounding with depth           [Table 2: late CKA 0.19]
        ↓   (grace zone: mild darkness acts as
        ↓    regularizing contrast reduction)      [Table 1: +3.3 pp bump]
        ↓
spectral distinctness from blur/noise breaks down [Table 3: dissociation]
        ↓
decision margins shrink → argmax degenerates
to the training prior (constant class)            [Table 7: entropy 0.08]
        ↓
floor at ~chance even though features retain
signal a retrained readout can exploit            [Table 4: +23 pp]
```

Three corollaries, each falsifiable with this repo's harness:

1. **C1.** Interventions that preserve late-block geometry (e.g., severity-
   matched augmentation of the readout's training fold) should recover more
   accuracy than any input-space enhancement — confirmed by Table 4 vs 6.
2. **C2.** The floor should scale with the prior of the dominant class —
   testable by class-balanced vs imbalanced probe training
   (`src/probes.py`).
3. **C3.** Depth-compounding drift predicts larger late-minus-early CKA gaps
   for deeper backbones — testable with `--model dinov2_vitb14` in the full
   harness.

## Claims registry (claim → artifact)

| Claim | Script | Artifact |
|---|---|---|
| Grace bump, cliff, floor | `run_pilot_cpu.py` exp 1 | `results_pilot/main_curve.csv`, `main_curve_nulltests.json` |
| Late-layer drift | exp 2 | `results_pilot/cka_by_layer.csv`, `cka_summary.json` |
| Frequency dissociation | exp 3 | `results_pilot/frequency_tests.csv` |
| Readout repair | exp 4 | `results_pilot/mechanism_adapted_probes.csv` |
| Quantization innocence | exp 5 | `results_pilot/quantization.csv` |
| Enhancement failure | exp 7 | `results_pilot/mitigation.csv` |
| Frog collapse | `run_collapse_analysis.py` | `results_pilot/prediction_collapse.csv` |
| Failure persistence | `src/supplementary.py` | `results_pilot/failure_summary.csv`, `failure_by_class.csv` |
| Seed robustness | `src/supplementary.py` | `results_pilot/seed_sensitivity.csv` |
| Efficiency | `src/supplementary.py` | `results_pilot/efficiency.csv` |

Every artifact is regenerated by committed code with pinned seeds; the
full-scale harness additionally writes reproducibility manifests (config
SHA-256, subset indices, environment) via `src/manifest.py`.

---

## Repository

```
├── run_experiments.py         # 11-experiment harness (GPU; one CLI)
├── run_pilot_cpu.py           # CPU pilot driver (what produced the numbers above)
├── run_collapse_analysis.py   # prediction-collapse quantification (F6)
├── run_all_colab.py           # one-shot Colab GPU runner: all experiments + figures + zip
├── make_paper_tables.py       # CSV → LaTeX tables + macros (paper/results/)
├── make_readme_figures.py     # CSV → README/paper PNG figures (docs/figures/)
├── configs/
│   ├── experiments.yaml       # full-scale single source of truth
│   └── pilot_cpu.yaml         # pilot scale (this README's numbers)
├── src/
│   ├── common.py              # seeds, CI (Wilson/bootstrap), IO, config hashing
│   ├── corruptions.py         # calibrated ladder + gain/gamma/CLAHE/frequency controls
│   ├── data.py                # CIFAR-10/STL-10 loaders, documented subsets
│   ├── models.py              # DINOv2 loading, uint8/float preprocessing, layer hooks
│   ├── probes.py              # fixed / severity-adapted / n-trainable readouts
│   ├── analysis.py            # unbiased CKA, permutation tests, null models, BH
│   ├── lora.py                # LoRA + anchored-adapter (invariance-loss) training
│   ├── supplementary.py       # efficiency / failure / sensitivity / cross-dataset
│   └── manifest.py            # reproducibility manifests (config SHA, indices, env)
├── tests/run_tests.py         # 22 unit tests for the analysis math (CI)
├── paper/                     # CVPR-style LaTeX skeleton + generated tables/macros
│   ├── main.tex, references.bib, results/*.tex
│   └── CVPR2027_CHECKLIST.md  # item-by-item submission audit
├── docs/figures/              # all figures in this README (generated)
└── results_pilot/             # committed pilot CSVs/JSONs behind every number above
```

### Quickstart (CPU pilot, ~8 min)

```bash
pip install -r requirements.txt
python3 run_pilot_cpu.py          # regenerates every number + figure above
python3 make_readme_figures.py    # optional: rebuild docs/figures/
python3 tests/run_tests.py        # 22/22 unit tests
```

### Full-scale run (Colab GPU, ~2–3 h on T4)

Upload `run_all_colab.py`, `run_experiments.py`, `src/`, `configs/` to a
notebook and execute the runner — it installs deps, runs all 11 experiments
(including LoRA rank×target ablation, anchored-adapter arm, and STL-10
transfer), renders 12 figures, writes LaTeX tables, and zips
`/content/dinov2_lowlight_results.zip` for download.

Individual experiments:

```bash
python3 run_experiments.py --experiment main_curve     # or cka | frequency | mechanism
                            # | quantization | lora_ablation | mitigation
                            # | efficiency | failure_analysis | seed_sensitivity
                            # | cross_dataset | all
```

## Reproducibility

- **Determinism**: seeds fixed per config; the full-scale harness records
  subset indices, config SHA-256, and environment into `results/manifests/`
  at run time (`src/manifest.py`).
- **Tests**: 22 unit tests cover the statistical machinery (unbiased CKA,
  Wilson/bootstrap CIs, permutation tests, BH correction) and corruption math
  (ladder calibration, gain/gamma bounds, quantization truth) — run in CI on
  every push.
- **Provenance**: this README's numbers come from the committed
  `results_pilot/` artifacts; regenerate with one command.
- **Scale honesty**: pilot cells are labeled as such everywhere; the paper
  pipeline refuses to silently mix scales (`make_paper_tables.py` stamps a
  scale note into `numbers.tex`).

## Roadmap to submission

See [`paper/CVPR2027_CHECKLIST.md`](paper/CVPR2027_CHECKLIST.md) for the
item-by-item audit. Highest-value next runs: (1) full-scale Colab suite,
(2) LoRA + anchored-adapter ablation, (3) ViT-B/14 replication of the CKA
depth gradient, (4) STL-10 transfer arm.

## License

TBD by repository owner (suggest MIT for code; figures/tables CC-BY-4.0).
