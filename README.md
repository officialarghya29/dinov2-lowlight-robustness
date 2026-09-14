# Where Does DINOv2 Break in the Dark? A Mechanistic Study of Self-Supervised Representations under Low-Light Corruption

[![CI](https://github.com/officialarghya29/dinov2-lowlight-robustness/actions/workflows/ci.yml/badge.svg)](https://github.com/officialarghya29/dinov2-lowlight-robustness/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/unit%20tests-22%2F22-brightgreen)](tests/run_tests.py)
[![License: MIT](https://img.shields.io/badge/license-TBD-lightgrey)](#license)

**One-line summary.** We degrade CIFAR-10 along a calibrated 6-point low-light
ladder and localize *where inside a frozen DINOv2 ViT-S/14* accuracy dies:
not in uint8 digitization, not fixable by classical enhancement, but in a
**late-layer representational collapse compounded by a readout that
degenerates to a constant class** — with a reproducible **grace regime**
at mild darkness (never below clean; replicates and steepens on ViT-B/14).

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
| F1 | **Grace regime**: mild darkness never hurts, and can help | main curve, sev 0→1; 9 seed×C configs | 93.3% → **95.0%** (+1.7 pp; ≥ clean in 9/9 configs) |
| F2 | **Cliff** between sev 3 and 4, then a **floor** at chance | main curve | 61.7% → 23.3% (−38 pp), floor **10.8%** vs 10% chance |
| F3 | Drift is **late-layer dominated** and **deepens with backbone size** | unbiased CKA per block; ViT-B/14 replication | ViT-S late blocks (8–11) drop **0.79** vs early **0.47**; ViT-B gap steepens +0.32 → **+0.44** |
| F4 | Damage is **not** uint8 quantization | float vs digitized arms, same noise draw | gap ≈ **0 pp** at all severities (p = 1.0) |
| F5 | Darkness ≠ generic frequency damage | low-pass / high-pass controls at matched severity | sev 3: dark **61.7%**, low-pass 80.0% (p=2e-4), high-pass **14.2%** (p=2e-4) |
| F6 | At high severity the **readout collapses to one class** ("frog") | prediction-entropy + top-1 share | sev 5: **99%** of predictions are "frog", only 2 classes used; entropy ratio **0.02** |
| F7 | Retraining the readout recovers **+22.5 pp** at sev 4 | severity-adapted probe | 23.3% → **45.8%** (backbone untouched) |
| F8 | Classical enhancement does **not** rescue and can *hurt* | gain / gamma / CLAHE | best (gain, sev 4): **+6.7 pp**; CLAHE at clean: **−27.5 pp** |
| F9 | Findings are seed-robust | 3 splits × 3 probe-C | sev 5: **10.9 ± 1.5%**; failures at sev 1 persist **100%** to sev 5 |
| F10 | The frog collapse is a **geometry attractor**, not prior replay | 5× cat-oversampled prior, same fold | sev-5 collapse stays frog @ 99% even when the prior's top class is cat |
| F11 | **Readout repair beats input-space enhancement** (corollary confirmed) | adapted probe vs best enhancement, paired test | **+15.8 pp** at sev 4 (p=0.007), **+20.0 pp** at sev 5 (p=4e-4) |

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
| H2 | Embedding drift grows with severity and concentrates late | ✅ Supported | cos 1.00→0.16; CKA late-minus-early = **+0.32** (ViT-B: **+0.44**) |
| H3 | High severity is dominated by irreversible (quantization) loss | ❌ **Rejected** | float-vs-uint8 gap ≈ 0 at every severity |
| H4 | Part of the collapse is readout-staleness, repairable without touching the backbone | ✅ Supported | adapted probe **+22.5 pp** at sev 4 |
| H5 | Classical photometric enhancement restores accuracy | ❌ **Rejected** | ≤ +6.7 pp at the cliff (gain/gamma), ~0 at floor; CLAHE actively harmful |
| H6 | The fixed readout degenerates toward a **constant class** in the dark | ✅ Supported (new) | frog share 28%→72%→99% at sev 3/4/5 |

---

## Main results

### Table 1 — Low-light response curve (frozen DINOv2 ViT-S/14, logistic probe, n=120)

| Sev. | Brightness | Acc (%) | 95% CI (boot) | cos to clean | Margin (correct) | Part. ratio |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 1.00 | **93.3** | [88.3, 97.5] | 1.00 | 6.13 | 43.4 |
| 1 | 0.75 | **95.0** | [90.0, 98.3] | 0.94 | 6.28 | 43.6 |
| 2 | 0.55 | 90.0 | [84.2, 95.0] | 0.80 | 5.74 | 39.3 |
| 3 | 0.38 | 61.7 | [52.5, 70.0] | 0.60 | 4.86 | 33.2 |
| 4 | 0.25 | 23.3 | [15.0, 30.0] | 0.31 | 2.80 | 24.6 |
| 5 | 0.15 | 10.8 | [5.0, 16.7] | 0.16 | 3.00 | 16.5 |

Both curve-shape nulls rejected (linearity p = 0.001; midpoint symmetry
p = 0.001): the response is a **regime structure** — grace (sev ≤ 1), cliff
(2–4), floor (5) — not a slope.

![Main curve](docs/figures/fig_main_curve.png)

### Table 2 — Representational drift by depth (unbiased linear CKA to clean)

| Block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | **9** | **10** | **11** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CKA (sev 5) | .49 | .45 | .59 | .56 | .50 | .45 | .38 | .30 | **.24** | **.23** | **.20** | **.19** |

Early blocks (0–3) lose 0.47 on average; late blocks (8–11) lose **0.79**.
The largest single drop is the **final block (0.82)** — drift *accumulates
through depth*, consistent with photometric damage compounding across
attention/MLP layers rather than being a shallow contrast-normalization
artifact. The gradient **replicates and steepens on ViT-B/14** (late−early
gap **+0.44** vs +0.32 on ViT-S; see the generality run in
`results_pilot/vitb_*`).

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
| 1 | 95.0 | 93.3 | −1.7 |
| 2 | 90.0 | 88.3 | −1.7 |
| 3 | 61.7 | 69.2 | +7.5 |
| 4 | 23.3 | **45.8** | **+22.5** |
| 5 | 10.8 | **31.7** | **+20.8** |

A probe retrained *on degraded images* recovers nearly a third of the cliff at sev 4
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
| 1 | 93.3 | 95.0 | −1.7 |
| 2 | 89.2 | 90.0 | −0.8 |
| 3 | 61.7 | 61.7 | 0.0 |
| 4 | 23.3 | 23.3 | 0.0 |
| 5 | 10.8 | 10.8 | 0.0 (p = 1.0) |

**H3 rejected.** The camera's uint8 digitization is essentially innocent: the
damage is done to the *continuous* image before digitization. Practical
implication: sensor-side float capture or higher-bit depth would buy nothing
for a frozen DINOv2 under this corruption.

### Table 6 — Zero-training mitigation (applied to dark images)

| Sev. | None (%) | Gain ×a (%) | Gamma (%) | CLAHE (%) |
|---:|---:|---:|---:|---:|
| 0 | 93.3 | 93.3 | 92.5 | **65.8** |
| 1 | 95.0 | 94.2 | 92.5 | 53.3 |
| 2 | 90.0 | 89.2 | 86.7 | 31.7 |
| 3 | 61.7 | **68.3** | **68.3** | 21.7 |
| 4 | 23.3 | **30.0** | 19.2 | 11.7 |
| 5 | 10.8 | 11.7 | 11.7 | 10.0 |

**H5 rejected as a fix.** The best classical interventions (gain, gamma) buy
+6.7 pp at the cliff and vanish at the floor; CLAHE — the standard low-light
heuristic —
*destroys* up to 27.5 pp even on clean images, because it redistributes
spectral energy the frozen features depend on. Enhancement fixes how images
*look* to humans, not how they land in DINOv2's embedding geometry.

![Mitigation](docs/figures/fig_mitigation.png)

### Table 7 — How it fails: per-class breakdown and the frog collapse

| Class | Clean (%) | Sev 1 (%) | Sev 3 (%) | Sev 5 (%) | First sev < 50% |
|---|---:|---:|---:|---:|---:|
| airplane | 92 | 92 | 85 | 8 | 4 |
| automobile | 100 | 100 | 33 | 0 | 3 |
| bird | 100 | 89 | 56 | 0 | 4 |
| cat | 83 | 92 | 42 | 0 | 3 |
| deer | 91 | 100 | 73 | 0 | 4 |
| dog | 85 | 85 | 38 | 0 | 3 |
| **frog** | 92 | 100 | 92 | **100** | **never** |
| horse | 93 | 93 | 80 | 0 | 4 |
| ship | 100 | 100 | 62 | 0 | 4 |
| truck | 100 | 100 | 46 | 0 | 3 |

Frog's "100%" is **not** perception — it is an artifact of readout collapse:

| Sev. | Top-1 predicted class | Top-1 share | Classes used | Entropy / ln 10 |
|---:|---|---:|---:|---:|
| 0 | horse | 0.12 | 10 | 0.99 |
| 1 | horse | 0.12 | 10 | 0.99 |
| 2 | ship | 0.12 | 10 | 0.99 |
| 3 | frog | 0.28 | 10 | 0.91 |
| 4 | frog | 0.72 | 10 | 0.50 |
| 5 | **frog** | **0.99** | **2** | **0.02** |

As darkness deepens, the fixed probe's output distribution collapses from
near-uniform (entropy ratio 0.99) to **near-constant "frog"** (0.02) — the
readout falls back to its training prior. Per-class accuracy alone would have
celebrated frog; the prediction-distribution analysis exposes the mechanism.
Failure persistence is total: **100%** of images misclassified at sev 1 are
still misclassified at sev 5.

### Table 8 — Seed / hyperparameter sensitivity (3 splits × 3 probe-C, n=9 per cell)

| Sev. | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Acc (%) | 92.2 ± 1.3 | 93.6 ± 1.1 | 89.4 ± 3.1 | 67.3 ± 7.1 | 27.2 ± 5.0 | 10.9 ± 1.5 |

The grace regime (never below clean), the cliff, and the floor reproduce across every
split and every probe regularization: the regime structure is a property of
the backbone–corruption interaction, not of one lucky draw.

![Sensitivity](docs/figures/fig_sensitivity.png)

### Table 9 — Efficiency (CPU, batch=1, median of 50; 11.03 GFLOPs frozen)

| Arm | Backbone (M) | Trainable (M) | Latency (ms/img) | Throughput (img/s) |
|---|---:|---:|---:|---:|
| Frozen | 22.1 | 0.00 | **40.6** | 24.6 |
| + LoRA r4 (attn_mlp) | 22.4 | 0.29 | 49.0 | 20.4 |
| + LoRA r8 (attn_mlp) | 22.6 | 0.59 | 50.9 | 19.6 |
| + LoRA r16 (attn_mlp) | 23.2 | 1.18 | 50.7 | 19.7 |

Adapters cost 21–25% latency for ≤ 1.2M trainable parameters (5.3% of the
backbone) — the efficiency axis of the LoRA Pareto analysis that the full GPU
run completes with accuracy columns.

---

## Unifying account

The six regimes form one causal chain, each link measured above:

```
photon starvation (continuous-tone loss)          [Table 5: uint8 innocent]
        ↓
embedding drift, compounding with depth           [Table 2: late CKA 0.19; ViT-B +0.44]
        ↓   (grace regime: mild darkness never hurts)   [Table 1: +1.7 pp]
        ↓
spectral distinctness from blur/noise breaks down [Table 3: dissociation]
        ↓
decision margins shrink → argmax degenerates
to the training prior (constant class)            [Table 7: entropy 0.02]
        ↓
floor at ~chance even though features retain
signal a retrained readout can exploit            [Table 4: +22.5 pp]
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
| Geometry-attractor test (F10) + readout-vs-enhancement (F11) | `run_corollaries.py` | `results_pilot/corollaries.json`, `corollary2_floor_stats.csv` |
| ViT-B/14 replication | `run_vitb_generality.py` | `results_pilot/vitb_*.csv`, `vitb_cka_summary.json` |
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
├── run_vitb_generality.py     # ViT-B/14 replication of curve + CKA gradient
├── run_all_colab.py           # one-shot Colab GPU runner: all experiments + figures + zip
├── make_paper_tables.py       # CSV → LaTeX tables + macros (paper/results/)
├── make_paper_figures.py      # CSV → publication PDF/PNG figures (paper/figures/)
├── make_readme_figures.py     # CSV → README figures (docs/figures/)
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
python3 run_vitb_generality.py    # optional: ViT-B/14 replication (~10 min)
python3 run_corollaries.py        # optional: tests the paper's corollaries
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

## Compiling the paper

The CVPR-style paper lives in [`paper/`](paper/) with **vendored official
style files** (`cvpr.sty`, `ieeenat_fullname.bst` from
[cvpr-org/author-kit](https://github.com/cvpr-org/author-kit)) and builds
with one command:

```bash
cd paper && make        # or: pdflatex main && bibtex main && pdflatex main ×2
```

Both `main.pdf` (8 pages) and `supl.pdf` build warning-free from committed
tables/figures; on Overleaf, upload the `paper/` folder as-is. All tables and
macros are regenerated from `results_pilot/` by `make_paper_tables.py`.

## Roadmap to submission

See [`paper/CVPR2027_CHECKLIST.md`](paper/CVPR2027_CHECKLIST.md) for the
item-by-item audit. Highest-value next runs: (1) full-scale Colab suite
(LoRA + anchored-adapter ablation, STL-10 transfer), (2) real-darkness
validation (ExDark), (3) predicting the geometry attractor from embedding
distances alone (the revised corollary 2).

## License

TBD by repository owner (suggest MIT for code; figures/tables CC-BY-4.0).

## Citation

If you use this harness or build on the findings, please cite
[`CITATION.cff`](CITATION.cff) (software) and the paper:

> A. Biswas. *When Vision Goes Dark: A Mechanistic Decomposition of Low-Light
> Failure in Self-Supervised Vision Transformers.* Under preparation for
> CVPR 2027.
