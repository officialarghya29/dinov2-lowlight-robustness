"""CPU pilot: runs all CPU-feasible paper experiments at pilot scale and writes
results/pilot_numbers.tex + results/PILOT_SUMMARY.md.

PILOT SCALE (configs/pilot_cpu.yaml): n_test=200 (test fold 60), n_perm=1000.
Purpose: validate the full pipeline end-to-end and fill provisional numbers.
Full-scale GPU runs (configs/experiments.yaml) supersede these in the paper.

Excludes: lora_ablation (training; Colab-only) and frequency (kept: cheap).
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(max(1, os.cpu_count() - 2))

from src.common import ensure_dir, load_yaml_config, save_json, seed_everything
from src.corruptions import (clahe_enhance, gamma_correct, high_pass, low_light,
                             low_light_stage1, low_pass, simple_gain)
from src.data import load_cifar10_subsets
from src.models import EmbeddingExtractor, load_dinov2
from src.probes import fit_probe, stratified_split
from src.analysis import (bootstrap_accuracy_ci, curve_permutation_test, linear_cka_unbiased,
                          paired_permutation_test, participation_ratio)

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "pilot_cpu.yaml")
OUTDIR = "results_pilot"
ensure_dir(OUTDIR)


def ckpt(name, rows):
    """Incremental checkpoint: a late crash must not erase earlier experiments."""
    from src.common import save_csv
    save_csv(rows, f"{OUTDIR}/{name}")
    print(f"  [ckpt] {name}", flush=True)

t_start = time.time()
cfg = load_yaml_config(CONFIG)
seed_everything(cfg["seed"])
device = "cpu"
d, m, p, st = cfg["data"], cfg["model"], cfg["probe"], cfg["stats"]

print("=== loading data + model ===", flush=True)
images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
model = load_dinov2(m["backbone"], device)
extractor = EmbeddingExtractor(model, device)
idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
print(f"test fold: {len(idx_te)} images | train fold: {len(idx_tr)}", flush=True)

# =========================================================================
# Exp 1: main curve (+ margins, geometry, null tests)
# =========================================================================
print("\n=== exp 1: main curve ===", flush=True)
E_by_sev = {}
for s in range(6):
    E_by_sev[s] = extractor([low_light(img, s) for img in images], batch_size=32)
    print(f"  sev {s} extracted", flush=True)

probe = fit_probe(E_by_sev[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

def margins_of(pr, X, y):
    sc = np.sort(pr.decision_function(X), axis=1)
    mgn = sc[:, -1] - sc[:, -2]
    return float(mgn.mean()), float(mgn[pr.predict(X) == y].mean())

main_rows, correctness = [], []
for s in range(6):
    E_te = E_by_sev[s][idx_te]
    preds = probe.predict(E_te)
    correct = (preds == labels[idx_te]).astype(float)
    correctness.append(correct)
    ci = bootstrap_accuracy_ci(correct, st["n_bootstrap"], st["ci"], seed=cfg["seed"])
    a = E_by_sev[0][idx_te] / (np.linalg.norm(E_by_sev[0][idx_te], axis=1, keepdims=True) + 1e-8)
    b = E_te / (np.linalg.norm(E_te, axis=1, keepdims=True) + 1e-8)
    cos = float((a * b).sum(1).mean())
    m_all, m_corr = margins_of(probe, E_te, labels[idx_te])
    main_rows.append({"severity": s, "brightness_factor": cfg["corruption"]["brightness_factors"][s],
                      "accuracy": float(correct.mean()), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                      "wilson_low": ci["wilson_low"], "wilson_high": ci["wilson_high"],
                      "mean_cosine_to_clean": cos, "mean_logit_margin": m_all,
                      "margin_correct_only": m_corr,
                      "participation_ratio": participation_ratio(E_te)})
    print(f"  sev {s}: acc={correct.mean():.3f} cos={cos:.3f} margin={m_all:.2f} PR={main_rows[-1]['participation_ratio']:.1f}", flush=True)

tests = curve_permutation_test(correctness, st["n_permutations"], seed=cfg["seed"])
print(f"  null tests: p_linear={tests['p_not_linear']:.4f} p_mid={tests['p_midpoint_asymmetric']:.4f}", flush=True)

pr_clean = main_rows[0]["participation_ratio"]
for r in main_rows:
    r["pr_ratio"] = r["participation_ratio"] / pr_clean
ckpt("main_curve.csv", main_rows)

# =========================================================================
# Exp 2: CKA (test fold)
# =========================================================================
print("\n=== exp 2: CKA ===", flush=True)
te_images = [images[i] for i in idx_te]
layer_cache = {}
for s in [0, 3, 5]:  # 3 severities keeps pilot cost down
    _, layers = extractor([low_light(img, s) for img in te_images], batch_size=32, collect_layers=True)
    layer_cache[s] = layers
    print(f"  sev {s} layers collected", flush=True)

n_layers = len(layer_cache[0])
cka_rows = []
for l in range(n_layers):
    row = {"block": l}
    for s in [0, 3, 5]:
        row[f"cka_sev{s}"] = linear_cka_unbiased(layer_cache[0][l], layer_cache[s][l])
    row["drop_0_to_5"] = row["cka_sev0"] - row["cka_sev5"]
    cka_rows.append(row)
    print(f"  block {l}: sev5 cka={row['cka_sev5']:.3f}", flush=True)

drops = np.array([r["drop_0_to_5"] for r in cka_rows])
cka_summary = {"max_drop_block": int(np.argmax(drops)), "max_drop": float(drops.max()),
               "early_mean_0_3": float(drops[:4].mean()), "late_mean_8_11": float(drops[8:].mean()),
               "late_minus_early": float(drops[8:].mean() - drops[:4].mean())}
print(f"  summary: {cka_summary}", flush=True)
ckpt("cka_by_layer.csv", cka_rows)

# =========================================================================
# Exp 3: frequency
# =========================================================================
print("\n=== exp 3: frequency ===", flush=True)
freq_rows = []
for s in [1, 2, 3]:
    conds = {"lowlight": [low_light(images[i], s) for i in idx_te],
             "lowpass": [low_pass(images[i], s) for i in idx_te],
             "highpass": [high_pass(images[i], s) for i in idx_te]}
    row = {"severity": s}
    corrects = {}
    for name, imgs in conds.items():
        c = (probe.predict(extractor(imgs, batch_size=32)) == labels[idx_te]).astype(float)
        corrects[name] = c
        row[f"acc_{name}"] = float(c.mean())
    row["p_dark_vs_lowpass"] = paired_permutation_test(corrects["lowlight"], corrects["lowpass"], seed=cfg["seed"])["p_value"]
    row["p_dark_vs_highpass"] = paired_permutation_test(corrects["lowlight"], corrects["highpass"], seed=cfg["seed"])["p_value"]
    freq_rows.append(row)
    print(f"  sev {s}: dark={row['acc_lowlight']:.3f} lp={row['acc_lowpass']:.3f} hp={row['acc_highpass']:.3f}", flush=True)

ckpt("frequency_tests.csv", freq_rows)

# =========================================================================
# Exp 4: mechanism — adapted probes (logit-lens/AOPC need token access; kept simple here)
# =========================================================================
print("\n=== exp 4: mechanism (adapted probes) ===", flush=True)
mech_rows = []
adapted_probe_by_sev = {}
for s in range(6):
    degraded_tr = [low_light(images[i], s) for i in idx_tr]
    adapted = fit_probe(extractor(degraded_tr, batch_size=32), labels[idx_tr],
                        C=p["C"], max_iter=p["max_iter"])
    adapted_probe_by_sev[s] = adapted
    acc_fixed = float((probe.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean())
    acc_adapted = float((adapted.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean())
    mech_rows.append({"severity": s, "acc_fixed": acc_fixed, "acc_adapted": acc_adapted,
                      "recovered_gap": acc_adapted - acc_fixed})
    print(f"  sev {s}: fixed={acc_fixed:.3f} adapted={acc_adapted:.3f} gap={acc_adapted-acc_fixed:+.3f}", flush=True)

t3 = paired_permutation_test(
    (adapted_probe_by_sev[3].predict(E_by_sev[3][idx_te]) == labels[idx_te]).astype(float),
    (probe.predict(E_by_sev[3][idx_te]) == labels[idx_te]).astype(float), seed=cfg["seed"])
print(f"  adapted-vs-fixed sev3 paired p={t3['p_value']:.4f}", flush=True)
ckpt("mechanism_adapted_probes.csv", mech_rows)

# =========================================================================
# Exp 5: quantization (SAME noise draws across arms)
# =========================================================================
print("\n=== exp 5: quantization ===", flush=True)
quant_rows = []
for s in range(6):
    float_imgs, uint8_imgs = [], []
    for i in idx_te:
        f = low_light_stage1(images[i], s)          # float stage-1 draw
        float_imgs.append(f)
        uint8_imgs.append(f.astype(np.uint8))        # SAME draw, digitized
    acc_f = float((probe.predict(extractor(float_imgs, batch_size=32, float_input=True)) == labels[idx_te]).mean())
    acc_u = float((probe.predict(extractor(uint8_imgs, batch_size=32)) == labels[idx_te]).mean())
    quant_rows.append({"severity": s, "acc_float_stage1": acc_f, "acc_uint8_full": acc_u,
                       "irreversible_gap": acc_f - acc_u})
    print(f"  sev {s}: float={acc_f:.3f} uint8={acc_u:.3f} gap={acc_f-acc_u:+.3f}", flush=True)
ckpt("quantization.csv", quant_rows)

q5 = paired_permutation_test(
    (probe.predict(extractor([low_light_stage1(images[i], 5) for i in idx_te], batch_size=32, float_input=True)) == labels[idx_te]).astype(float),
    (probe.predict(extractor([low_light(images[i], 5) for i in idx_te], batch_size=32)) == labels[idx_te]).astype(float),
    seed=cfg["seed"])

# =========================================================================
# Exp 7: mitigation
# =========================================================================
print("\n=== exp 7: mitigation ===", flush=True)
mit_rows = []
for s in range(6):
    dark = [low_light(images[i], s) for i in idx_te]
    row = {"severity": s,
           "acc_none": float((probe.predict(extractor(dark, batch_size=32)) == labels[idx_te]).mean())}
    for name, fn in [("gain", lambda im: simple_gain(im, s)),
                     ("gamma", lambda im: gamma_correct(im, s)),
                     ("clahe", lambda im: clahe_enhance(im))]:
        row[f"acc_{name}"] = float((probe.predict(extractor([fn(im) for im in dark], batch_size=32)) == labels[idx_te]).mean())
    mit_rows.append(row)
    print(f"  sev {s}: none={row['acc_none']:.3f} gain={row['acc_gain']:.3f} gamma={row['acc_gamma']:.3f} clahe={row['acc_clahe']:.3f}", flush=True)
ckpt("mitigation.csv", mit_rows)

# =========================================================================
# Save everything
# =========================================================================
from src.common import save_csv
save_csv(main_rows, f"{OUTDIR}/main_curve.csv")
save_csv(cka_rows, f"{OUTDIR}/cka_by_layer.csv")
save_csv(freq_rows, f"{OUTDIR}/frequency_tests.csv")
save_csv(mech_rows, f"{OUTDIR}/mechanism_adapted_probes.csv")
save_csv(quant_rows, f"{OUTDIR}/quantization.csv")
save_csv(mit_rows, f"{OUTDIR}/mitigation.csv")
save_json({**tests, "pilot_note": f"PILOT SCALE: n_test={d['n_test']}, fold={len(idx_te)}, n_perm={st['n_permutations']}"},
          f"{OUTDIR}/main_curve_nulltests.json")
save_json(cka_summary, f"{OUTDIR}/cka_summary.json")
save_json({"adapted_sev3_p": t3["p_value"], "quant_sev5_p": q5["p_value"]},
          f"{OUTDIR}/paired_tests.json")

# ---- numbers.tex macros ----
def pct(x): return f"{100*x:.1f}\\%"
macros = f"""% AUTO-GENERATED by run_pilot_cpu.py — PILOT SCALE (n_test=200, fold={len(idx_te)}).
% Supersede with a full-scale GPU run before submission.
\\newcommand{{\\AccClean}}{{{pct(main_rows[0]['accuracy'])}}}
\\newcommand{{\\AccFloor}}{{{pct(main_rows[5]['accuracy'])}}}
\\newcommand{{\\AccCliffDroppp}}{{{100*(main_rows[0]['accuracy']-main_rows[4]['accuracy']):.0f}\\,pp}}
\\newcommand{{\\CosSevOne}}{{{main_rows[1]['mean_cosine_to_clean']:.2f}}}
\\newcommand{{\\PNotNull}}{{{tests['p_not_linear']:.3f}}}
\\newcommand{{\\PMidNotNull}}{{{tests['p_midpoint_asymmetric']:.3f}}}
\\newcommand{{\\LateDrift}}{{{cka_summary['late_mean_8_11']:.3f}}}
\\newcommand{{\\EarlyDrift}}{{{cka_summary['early_mean_0_3']:.3f}}}
\\newcommand{{\\QuantGapFive}}{{{100*quant_rows[5]['irreversible_gap']:+.1f}\\,pp}}
\\newcommand{{\\QuantGapFiveP}}{{{q5['p_value']:.3f}}}
\\newcommand{{\\AdaptedGapThree}}{{{100*mech_rows[3]['recovered_gap']:+.1f}\\,pp}}
\\newcommand{{\\AdaptedGapThreeP}}{{{t3['p_value']:.3f}}}
"""
with open(f"{OUTDIR}/numbers.tex", "w") as f:
    f.write(macros)

print(f"\n=== PILOT DONE in {(time.time()-t_start)/60:.1f} min ===")
print(f"Artifacts in {OUTDIR}/ (numbers.tex, 6 CSVs, summaries)")
