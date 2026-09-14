"""Backbone-generality check: does the late-layer CKA collapse replicate on a
bigger ViT? Runs the main curve + layer-wise CKA with dinov2_vitb14 at pilot
scale and writes results_pilot/vitb_*.csv / .json next to the ViT-S artifacts.

    python3 run_vitb_generality.py           # ~15 min on 16-core CPU
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(max(1, os.cpu_count() - 2))

from src.common import load_yaml_config, save_csv, save_json, seed_everything
from src.corruptions import low_light
from src.data import load_cifar10_subsets
from src.models import EmbeddingExtractor, load_dinov2
from src.probes import fit_probe, stratified_split
from src.analysis import linear_cka_unbiased

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "pilot_cpu.yaml")
OUTDIR = "results_pilot"
BACKBONE = "dinov2_vitb14"

cfg = load_yaml_config(CONFIG)
seed_everything(cfg["seed"])
d, m, p = cfg["data"], cfg["model"], cfg["probe"]
t0 = time.time()

print(f"=== {BACKBONE}: main curve + CKA (pilot scale) ===", flush=True)
images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
model = load_dinov2(BACKBONE, "cpu")
extractor = EmbeddingExtractor(model, "cpu")
idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])

# ---- main curve ----
E_by_sev = {}
for s in range(6):
    E_by_sev[s] = extractor([low_light(img, s) for img in images], batch_size=16)
    print(f"  sev {s} extracted ({time.time()-t0:.0f}s)", flush=True)

probe = fit_probe(E_by_sev[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
rows = []
for s in range(6):
    E_te = E_by_sev[s][idx_te]
    acc = float((probe.predict(E_te) == labels[idx_te]).mean())
    a = E_by_sev[0][idx_te] / (np.linalg.norm(E_by_sev[0][idx_te], axis=1, keepdims=True) + 1e-8)
    b = E_te / (np.linalg.norm(E_te, axis=1, keepdims=True) + 1e-8)
    cos = float((a * b).sum(1).mean())
    rows.append({"severity": s, "accuracy": acc, "mean_cosine_to_clean": cos})
    print(f"  sev {s}: acc={acc:.3f} cos={cos:.3f}", flush=True)
save_csv(rows, f"{OUTDIR}/vitb_main_curve.csv")

# ---- layer-wise CKA (test fold, severities 0/3/5) ----
te_images = [images[i] for i in idx_te]
layer_cache = {}
for s in [0, 3, 5]:
    _, layers = extractor([low_light(img, s) for img in te_images], batch_size=16,
                          collect_layers=True)
    layer_cache[s] = layers
    print(f"  sev {s} layers collected ({time.time()-t0:.0f}s)", flush=True)

n_layers = len(layer_cache[0])
drops = []
for l in range(n_layers):
    cka3 = linear_cka_unbiased(layer_cache[0][l], layer_cache[3][l])
    cka5 = linear_cka_unbiased(layer_cache[0][l], layer_cache[5][l])
    drops.append({"block": l, "cka_sev3": cka3, "cka_sev5": cka5,
                  "drop_0_to_5": 1.0 - cka5})
save_csv(drops, f"{OUTDIR}/vitb_cka_by_layer.csv")

d5 = np.array([r["drop_0_to_5"] for r in drops])
summary = {"backbone": BACKBONE,
           "max_drop_block": int(np.argmax(d5)), "max_drop": float(d5.max()),
           "early_mean_0_3": float(d5[:4].mean()), "late_mean_8_11": float(d5[8:].mean()),
           "late_minus_early": float(d5[8:].mean() - d5[:4].mean())}
save_json(summary, f"{OUTDIR}/vitb_cka_summary.json")

# comparison row for the paper
try:
    s_summary = __import__("json").load(open(f"{OUTDIR}/cka_summary.json"))
    print("\n=== depth-gradient comparison (drop 0->5) ===")
    print(f"  ViT-S/14: late={s_summary['late_mean_8_11']:.3f} early={s_summary['early_mean_0_3']:.3f} "
          f"diff={s_summary['late_minus_early']:+.3f}")
    print(f"  ViT-B/14: late={summary['late_mean_8_11']:.3f} early={summary['early_mean_0_3']:.3f} "
          f"diff={summary['late_minus_early']:+.3f}")
except FileNotFoundError:
    pass
print(f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)
