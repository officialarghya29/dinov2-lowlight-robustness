"""Quantify readout collapse: prediction-distribution statistics per severity.

Finding (pilot): at high severity the fixed probe degenerates toward
constant-class prediction ("frog"), so per-class accuracy tracks the prior,
not perception. This script produces the evidence table:

    results_pilot/prediction_collapse.csv

Also re-measures LoRA-arm latency with more iterations (the pilot's 30-iter
r4 measurement was contaminated by cold-start cache effects).
"""

import os
import sys

import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(max(1, os.cpu_count() - 2))

from src.common import load_yaml_config, save_csv, seed_everything
from src.corruptions import low_light
from src.data import CIFAR10_CLASS_NAMES, load_cifar10_subsets
from src.models import EmbeddingExtractor, load_dinov2
from src.probes import fit_probe, stratified_split

cfg = load_yaml_config("configs/pilot_cpu.yaml")
seed_everything(42)
images, labels, *_ = load_cifar10_subsets(cfg["data"]["n_test"], cfg["data"]["n_train"], 42)
model = load_dinov2(cfg["model"]["backbone"], "cpu")
ex = EmbeddingExtractor(model, "cpu")
idx_tr, idx_te = stratified_split(labels, cfg["data"]["test_fraction"], 42)
probe = fit_probe(ex([low_light(images[i], 0) for i in idx_tr], batch_size=64),
                  labels[idx_tr])

rows = []
for s in range(6):
    preds = probe.predict(ex([low_light(images[i], s) for i in idx_te], batch_size=64))
    dist = Counter(preds)
    top_cls, top_n = dist.most_common(1)[0]
    n = len(preds)
    ent = -sum((c / n) * np.log(c / n) for c in dist.values())  # nats, max = ln 10
    rows.append({
        "severity": s,
        "accuracy": float((preds == labels[idx_te]).mean()),
        "top1_class": CIFAR10_CLASS_NAMES[top_cls],
        "top1_share": top_n / n,
        "n_classes_predicted": len(dist),
        "pred_entropy_nats": float(ent),
        "entropy_ratio_vs_uniform": float(ent / np.log(10)),
    })
    print(f"sev {s}: top1={CIFAR10_CLASS_NAMES[top_cls]} share={top_n / n:.2f} "
          f"classes={len(dist)} entropy_ratio={ent / np.log(10):.3f}", flush=True)

save_csv(rows, "results_pilot/prediction_collapse.csv")
print("saved results_pilot/prediction_collapse.csv")

# ---- latency re-measurement with proper warmup ----
from src.lora import apply_lora
from src.supplementary import _measure_latency


def _rows_safe(path):
    import csv
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


eff = _rows_safe("results_pilot/efficiency.csv")
if eff:
    for rank in cfg["lora"]["ranks"]:
        seeded = load_dinov2(cfg["model"]["backbone"], "cpu")
        apply_lora(seeded, rank=rank, alpha=rank * cfg["lora"]["alpha_multiplier"],
                   target=cfg["lora"]["targets"][-1])
        lat = _measure_latency(seeded, "cpu", n_warmup=10, n_iter=50)
        for r in eff:
            if r["arm"] == f"lora_r{rank}":
                r["latency_ms"] = lat["latency_ms_per_image"]
                r["throughput_img_per_s"] = lat["throughput_img_per_s"]
        print(f"lora_r{rank}: {lat['latency_ms_per_image']:.1f} ms/img", flush=True)
    save_csv(eff, "results_pilot/efficiency.csv")
    print("updated results_pilot/efficiency.csv")
