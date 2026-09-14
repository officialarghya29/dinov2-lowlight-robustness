"""Test the paper's falsifiable corollaries 1 and 2 with the real harness.

Corollary 2 (floor class identity): "the accuracy floor scales with the
readout's prior over the dominant class -- class-balanced readout training
should change the floor's class identity even if its height does not."

    Test: fixed probe trained with class_weight='balanced' vs default
    (prior-weighted) on the same clean fold; compare the floor's dominant
    predicted class and its entropy/share. Prediction: the dominant class
    MOVES (no longer frog) while the floor HEIGHT stays ~chance.

Corollary 1 (readout-side repair beats input-space enhancement):

    Test: adapted probe at sev 4/5 vs best input-space enhancement (gain) on
    the same fold, paired permutation test. Prediction: adapted >> gain.

Writes results_pilot/corollaries.json + corollaries CSV.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(max(1, os.cpu_count() - 2))

from collections import Counter

from src.common import load_yaml_config, save_csv, save_json, seed_everything
from src.corruptions import low_light, simple_gain
from src.data import CIFAR10_CLASS_NAMES, load_cifar10_subsets
from src.models import EmbeddingExtractor, load_dinov2
from src.probes import fit_probe, stratified_split
from src.analysis import paired_permutation_test

cfg = load_yaml_config("configs/pilot_cpu.yaml")
seed_everything(cfg["seed"])
d, p = cfg["data"], cfg["probe"]

images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
model = load_dinov2(cfg["model"]["backbone"], "cpu")
ex = EmbeddingExtractor(model, "cpu")
idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])

E_te = {s: ex([low_light(images[i], s) for i in idx_te], batch_size=64) for s in range(6)}
E_tr_clean = ex([low_light(images[i], 0) for i in idx_tr], batch_size=64)

# ---- Corollary 2: class-balanced readout changes the floor's class identity ----
probe_prior = fit_probe(E_tr_clean, labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
probe_balanced = fit_probe(E_tr_clean, labels[idx_tr], C=p["C"], max_iter=p["max_iter"],
                           class_weight="balanced")


def floor_stats(probe, preds, y_true):
    dist = Counter(preds)
    top_cls, top_n = dist.most_common(1)[0]
    n = len(preds)
    ent = -sum((c / n) * np.log(c / n) for c in dist.values())
    return {"top1_class": CIFAR10_CLASS_NAMES[top_cls], "top1_share": top_n / n,
            "n_classes": len(dist), "entropy_ratio": ent / np.log(10),
            "accuracy": float((preds == y_true).mean())}


rows = []
for s in [4, 5]:
    preds_p = probe_prior.predict(E_te[s])
    preds_b = probe_balanced.predict(E_te[s])
    sp, sb = floor_stats(probe_prior, preds_p, labels[idx_te]), floor_stats(probe_balanced, preds_b, labels[idx_te])
    rows.append({"severity": s, "probe": "prior", **sp})
    rows.append({"severity": s, "probe": "balanced", **sb})
    print(f"sev {s} prior:    top1={sp['top1_class']} share={sp['top1_share']:.2f} "
          f"acc={sp['accuracy']:.3f}", flush=True)
    print(f"sev {s} balanced: top1={sb['top1_class']} share={sb['top1_share']:.2f} "
          f"acc={sb['accuracy']:.3f}", flush=True)

# does the dominant class move?
c2 = {
    "prior_top1_sev5": rows[-2]["top1_class"],
    "balanced_top1_sev5": rows[-1]["top1_class"],
    "identity_moved": rows[-2]["top1_class"] != rows[-1]["top1_class"],
    "prior_share_sev5": rows[-2]["top1_share"],
    "balanced_share_sev5": rows[-1]["top1_share"],
    "prior_entropy_sev5": rows[-2]["entropy_ratio"],
    "balanced_entropy_sev5": rows[-1]["entropy_ratio"],
    "prior_acc_sev5": rows[-2]["accuracy"],
    "balanced_acc_sev5": rows[-1]["accuracy"],
}
print("C2:", c2, flush=True)

# ---- Corollary 1: readout-side repair beats input-space enhancement ----
# adapted probe (trained on degraded fold) vs gain-restored input, fixed probe
correct_adapted, correct_gain, correct_none = {}, {}, {}
probe_adapted = {}
for s in [4, 5]:
    E_tr_deg = ex([low_light(images[i], s) for i in idx_tr], batch_size=64)
    probe_adapted[s] = fit_probe(E_tr_deg, labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
    correct_adapted[s] = (probe_adapted[s].predict(E_te[s]) == labels[idx_te]).astype(float)
    dark = [low_light(images[i], s) for i in idx_te]
    correct_gain[s] = (probe_prior.predict(ex([simple_gain(im, s) for im in dark],
                                              batch_size=64)) == labels[idx_te]).astype(float)
    correct_none[s] = (probe_prior.predict(E_te[s]) == labels[idx_te]).astype(float)

c1 = {}
for s in [4, 5]:
    t = paired_permutation_test(correct_adapted[s], correct_gain[s], seed=cfg["seed"])
    c1[f"sev{s}"] = {
        "acc_adapted": float(correct_adapted[s].mean()),
        "acc_gain": float(correct_gain[s].mean()),
        "acc_none": float(correct_none[s].mean()),
        "adapted_minus_gain_pp": 100 * (correct_adapted[s].mean() - correct_gain[s].mean()),
        "p_adapted_vs_gain": t["p_value"],
    }
    print(f"C1 sev {s}: {c1[f'sev{s}']}", flush=True)

# ---- Corollary 2b (discriminating test): IMBALANCED probe training fold ----
# The stratified train fold is nearly uniform, so 'balanced' reweighting is a
# near no-op (see C2 above). Oversampling a NON-frog class x5 makes the
# training prior genuinely non-uniform: if the sev-5 collapse follows the
# oversampled class, the collapse is prior-replay; if it stays frog, the
# collapse is a geometry attractor in the dark embedding space.
OVERSAMPLE_CLASS = CIFAR10_CLASS_NAMES.index("cat")
OVERSAMPLE_FACTOR = 5
rng = np.random.default_rng(cfg["seed"])
cat_idx = np.where(labels[idx_tr] == OVERSAMPLE_CLASS)[0]
extra = rng.choice(cat_idx, size=min(OVERSAMPLE_FACTOR * len(cat_idx), len(cat_idx) * 5) - len(cat_idx), replace=True)
imb_idx = np.concatenate([np.arange(len(idx_tr)), extra])
E_tr_imb = E_tr_clean[imb_idx]
y_imb = labels[idx_tr][imb_idx]
probe_imb = fit_probe(E_tr_imb, y_imb, C=p["C"], max_iter=p["max_iter"])
preds_imb = probe_imb.predict(E_te[5])
s_imb = floor_stats(probe_imb, preds_imb, labels[idx_te])
rows.append({"severity": 5, "probe": f"imbalanced_cat_x{OVERSAMPLE_FACTOR}", **s_imb})
print(f"sev 5 imbalanced(cat x{OVERSAMPLE_FACTOR}): top1={s_imb['top1_class']} "
      f"share={s_imb['top1_share']:.2f} acc={s_imb['accuracy']:.3f}", flush=True)
c2["imbalanced_top1_sev5"] = s_imb["top1_class"]
c2["imbalanced_share_sev5"] = s_imb["top1_share"]
c2["imbalanced_follows_prior"] = s_imb["top1_class"] == "cat"
c2["interpretation"] = ("prior-replay: collapse follows the training prior" if
                        c2["imbalanced_follows_prior"] else
                        "geometry-attractor: collapse persists despite imbalanced prior")
print("C2 interpretation:", c2["interpretation"], flush=True)

save_csv(rows, "results_pilot/corollary2_floor_stats.csv")
save_json({"corollary1_readout_vs_enhancement": c1, "corollary2_floor_identity": c2,
           "note": "pilot scale; tests the falsifiable corollaries in paper Sec. Discussion"},
          "results_pilot/corollaries.json")
print("saved results_pilot/corollaries.json", flush=True)
