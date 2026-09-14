"""Experiment harness: 7 paper experiments behind one CLI.

    python3 run_experiments.py --experiment main_curve
    python3 run_experiments.py --experiment all --outdir results

Experiments
    main_curve    accuracy + cosine drift across severities (Table 1, Fig. 2)
                  + permutation tests vs linearity/symmetry nulls (Table 5)
    cka           layer-wise unbiased CKA (Fig. 3) + per-block drift stats
    frequency     low-pass vs high-pass vs low-light (Fig. 4) + paired tests
    mechanism     severity-adapted probes, logit margins, logit-lens, AOPC,
                  n-trainable readout (Fig. 5, Table 3)
    quantization  float stage-1 vs uint8 two-stage (Table 2) + RMS truth
    lora_ablation rank x target grid (Table 4, Pareto Fig. 6)
    mitigation    gain / gamma / CLAHE vs none (Table 6)

Every call writes CSVs + JSON to --outdir and appends a manifest line.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.common import (ensure_dir, get_device, load_yaml_config, save_csv, save_json,
                        seed_everything, sha256_of_file)
from src.corruptions import (BRIGHTNESS_FACTORS, clahe_enhance, gamma_correct, high_pass,
                             low_light, low_light_stage1, low_pass, quantization_gap_truth,
                             simple_gain)
from src.data import load_cifar10_subsets
from src.models import EmbeddingExtractor, load_dinov2
from src.probes import NTrainableProbe, fit_probe, stratified_split
from src.analysis import (benjamini_hochberg, bootstrap_accuracy_ci, curve_permutation_test,
                          linear_cka_unbiased, paired_permutation_test, participation_ratio)
from src.manifest import write_manifest

# torch is imported lazily inside the functions that need it so that the test
# suite (and CI) can exercise the analysis code on machines without torch.

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "experiments.yaml")


def cosine_to_clean(E_clean, E_sev):
    a = E_clean / (np.linalg.norm(E_clean, axis=1, keepdims=True) + 1e-8)
    b = E_sev / (np.linalg.norm(E_sev, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


# ---------------------------------------------------------------------------
# Experiment 1: main curve
# ---------------------------------------------------------------------------

def exp_main_curve(cfg, device, outdir):
    import torch

    d, m, p, st = cfg["data"], cfg["model"], cfg["probe"], cfg["stats"]
    seed_everything(cfg["seed"])
    images, labels, test_pool_idx, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])

    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)

    E_by_sev = {}
    for s in range(6):
        degraded = [low_light(img, s) for img in images]
        E_by_sev[s] = extractor(degraded)
        print(f"  severity {s}: extracted {E_by_sev[s].shape}")

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
    probe = fit_probe(E_by_sev[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

    rows, correctness = [], []
    clean_acc = None
    for s in range(6):
        E_te = E_by_sev[s][idx_te]
        preds = probe.predict(E_te)
        correct = (preds == labels[idx_te]).astype(float)
        correctness.append(correct)
        ci = bootstrap_accuracy_ci(correct, st["n_bootstrap"], st["ci"])
        cos = cosine_to_clean(E_by_sev[0][idx_te], E_te)
        m_mean, _, m_corr = logit_margins(probe, E_te, labels[idx_te])
        if clean_acc is None:
            clean_acc = correct.mean()
        rows.append({"severity": s, "brightness_factor": BRIGHTNESS_FACTORS[s],
                     "accuracy": correct.mean(), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                     "wilson_low": ci["wilson_low"], "wilson_high": ci["wilson_high"],
                     "mean_cosine_to_clean": float(cos.mean()),
                     "cosine_std": float(cos.std()),
                     "mean_logit_margin": m_mean,
                     "margin_correct_only": m_corr,
                     "accuracy_drop_pp": (clean_acc - correct.mean()) * 100})
        print(f"  severity {s}: acc={correct.mean():.4f} [{ci['ci_low']:.4f},{ci['ci_high']:.4f}]")

    save_csv(rows, f"{outdir}/main_curve.csv")
    save_json({"severity": list(range(6)),
               "correctness": [c.astype(int).tolist() for c in correctness]},
              f"{outdir}/main_curve_correctness.json")

    tests = curve_permutation_test(correctness, st["n_permutations"], seed=cfg["seed"])
    save_json(tests, f"{outdir}/main_curve_nulltests.json")
    print(f"  linearity null p={tests['p_not_linear']:.5f}  midpoint null p={tests['p_midpoint_asymmetric']:.5f}")

    write_manifest(outdir, CONFIG_PATH, "main_curve",
                   {"n_test": d["n_test"], "test_fold": len(idx_te)})
    return rows, tests


def logit_margins(probe, X, y):
    scores = probe.decision_function(X)
    srt = np.sort(scores, axis=1)
    margins = srt[:, -1] - srt[:, -2]
    preds = probe.predict(X)
    return float(margins.mean()), margins, float(margins[preds == y].mean())


# ---------------------------------------------------------------------------
# Experiment 2: layer-wise CKA
# ---------------------------------------------------------------------------

def exp_cka(cfg, device, outdir):
    import torch

    d, m = cfg["data"], cfg["model"]
    seed_everything(cfg["seed"])
    images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
    te_images = [images[i] for i in idx_te]  # CKA on the held-out fold only

    layers = {}
    pooled = {}
    for s in range(6):
        degraded = [low_light(img, s) for img in te_images]
        E, layerwise = extractor(degraded, collect_layers=True)
        pooled[s] = E
        for l, v in layerwise.items():
            layers.setdefault(l, {})[s] = v
        print(f"  severity {s}: collected {len(layerwise)} layers")

    n_layers = len(layers)
    cka = np.zeros((n_layers, 6))
    for l in range(n_layers):
        for s in range(6):
            cka[l, s] = linear_cka_unbiased(layers[l][0], layers[l][s])

    drop = cka[:, 0] - cka[:, 5]
    rows = []
    for l in range(n_layers):
        rows.append({"block": l, "cka_sev0": cka[l, 0], "cka_sev1": cka[l, 1],
                     "cka_sev2": cka[l, 2], "cka_sev3": cka[l, 3], "cka_sev4": cka[l, 4],
                     "cka_sev5": cka[l, 5], "drop_0_to_5": drop[l]})
    save_csv(rows, f"{outdir}/cka_by_layer.csv")

    summary = {
        "max_drop_block": int(np.argmax(drop)),
        "max_drop_value": float(drop.max()),
        "mean_drop": float(drop.mean()),
        "early_mean_drop_blocks_0_3": float(drop[:4].mean()),
        "late_mean_drop_blocks_8_11": float(drop[8:].mean()),
        "late_minus_early": float(drop[8:].mean() - drop[:4].mean()),
    }
    save_json(summary, f"{outdir}/cka_summary.json")
    print(f"  max drift block {summary['max_drop_block']} (drop={summary['max_drop_value']:.4f}); "
          f"late-early={summary['late_minus_early']:+.4f}")

    write_manifest(outdir, CONFIG_PATH, "cka", {"n_cka_samples": len(idx_te)})
    return rows, summary


# ---------------------------------------------------------------------------
# Experiment 3: frequency-band controls
# ---------------------------------------------------------------------------

def exp_frequency(cfg, device, outdir):
    d, m, p = cfg["data"], cfg["model"], cfg["probe"]
    seed_everything(cfg["seed"])
    images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
    probe = fit_probe(extractor([low_light(images[i], 0) for i in idx_tr]),
                      labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

    def eval_fn(fn):
        """Returns dicts keyed by severity (1..5)."""
        accs, corrects = {}, {}
        for s in [1, 2, 3, 4, 5]:
            degraded = [fn(images[i], s) for i in idx_te]
            E = extractor(degraded)
            preds = probe.predict(E)
            correct = (preds == labels[idx_te]).astype(float)
            accs[s] = correct.mean()
            corrects[s] = correct
        return accs, corrects

    ll_accs, ll_corrects = eval_fn(low_pass)
    hp_accs, hp_corrects = eval_fn(high_pass)
    dark_accs, dark_corrects = eval_fn(low_light)

    # Paired tests at each matched severity
    tests = []
    pvals = []
    for s in [1, 2, 3, 4, 5]:
        t1 = paired_permutation_test(dark_corrects[s], ll_corrects[s], seed=cfg["seed"])
        t2 = paired_permutation_test(dark_corrects[s], hp_corrects[s], seed=cfg["seed"])
        tests.append({"severity": s, "acc_lowlight": dark_accs[s], "acc_lowpass": ll_accs[s],
                      "acc_highpass": hp_accs[s],
                      "p_dark_vs_lowpass": t1["p_value"], "p_dark_vs_highpass": t2["p_value"]})
        pvals.extend([t1["p_value"], t2["p_value"]])

    adj = benjamini_hochberg(pvals)
    for i, t in enumerate(tests):
        t["p_dark_vs_lowpass_bh"] = adj[2 * i]
        t["p_dark_vs_highpass_bh"] = adj[2 * i + 1]

    save_csv(tests, f"{outdir}/frequency_tests.csv")
    write_manifest(outdir, CONFIG_PATH, "frequency", {"test_fold": len(idx_te)})
    print(f"  low-pass sev3={ll_accs[3]:.4f}  high-pass sev3={hp_accs[3]:.4f}  dark sev3={dark_accs[3]:.4f}")
    return tests


# ---------------------------------------------------------------------------
# Experiment 4: mechanism (adapted probes, margins, logit-lens, AOPC, n-shot)
# ---------------------------------------------------------------------------

def _patch_token_embeddings(model, tensors, device):
    """Return (n_tokens, d) patch-token embeddings (excluding CLS) from the final block."""
    import torch
    fe = model.prepare_tokens_with_masks
    x = fe(tensors)
    for blk in model.blocks:
        x = blk(x)
    x_norm = model.norm(x)
    return x_norm[:, 1:, :]  # drop CLS


def exp_mechanism(cfg, device, outdir):
    import torch

    d, m, p, st, mech = cfg["data"], cfg["model"], cfg["probe"], cfg["stats"], cfg["mechanism"]
    seed_everything(cfg["seed"])
    images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])

    E_by_sev = {}
    for s in range(6):
        E_by_sev[s] = extractor([low_light(img, s) for img in images])

    fixed_probe = fit_probe(E_by_sev[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

    # --- (a) severity-adapted probes
    adapted_rows = []
    for s in range(6):
        degraded_tr = [low_light(images[i], s) for i in idx_tr]
        adapted_probe = fit_probe(extractor(degraded_tr), labels[idx_tr],
                                  C=p["C"], max_iter=p["max_iter"])
        acc_fixed = (fixed_probe.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean()
        acc_adapted = (adapted_probe.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean()
        m_mean, _, m_corr = logit_margins(fixed_probe, E_by_sev[s][idx_te], labels[idx_te])
        adapted_rows.append({"severity": s, "acc_fixed": acc_fixed, "acc_adapted": acc_adapted,
                             "recovered_gap": acc_adapted - acc_fixed,
                             "mean_margin": m_mean, "margin_correct": m_corr})
        print(f"  sev {s}: fixed={acc_fixed:.4f} adapted={acc_adapted:.4f}")
    save_csv(adapted_rows, f"{outdir}/mechanism_adapted_probes.csv")

    # paired test: does adaptation help at severity 3?
    # (compare the same 300 test images under fixed vs adapted probe — NOT the
    # 50/50 label-flip masks the earlier draft accidentally compared)
    t3 = paired_permutation_test(
        (fixed_probe.predict(E_by_sev[3][idx_te]) == labels[idx_te]).astype(float),
        (adapted_probe.predict(E_by_sev[3][idx_te]) == labels[idx_te]).astype(float),
        seed=cfg["seed"])

    # --- (b) logit lens: keep CLS + top-k patch tokens by norm, re-normalize, re-probe
    preprocess = extractor.preprocess
    lens_rows = []
    for s in [0, 3, 5]:
        degraded = [low_light(images[i], s) for i in idx_te]
        tensors = torch.stack([preprocess(img) for img in degraded]).to(device)
        tokens = _patch_token_embeddings(model, tensors, device)  # (n, L, d)
        norms = tokens.norm(dim=-1)                                # (n, L)
        k = mech["logit_lens_topk"]
        topk = norms.topk(k, dim=1).indices                        # (n, k)
        n_examples = tokens.shape[0]
        E_lens = np.zeros((n_examples, model.embed_dim), dtype=np.float32)
        for b in range(0, n_examples, 64):
            tok_b = tokens[b:b + 64]
            idx_b = topk[b:b + 64]
            sel = torch.gather(tok_b, 1, idx_b.unsqueeze(-1).expand(-1, -1, tok_b.shape[-1]))
            pooled = sel.mean(dim=1)                               # (B, d)
            E_lens[b:b + 64] = pooled.cpu().numpy()
        acc_lens = (fixed_probe.predict(E_lens) == labels[idx_te]).mean()
        # sanity baseline: CLS-only at same severity
        acc_cls = (fixed_probe.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean()
        lens_rows.append({"severity": s, "acc_logit_lens_topk": acc_lens, "acc_cls": acc_cls})
        print(f"  logit lens sev {s}: topk={acc_lens:.4f} cls={acc_cls:.4f}")
    save_csv(lens_rows, f"{outdir}/mechanism_logit_lens.csv")

    # --- (c) AOPC: drop high-norm tokens, measure accuracy change
    aopc_rows = []
    for s in [0, 3, 5]:
        degraded = [low_light(images[i], s) for i in idx_te]
        tensors = torch.stack([preprocess(img) for img in degraded]).to(device)
        tokens = _patch_token_embeddings(model, tensors, device)
        norms = tokens.norm(dim=-1)
        base_acc = (fixed_probe.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean()
        for frac in mech["aopc_fractions"]:
            k = int(norms.shape[1] * frac)
            if k == 0:
                continue
            keep_mask = torch.ones_like(norms, dtype=torch.bool)
            keep_mask.scatter_(1, norms.topk(k, dim=1).indices, False)
            masked = tokens * keep_mask.unsqueeze(-1)
            E_masked = masked.mean(dim=1).cpu().numpy()
            acc = (fixed_probe.predict(E_masked) == labels[idx_te]).mean()
            aopc_rows.append({"severity": s, "drop_fraction": frac, "acc_after_drop": acc,
                              "delta_vs_full": acc - base_acc})
            print(f"  AOPC sev {s} drop {frac:.0%}: acc={acc:.4f} (delta {acc-base_acc:+.4f})")
    save_csv(aopc_rows, f"{outdir}/mechanism_aopc.csv")

    # --- (d) n-trainable readout: is the information there but hard to read?
    nshot_rows = []
    for n_pc in [1, 5, 20, 100]:
        row = {"n_per_class": n_pc}
        for s in [0, 3, 5]:
            accs = []
            for rep in range(5):  # 5 resamples of the n-shot support set
                probe_n = NTrainableProbe(n_pc, seed=cfg["seed"] + rep)
                tr_emb = E_by_sev[s][idx_tr]
                probe_n.fit(tr_emb, labels[idx_tr])
                accs.append((probe_n.predict(E_by_sev[s][idx_te]) == labels[idx_te]).mean())
            row[f"acc_sev{s}"] = float(np.mean(accs))
            row[f"acc_sev{s}_std"] = float(np.std(accs))
        nshot_rows.append(row)
        print(f"  n-shot {n_pc}/class: sev0={row['acc_sev0']:.4f} sev3={row['acc_sev3']:.4f} sev5={row['acc_sev5']:.4f}")
    save_csv(nshot_rows, f"{outdir}/mechanism_nshot.csv")

    save_json({"adapted_probe_sev3_paired_p": t3["p_value"],
               "adapted_probe_sev3_diff": t3["observed_diff"]},
              f"{outdir}/mechanism_adapted_test.json")

    write_manifest(outdir, CONFIG_PATH, "mechanism", {"test_fold": len(idx_te)})
    return adapted_rows, lens_rows, aopc_rows, nshot_rows


# ---------------------------------------------------------------------------
# Experiment 5: quantization (two-stage decomposition)
# ---------------------------------------------------------------------------

def exp_quantization(cfg, device, outdir):
    import torch

    d, m, p = cfg["data"], cfg["model"], cfg["probe"]
    seed_everything(cfg["seed"])
    images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
    probe = fit_probe(extractor([low_light(images[i], 0) for i in idx_tr]),
                      labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

    rows = []
    for s in range(6):
        float_imgs = [low_light_stage1(images[i], s) for i in idx_te]
        uint8_imgs = [low_light(images[i], s) for i in idx_te]
        E_float = extractor(float_imgs, float_input=True)
        E_uint8 = extractor(uint8_imgs)
        acc_float = (probe.predict(E_float) == labels[idx_te]).mean()
        acc_uint8 = (probe.predict(E_uint8) == labels[idx_te]).mean()
        rms = float(np.mean([quantization_gap_truth(images[i], s) for i in idx_te[:100]]))
        rows.append({"severity": s, "acc_float_stage1": acc_float, "acc_uint8_full": acc_uint8,
                     "irreversible_gap": acc_float - acc_uint8, "quantization_rms": rms,
                     "distinct_levels_mean": float(np.mean([
                         len(np.unique(low_light(images[i], s).ravel())) for i in idx_te[:100]]))})
        print(f"  sev {s}: float={acc_float:.4f} uint8={acc_uint8:.4f} gap={rows[-1]['irreversible_gap']:+.4f}")
    save_csv(rows, f"{outdir}/quantization.csv")

    # paired test at severity 3 and 5
    tests = {}
    for s in [3, 5]:
        float_imgs = [low_light_stage1(images[i], s) for i in idx_te]
        uint8_imgs = [low_light(images[i], s) for i in idx_te]
        cf = (probe.predict(extractor(float_imgs, float_input=True)) == labels[idx_te]).astype(float)
        cu = (probe.predict(extractor(uint8_imgs)) == labels[idx_te]).astype(float)
        tests[f"sev{s}"] = paired_permutation_test(cf, cu, seed=cfg["seed"])
    save_json(tests, f"{outdir}/quantization_tests.json")

    write_manifest(outdir, CONFIG_PATH, "quantization", {"test_fold": len(idx_te)})
    return rows, tests


# ---------------------------------------------------------------------------
# Experiment 6: LoRA ablation
# ---------------------------------------------------------------------------

def exp_lora_ablation(cfg, device, outdir):
    import torch
    from src.lora import extract_backbone, train_lora

    d, m, p, lc = cfg["data"], cfg["model"], cfg["probe"], cfg["lora"]
    seed_everything(lc["seed"])
    images, labels, _, train_images, train_labels, _ = load_cifar10_subsets(
        d["n_test"], d["n_train"], d["split_seed"])

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])

    # frozen baseline for reference
    base_model = load_dinov2(m["backbone"], device)
    base_extractor = EmbeddingExtractor(base_model, device)
    E0 = {s: base_extractor([low_light(img, s) for img in images]) for s in range(6)}
    base_probe = fit_probe(E0[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
    base_accs = [(base_probe.predict(E0[s][idx_te]) == labels[idx_te]).mean() for s in range(6)]
    print(f"  frozen baseline: sev0={base_accs[0]:.4f} sev3={base_accs[3]:.4f} sev5={base_accs[5]:.4f}")

    from torchvision.transforms import Compose, ToTensor, Resize, Normalize
    preprocess = Compose([ToTensor(), Resize((m["input_size"], m["input_size"]), antialias=True),
                          Normalize(mean=m["imagenet_mean"], std=m["imagenet_std"])])

    rows = []
    for rank in lc["ranks"]:
        for target in lc["targets"]:
            print(f"  === LoRA r={rank} {target} ===")
            # fresh pretrained weights each config so adapters don't accumulate
            fresh = load_dinov2(m["backbone"], device)
            clf = train_lora(fresh, train_images, train_labels, preprocess, lc,
                             rank, target, device, epochs=lc["epochs"])
            pooled = {s: extract_backbone(clf, [low_light(im, s) for im in images],
                                          preprocess, device) for s in range(6)}
            probe = fit_probe(pooled[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
            accs = [(probe.predict(pooled[s][idx_te]) == labels[idx_te]).mean() for s in range(6)]
            n_trainable = sum(pp.numel() for pp in clf.parameters() if pp.requires_grad)
            rows.append({"rank": rank, "target": target, "trainable_params": int(n_trainable),
                         "acc_sev0": accs[0], "acc_sev3": accs[3], "acc_sev5": accs[5],
                         "mean_acc": float(np.mean(accs)),
                         "delta_vs_frozen_mean": float(np.mean(accs) - np.mean(base_accs)),
                         "delta_vs_frozen_sev5": accs[5] - base_accs[5]})
            print(f"  -> sev0={accs[0]:.4f} sev3={accs[3]:.4f} sev5={accs[5]:.4f} "
                  f"mean={np.mean(accs):.4f} (delta vs frozen {rows[-1]['delta_vs_frozen_mean']:+.4f})")
            del clf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- anchored-adapter arm (only the OBJECTIVE differs; controlled
    # comparison for the objective-conflict claim). Runs at the mid-grid
    # configuration so the delta is attributable to the loss, not capacity.
    if lc.get("anchored", {}).get("enabled", False):
        from src.lora import extract_backbone, train_lora_anchored
        rank, target = lc["ranks"][len(lc["ranks"]) // 2], lc["targets"][-1]
        print(f"  === LoRA r={rank} {target} + invariance anchor ===")
        fresh = load_dinov2(m["backbone"], device)
        clf = train_lora_anchored(fresh, train_images, train_labels, preprocess, lc,
                                  rank, target, device, epochs=lc["epochs"])
        pooled = {s: extract_backbone(clf, [low_light(im, s) for im in images],
                                      preprocess, device) for s in range(6)}
        probe = fit_probe(pooled[0][idx_tr], labels[idx_tr], C=p["C"], max_iter=p["max_iter"])
        accs = [(probe.predict(pooled[s][idx_te]) == labels[idx_te]).mean() for s in range(6)]
        n_trainable = sum(pp.numel() for pp in clf.parameters() if pp.requires_grad)
        rows.append({"rank": rank, "target": f"{target}+anchor", "trainable_params": int(n_trainable),
                     "acc_sev0": accs[0], "acc_sev3": accs[3], "acc_sev5": accs[5],
                     "mean_acc": float(np.mean(accs)),
                     "delta_vs_frozen_mean": float(np.mean(accs) - np.mean(base_accs)),
                     "delta_vs_frozen_sev5": accs[5] - base_accs[5]})
        print(f"  -> sev0={accs[0]:.4f} sev3={accs[3]:.4f} sev5={accs[5]:.4f} "
              f"mean={np.mean(accs):.4f} (delta vs frozen {rows[-1]['delta_vs_frozen_mean']:+.4f})")
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_csv(rows, f"{outdir}/lora_ablation.csv")

    save_json([{"rank": r["rank"], "target": r["target"],
                "delta_sev5": r["delta_vs_frozen_sev5"],
                "delta_mean": r["delta_vs_frozen_mean"]} for r in rows],
              f"{outdir}/lora_deltas.json")

    write_manifest(outdir, CONFIG_PATH, "lora_ablation", {"n_train": d["n_train"],
                                                          "epochs": lc["epochs"]})
    return rows, base_accs


# ---------------------------------------------------------------------------
# Experiment 7: classical mitigation
# ---------------------------------------------------------------------------

def exp_mitigation(cfg, device, outdir):
    import torch

    d, m, p, mit = cfg["data"], cfg["model"], cfg["probe"], cfg["mitigation"]
    seed_everything(cfg["seed"])
    images, labels, _, _, _, _ = load_cifar10_subsets(d["n_test"], d["n_train"], d["split_seed"])
    model = load_dinov2(m["backbone"], device)
    extractor = EmbeddingExtractor(model, device)

    idx_tr, idx_te = stratified_split(labels, d["test_fraction"], d["split_seed"])
    probe = fit_probe(extractor([low_light(images[i], 0) for i in idx_tr]),
                      labels[idx_tr], C=p["C"], max_iter=p["max_iter"])

    rows = []
    correct_by_cond = {c: {} for c in ["none", "gain", "gamma", "clahe"]}
    for s in range(6):
        dark = [low_light(images[i], s) for i in idx_te]
        conds = {
            "none": dark,
            "gain": [simple_gain(img, s) for img in dark],
            "gamma": [gamma_correct(img, s, mit["gamma_base"]) for img in dark],
            "clahe": [clahe_enhance(img, mit["clahe_clip_limit"]) for img in dark],
        }
        row = {"severity": s}
        for name, restored in conds.items():
            E = extractor(restored)
            correct = (probe.predict(E) == labels[idx_te]).astype(float)
            correct_by_cond[name][s] = correct
            row[f"acc_{name}"] = correct.mean()
        rows.append(row)
        print(f"  sev {s}: " + "  ".join(f"{k}={row[f'acc_{k}']:.4f}" for k in conds))
    save_csv(rows, f"{outdir}/mitigation.csv")

    # paired tests vs 'none' at severities 2..5, BH-corrected
    tests, pvals = [], []
    for s in [2, 3, 4, 5]:
        for name in ["gain", "gamma", "clahe"]:
            t = paired_permutation_test(correct_by_cond[name][s], correct_by_cond["none"][s],
                                        seed=cfg["seed"])
            tests.append({"severity": s, "condition": name, "delta": t["observed_diff"],
                          "p_value": t["p_value"]})
            pvals.append(t["p_value"])
    adj = benjamini_hochberg(pvals)
    for i, t in enumerate(tests):
        t["p_bh"] = adj[i]
    save_json(tests, f"{outdir}/mitigation_tests.json")

    write_manifest(outdir, CONFIG_PATH, "mitigation", {"test_fold": len(idx_te)})
    return rows, tests


# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "main_curve": exp_main_curve,
    "cka": exp_cka,
    "frequency": exp_frequency,
    "mechanism": exp_mechanism,
    "quantization": exp_quantization,
    "lora_ablation": exp_lora_ablation,
    "mitigation": exp_mitigation,
    # supplementary experiments (torch-dependent; imported lazily)
    "efficiency": lambda cfg, dev, out: _supp().exp_efficiency(cfg, dev, out),
    "failure_analysis": lambda cfg, dev, out: _supp().exp_failure_analysis(cfg, dev, out),
    "seed_sensitivity": lambda cfg, dev, out: _supp().exp_seed_sensitivity(cfg, dev, out),
    "cross_dataset": lambda cfg, dev, out: _supp().exp_cross_dataset(cfg, dev, out),
}


def _supp():
    """Lazy import so CI (torch-free) can still import this module."""
    import src.supplementary as supplementary
    return supplementary


def main():
    ap = argparse.ArgumentParser(description="Run paper experiments")
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    device = get_device()
    ensure_dir(args.outdir)
    print(f"device={device}  config={args.config} (sha {sha256_of_file(args.config)[:12]})")

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        print(f"\n=== {name} ===")
        EXPERIMENTS[name](cfg, device, args.outdir)
    print("\ndone.")


if __name__ == "__main__":
    main()
