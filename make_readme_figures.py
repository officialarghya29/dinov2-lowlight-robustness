"""Generate README/paper figures from a results directory (CPU-friendly).

    python3 make_readme_figures.py --results results_pilot --out docs/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pilot")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.out and args.results

    def df(name):
        return pd.read_csv(os.path.join(args.results, name))

    # Fig A: main curve + cosine + margin
    mc = df("main_curve.csv")
    fig, ax1 = plt.subplots(figsize=(7, 4.4))
    ax1.errorbar(mc.severity, mc.accuracy * 100,
                 yerr=[(mc.accuracy - mc.ci_low) * 100, (mc.ci_high - mc.accuracy) * 100],
                 marker="o", capsize=3, color="tab:blue", label="Probe accuracy")
    ax2 = ax1.twinx()
    ax2.plot(mc.severity, mc.mean_cosine_to_clean, marker="s", linestyle="--",
             color="tab:red", label="cos(clean, dark)")
    ax2.plot(mc.severity, mc.margin_correct_only / mc.margin_correct_only.max() * 100,
             marker="^", linestyle=":", color="tab:green", alpha=0.8,
             label="margin (correct, norm.)")
    ax1.set_xlabel("Low-light severity")
    ax1.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Cosine / normalized margin")
    ax1.set_ylim(0, 100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    plt.title("Accuracy collapse and embedding drift (DINOv2 ViT-S/14)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_main_curve.png", dpi=150)
    plt.close(fig)

    # Fig B: CKA by depth
    ck = df("cka_by_layer.csv")
    plt.figure(figsize=(6.5, 4.2))
    for col, lbl in [("cka_sev3", "sev 3"), ("cka_sev5", "sev 5")]:
        plt.plot(ck.block, ck[col], marker="o", label=f"CKA to clean, {lbl}")
    plt.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    plt.xlabel("Transformer block")
    plt.ylabel("Unbiased linear CKA to clean")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.title("Representational drift concentrates in late blocks")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_cka.png", dpi=150)
    plt.close()

    # Fig C: frequency dissociation
    fq = df("frequency_tests.csv")
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(fq.severity, fq.acc_lowlight * 100, marker="^", linestyle="--",
             color="gray", label="Low light")
    plt.plot(fq.severity, fq.acc_lowpass * 100, marker="o", label="Low-pass")
    plt.plot(fq.severity, fq.acc_highpass * 100, marker="s", label="High-pass")
    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Darkness is not a frequency artifact")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_frequency.png", dpi=150)
    plt.close()

    # Fig D: mechanism — fixed vs adapted probe
    ap_ = df("mechanism_adapted_probes.csv")
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(ap_.severity, ap_.acc_fixed * 100, marker="o", label="Fixed probe (clean-trained)")
    plt.plot(ap_.severity, ap_.acc_adapted * 100, marker="s", label="Severity-adapted probe")
    for _, r in ap_.iterrows():
        if r["recovered_gap"] > 0.02:
            plt.annotate(f"+{100 * r['recovered_gap']:.0f}pp",
                         (r.severity, r.acc_adapted * 100),
                         textcoords="offset points", xytext=(4, 6), fontsize=8,
                         color="tab:orange")
    plt.xlabel("Low-light severity")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Readout failure dominates at high severity")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_mechanism.png", dpi=150)
    plt.close()

    # Fig E: mitigation
    mit = df("mitigation.csv")
    plt.figure(figsize=(6.5, 4.2))
    for cond, mk in [("none", "o"), ("gain", "s"), ("gamma", "^"), ("clahe", "d")]:
        plt.plot(mit.severity, mit[f"acc_{cond}"] * 100, marker=mk, label=cond)
    plt.xlabel("Low-light severity")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Zero-training mitigation: classical enhancement")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_mitigation.png", dpi=150)
    plt.close()

    # Fig F: failure by class (if present)
    p = os.path.join(args.results, "failure_by_class.csv")
    if os.path.exists(p):
        fb = pd.read_csv(p)
        plt.figure(figsize=(8, 4.4))
        for col, lbl in [("acc_clean", "clean"), ("acc_sev1", "sev 1"),
                         ("acc_sev3", "sev 3"), ("acc_sev5", "sev 5")]:
            plt.plot(fb["class"], fb[col] * 100, marker="o", label=lbl)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.title("Per-class breakdown of the low-light collapse")
        plt.tight_layout()
        plt.savefig(f"{args.out}/fig_failure_by_class.png", dpi=150)
        plt.close()

    # Fig G: seed sensitivity (if present)
    p = os.path.join(args.results, "seed_sensitivity.csv")
    if os.path.exists(p):
        ss = pd.read_csv(p)
        g = ss.groupby("severity")["accuracy"]
        plt.figure(figsize=(6.5, 4.2))
        plt.errorbar(g.mean().index, g.mean() * 100, yerr=g.std() * 100,
                     marker="o", capsize=3, color="tab:blue")
        plt.xlabel("Low-light severity")
        plt.ylabel("Accuracy (%)")
        plt.title("Degradation across 3 seeds x 3 probe-C (mean +/- std)")
        plt.tight_layout()
        plt.savefig(f"{args.out}/fig_sensitivity.png", dpi=150)
        plt.close()

    # Fig H: cross-dataset (if present)
    p = os.path.join(args.results, "cross_dataset.csv")
    if os.path.exists(p):
        cd = pd.read_csv(p)
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(mc.severity, mc.accuracy * 100, marker="o", label="CIFAR-10 (test fold)")
        plt.plot(cd.severity, cd.accuracy * 100, marker="s", linestyle="--",
                 label="STL-10 (zero-shot)")
        plt.xlabel("Low-light severity")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.title("Cross-dataset transfer of the degradation curve")
        plt.tight_layout()
        plt.savefig(f"{args.out}/fig_cross_dataset.png", dpi=150)
        plt.close()

    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
