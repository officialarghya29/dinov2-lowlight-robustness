"""Publication-quality figures for the paper (PDF, colorblind-safe palette).

    python3 make_paper_figures.py --results results_pilot --out paper/figures

PDF vector output for camera-ready; falls back to including PNGs in the repo
for quick GitHub preview (both are written).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colorblind-safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def save(fig, out, name):
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pilot")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.results

    def df(name):
        return pd.read_csv(os.path.join(R, name))

    # ---- Fig 1: main curve + cosine (dual axis) ----
    mc = df("main_curve.csv")
    fig, ax1 = plt.subplots(figsize=(4.6, 3.2))
    ax1.errorbar(mc.severity, mc.accuracy * 100,
                 yerr=[(mc.accuracy - mc.ci_low) * 100, (mc.ci_high - mc.accuracy) * 100],
                 marker="o", capsize=3, color=C["blue"], lw=1.6, label="Probe accuracy")
    ax2 = ax1.twinx()
    ax2.plot(mc.severity, mc.mean_cosine_to_clean, marker="s", linestyle="--",
             color=C["red"], lw=1.6, label=r"cos(clean, dark)")
    ax2.axhline(1.0, color=C["grey"], lw=0.6, ls=":")
    ax1.set_xlabel("Low-light severity $s$", fontsize=10)
    ax1.set_ylabel("Accuracy (\\%)", fontsize=10)
    ax2.set_ylabel("Cosine to clean", fontsize=10, color=C["red"])
    ax2.tick_params(axis="y", labelcolor=C["red"])
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(6))
    style_axes(ax1)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, args.out, "fig_main_curve")

    # ---- Fig 2: CKA by depth (ViT-S + ViT-B overlay when available) ----
    ck = df("cka_by_layer.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ck.block, ck.cka_sev3, marker="o", ms=4, color=C["orange"],
            lw=1.6, label="ViT-S/14, $s{=}3$")
    ax.plot(ck.block, ck.cka_sev5, marker="s", ms=4, color=C["blue"],
            lw=1.6, label="ViT-S/14, $s{=}5$")
    p_vb = os.path.join(R, "vitb_cka_by_layer.csv")
    if os.path.exists(p_vb):
        vb = pd.read_csv(p_vb)
        ax.plot(vb.block, vb.cka_sev5, marker="^", ms=4, linestyle="--",
                color=C["green"], lw=1.6, label="ViT-B/14, $s{=}5$")
    ax.axhline(1.0, color=C["grey"], lw=0.6, ls=":")
    ax.set_xlabel("Transformer block", fontsize=10)
    ax.set_ylabel("Unbiased CKA to clean", fontsize=10)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, args.out, "fig_cka_depth")

    # ---- Fig 3: frequency controls ----
    fq = df("frequency_tests.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(fq.severity, fq.acc_lowlight * 100, marker="^", linestyle="--",
            color=C["grey"], lw=1.6, label="Low light")
    ax.plot(fq.severity, fq.acc_lowpass * 100, marker="o", color=C["blue"],
            lw=1.6, label="Low-pass")
    ax.plot(fq.severity, fq.acc_highpass * 100, marker="s", color=C["red"],
            lw=1.6, label="High-pass")
    ax.set_xlabel("Severity $s$", fontsize=10)
    ax.set_ylabel("Accuracy (\\%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, 6))
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, args.out, "fig_frequency")

    # ---- Fig 4: readout repair + collapse (two panels) ----
    ap_ = df("mechanism_adapted_probes.csv")
    pc = df("prediction_collapse.csv")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axa.plot(ap_.severity, ap_.acc_fixed * 100, marker="o", ms=4,
             color=C["blue"], lw=1.6, label="Fixed probe")
    axa.plot(ap_.severity, ap_.acc_adapted * 100, marker="s", ms=4,
             color=C["orange"], lw=1.6, label="Severity-adapted")
    axa.set_xlabel("Severity $s$", fontsize=10)
    axa.set_ylabel("Accuracy (\\%)", fontsize=10)
    axa.set_ylim(0, 100)
    axa.set_xticks(range(6))
    style_axes(axa)
    axa.legend(fontsize=8, frameon=False)
    axa.set_title("(a) Readout repair", fontsize=10)
    axb.plot(pc.severity, pc.entropy_ratio_vs_uniform, marker="o", ms=4,
             color=C["purple"], lw=1.6)
    axb.set_xlabel("Severity $s$", fontsize=10)
    axb.set_ylabel("Prediction entropy / $\\ln 10$", fontsize=10)
    axb.set_ylim(0, 1.05)
    axb.set_xticks(range(6))
    style_axes(axb)
    axb.set_title("(b) Readout degeneracy", fontsize=10)
    fig.tight_layout()
    save(fig, args.out, "fig_mechanism")

    # ---- Fig 5: mitigation ----
    mit = df("mitigation.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for cond, mk, col, lbl in [("none", "o", C["grey"], "None"),
                               ("gain", "s", C["blue"], "Gain $I/a_s$"),
                               ("gamma", "^", C["green"], "Gamma"),
                               ("clahe", "d", C["red"], "CLAHE")]:
        ax.plot(mit.severity, mit[f"acc_{cond}"] * 100, marker=mk, ms=4,
                color=col, lw=1.6, label=lbl)
    ax.set_xlabel("Severity $s$", fontsize=10)
    ax.set_ylabel("Accuracy (\\%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(6))
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, args.out, "fig_mitigation")

    print(f"figures written to {args.out}/ (pdf + png)")


if __name__ == "__main__":
    main()
