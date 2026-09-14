"""Two-panel PDF figure for the paper's Mitigation section (followup deliverable).

Left panel: per-severity accuracy delta vs. the no-mitigation baseline for
every zero-training mitigation (readout repair dominates; gain/gamma cluster
near zero; CLAHE goes deeply negative).

Right panel: the decisive corollary-1 comparison at the cliff severities —
readout repair vs. the best input-space enhancement (gain), with exact
permutation p-values.

All numbers come from committed artifacts in results_pilot/:
  - mechanism_adapted_probes.csv (acc_fixed / acc_adapted per severity)
  - mitigation.csv               (acc_none / acc_gain / acc_gamma / acc_clahe)
  - corollaries.json             (corollary1_readout_vs_enhancement p-values)

Usage:  python3 make_mitigation_figure.py [--results results_pilot]
Output: paper/figures/fig_readout_vs_enhancement.pdf
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette
C_REPAIR = "#0072B2"   # blue
C_GAIN = "#009E73"     # green
C_GAMMA = "#56B4E9"    # sky blue
C_CLAHE = "#D55E00"    # vermillion
C_NEUTRAL = "#999999"


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_pilot")
    ap.add_argument("--out", default="paper/figures/fig_readout_vs_enhancement.pdf")
    args = ap.parse_args()

    # --- readout repair: fixed vs adapted probe ---
    sev, repair_delta = [], []
    for r in read_csv_rows(os.path.join(args.results, "mechanism_adapted_probes.csv")):
        sev.append(int(r["severity"]))
        repair_delta.append(100 * (float(r["acc_adapted"]) - float(r["acc_fixed"])))

    # --- enhancement arms: deltas vs acc_none (same baseline by construction) ---
    arms = {"gain": {}, "gamma": {}, "clahe": {}}
    for r in read_csv_rows(os.path.join(args.results, "mitigation.csv")):
        s = int(r["severity"])
        for a in arms:
            arms[a][s] = 100 * (float(r[f"acc_{a}"]) - float(r["acc_none"]))

    # --- corollary-1 exact p-values (adapted probe vs gain) ---
    with open(os.path.join(args.results, "corollaries.json")) as f:
        cor = json.load(f)["corollary1_readout_vs_enhancement"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

    # ---------------- left panel: deltas across the ladder ----------------
    ax1.axhline(0, color=C_NEUTRAL, lw=0.8, zorder=1)
    x = np.array(sev)
    ax1.plot(x, [arms["gain"][s] for s in sev], marker="o", color=C_GAIN,
             lw=1.6, ms=4.5, label="gain boost")
    ax1.plot(x, [arms["gamma"][s] for s in sev], marker="v", ls="--",
             color=C_GAMMA, lw=1.6, ms=4.5, label="gamma boost")
    ax1.plot(x, [arms["clahe"][s] for s in sev], marker="x", ls=":",
             color=C_CLAHE, lw=1.6, ms=4.5, label="CLAHE")
    ax1.plot(x, repair_delta, marker="s", color=C_REPAIR, lw=2.2, ms=5,
             label="readout repair", zorder=4)
    ax1.set_xlabel("corruption severity")
    ax1.set_ylabel("accuracy delta vs. no mitigation (pp)")
    ax1.set_title("(a) Zero-training mitigations", fontsize=10)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.spines[["top", "right"]].set_visible(False)

    # ---------------- right panel: decisive comparison ----------------
    cliff = [s for s in (4, 5) if s in sev]
    xs = np.arange(len(cliff))
    w = 0.36

    repair_at = [repair_delta[sev.index(s)] for s in cliff]
    gain_at = [arms["gain"][s] for s in cliff]

    b1 = ax2.bar(xs - w / 2, repair_at, w, color=C_REPAIR, label="readout repair")
    b2 = ax2.bar(xs + w / 2, gain_at, w, color=C_GAIN, label="gain boost (best input fix)")
    for xi, s in enumerate(cliff):
        p = cor[f"sev{s}"]["p_adapted_vs_gain"]
        label = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        ax2.text(xi, max(repair_at[xi], gain_at[xi], 0) + 1.5, label,
                 ha="center", fontsize=8, color="#333333")
    ax2.bar_label(b1, fmt="%.1f", fontsize=7, padding=1)
    ax2.bar_label(b2, fmt="%.1f", fontsize=7, padding=1)
    ax2.axhline(0, color=C_NEUTRAL, lw=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"severity {s}" for s in cliff], fontsize=9)
    ax2.set_ylabel("accuracy delta vs. no mitigation (pp)")
    ax2.set_title("(b) Readout repair vs. best input-space fix", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {args.out} (+png)")
    print("  cliff:", cliff, "| repair pp:", [f"{v:.1f}" for v in repair_at],
          "| gain pp:", [f"{v:.1f}" for v in gain_at])


if __name__ == "__main__":
    main()
