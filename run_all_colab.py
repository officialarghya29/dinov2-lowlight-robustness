"""
run_all_colab.py — one-shot GPU runner for the full paper experiment suite.

Upload to Colab (T4/V100/A100) and run:  %run run_all_colab.py

Produces results/ with:
  - one CSV + JSON per experiment (main_curve, cka, frequency, mechanism,
    quantization, lora_ablation, mitigation)
  - figures/ (publication plots)
  - paper_tables.tex (LaTeX tables, ready to paste into the paper)
  - manifest.jsonl (config hash, package versions, dataset indices)
  - RUN_SUMMARY.md (human-readable digest of every number produced)

Total runtime on T4: ~60-90 min (dominated by the 6-config LoRA ablation).
"""

import sys, os, subprocess, json, time

os.makedirs("/content/results", exist_ok=True)
os.chdir("/content")

print("=" * 70)
print("DINOv2 Low-Light Robustness — full paper suite")
print("=" * 70)

# --- deps ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-learn", "scikit-image", "matplotlib", "scipy", "pyyaml"], check=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t_start = time.time()

# --- run all 7 experiments through the paper harness ---
# (run_experiments.py must be uploaded alongside this file, with src/ and configs/)
subprocess.run([sys.executable, "run_experiments.py", "--experiment", "all",
                "--outdir", "/content/results"], check=True)

# ============================================================================
# Figures
# ============================================================================
FIGDIR = "/content/results/figures"
os.makedirs(FIGDIR, exist_ok=True)

def df(path):
    return pd.read_csv(path)

# Fig 2: main curve with CIs
mc = df("/content/results/main_curve.csv")
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.errorbar(mc.severity, mc.accuracy * 100, yerr=[(mc.accuracy - mc.ci_low) * 100,
             (mc.ci_high - mc.accuracy) * 100], marker="o", capsize=3,
             color="tab:blue", label="Probe accuracy")
ax2 = ax1.twinx()
ax2.plot(mc.severity, mc.mean_cosine_to_clean, marker="s", linestyle="--",
         color="tab:red", label="cos(clean, dark)")
ax1.set_xlabel("Low-light severity"); ax1.set_ylabel("Accuracy (%)")
ax2.set_ylabel("Mean cosine to clean embedding")
ax1.set_ylim(0, 100)
plt.title("Accuracy collapse and embedding drift under low light")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig2_main_curve.png", dpi=200); plt.close()

# Fig 3: CKA heatmap
cka = df("/content/results/cka_by_layer.csv")
mat = cka[[c for c in cka.columns if c.startswith("cka_sev")]].values
plt.figure(figsize=(6, 6))
plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label="Unbiased linear CKA to clean")
plt.xlabel("Severity"); plt.ylabel("Transformer block")
plt.title("Representational drift by layer")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig3_cka.png", dpi=200); plt.close()

# Fig 4: frequency
fq = df("/content/results/frequency_tests.csv")
plt.figure(figsize=(7, 4.5))
plt.plot(fq.severity, fq.acc_lowlight * 100, marker="^", linestyle="--", color="gray", label="Low light")
plt.plot(fq.severity, fq.acc_lowpass * 100, marker="o", label="Low-pass")
plt.plot(fq.severity, fq.acc_highpass * 100, marker="s", label="High-pass")
plt.xlabel("Severity"); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
plt.legend(); plt.title("Frequency-band controls")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig4_frequency.png", dpi=200); plt.close()

# Fig 5: mechanism (adapted probes + n-shot)
ap = df("/content/results/mechanism_adapted_probes.csv")
plt.figure(figsize=(7, 4.5))
plt.plot(ap.severity, ap.acc_fixed * 100, marker="o", label="Fixed probe (clean-trained)")
plt.plot(ap.severity, ap.acc_adapted * 100, marker="s", label="Severity-adapted probe")
plt.xlabel("Severity"); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
plt.legend(); plt.title("Mechanism: representation vs readout")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig5_mechanism.png", dpi=200); plt.close()

# Fig 6: quantization
qz = df("/content/results/quantization.csv")
plt.figure(figsize=(7, 4.5))
plt.plot(qz.severity, qz.acc_float_stage1 * 100, marker="o", label="Stage-1 only (float)")
plt.plot(qz.severity, qz.acc_uint8_full * 100, marker="s", label="Two-stage (uint8)")
plt.fill_between(qz.severity, qz.acc_uint8_full * 100, qz.acc_float_stage1 * 100,
                 alpha=0.15, color="tab:red", label="Irreversible gap")
plt.xlabel("Severity"); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
plt.legend(); plt.title("Quantization decomposition")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig6_quantization.png", dpi=200); plt.close()

# Fig 7: mitigation + LoRA Pareto
mit = df("/content/results/mitigation.csv")
plt.figure(figsize=(7, 4.5))
for cond, mk in [("none", "o"), ("gain", "s"), ("gamma", "^"), ("clahe", "d")]:
    plt.plot(mit.severity, mit[f"acc_{cond}"] * 100, marker=mk, label=cond)
plt.xlabel("Severity"); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
plt.legend(); plt.title("Zero-training mitigation")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig7_mitigation.png", dpi=200); plt.close()

try:
    la = df("/content/results/lora_ablation.csv")
    plt.figure(figsize=(7, 4.5))
    for tgt, mk in [("attn", "o"), ("attn_mlp", "s")]:
        sub = la[la.target == tgt]
        plt.scatter(sub.trainable_params, sub.mean_acc * 100, s=90, marker=mk, label=tgt)
        for _, r in sub.iterrows():
            plt.annotate(f"r={int(r['rank'])}", (r.trainable_params, r.mean_acc * 100),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xscale("log"); plt.xlabel("Trainable parameters (log)")
    plt.ylabel("Mean accuracy (%)"); plt.legend()
    plt.title("LoRA rank x target Pareto front")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig8_lora_pareto.png", dpi=200); plt.close()
except Exception as e:
    print("lora pareto plot skipped:", e)

# ============================================================================
# LaTeX tables (auto-generated from the CSVs)
# ============================================================================

def f3(x): return f"{100*x:.1f}"
def f4(x): return f"{x:.4f}"

lines = []
lines.append("% ---- Table 1: main curve (auto-generated) ----\n")
lines.append("\\begin{tabular}{cccccc}\n\\toprule\n")
lines.append("Sev. & $a$ & Acc (\\%) & 95\\% CI & Wilson 95\\% CI & cos to clean \\\\\n\\midrule\n")
for _, r in mc.iterrows():
    lines.append(f"{int(r.severity)} & {r.brightness_factor} & {f3(r.accuracy)} & "
                 f"[{f3(r.ci_low)}, {f3(r.ci_high)}] & [{f3(r.wilson_low)}, {f3(r.wilson_high)}] & "
                 f"{f4(r.mean_cosine_to_clean)} \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n")

lines.append("\n% ---- Table 2: quantization (auto-generated) ----\n")
lines.append("\\begin{tabular}{ccccc}\n\\toprule\n")
lines.append("Sev. & Float acc & uint8 acc & Gap & RMS \\\\\n\\midrule\n")
for _, r in qz.iterrows():
    lines.append(f"{int(r.severity)} & {f3(r.acc_float_stage1)} & {f3(r.acc_uint8_full)} & "
                 f"{100*(r.acc_float_stage1-r.acc_uint8_full):+.1f} & {r.quantization_rms:.2f} \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n")

lines.append("\n% ---- Table 4: LoRA ablation (auto-generated) ----\n")
try:
    lines.append("\\begin{tabular}{ccccc}\n\\toprule\n")
    lines.append("Rank & Target & Params & Mean acc & $\\Delta$ vs frozen \\\\\n\\midrule\n")
    for _, r in la.iterrows():
        lines.append(f"{int(r['rank'])} & {r.target} & {int(r.trainable_params):,} & "
                     f"{f3(r.mean_acc)} & {100*r.delta_vs_frozen_mean:+.1f} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n")
except NameError:
    lines.append("% lora_ablation.csv missing\n")

with open("/content/results/paper_tables.tex", "w") as f:
    f.writelines(lines)

# ============================================================================
# Human-readable digest
# ============================================================================
summary = []
summary.append("# RUN SUMMARY\n")
summary.append(f"Total runtime: {(time.time()-t_start)/60:.1f} min\n")
summary.append("\n## Main curve\n```\n" + mc.to_string(index=False) + "\n```")
summary.append("\n## Null tests (linearity / midpoint symmetry)\n```")
summary.append(json.dumps(json.load(open("/content/results/main_curve_nulltests.json")), indent=2))
summary.append("\n```")
summary.append("\n## Quantization decomposition\n```\n" + qz.to_string(index=False) + "\n```")
try:
    summary.append("\n## LoRA ablation\n```\n" + la.to_string(index=False) + "\n```")
except NameError:
    pass
summary.append("\n## Mitigation\n```\n" + mit.to_string(index=False) + "\n```")
with open("/content/results/RUN_SUMMARY.md", "w") as f:
    f.write("\n".join(summary))

print("\n" + "=" * 70)
print("ALL DONE. Artifacts in /content/results/:")
for root, _, files in os.walk("/content/results"):
    for fn in sorted(files):
        print("  ", os.path.relpath(os.path.join(root, fn), "/content/results"))
print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")
print("=" * 70)
