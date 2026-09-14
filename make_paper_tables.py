"""Generate paper/results/*.tex tables + numbers.tex from a results directory.

    python3 make_paper_tables.py --results results_pilot --out paper/results
    python3 make_paper_tables.py --results results --out paper/results   # full GPU run

Tables follow the same schema for pilot and full runs, so the paper regenerates
identically from either scale (with a scale note baked into numbers.tex).
"""

import argparse
import csv
import json
import os


def read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x, digits=1):
    return f"{100 * float(x):.{digits}f}"


def f(x, digits=3):
    return f"{float(x):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = args.results

    # ---------- Table: main curve ----------
    mc = read(f"{R}/main_curve.csv")
    if mc:
        lines = ["\\begin{tabular}{cccccc}", "\\toprule",
                 "Sev. & $a$ & Acc (\\%) & 95\\% CI & cos to clean & margin \\\\", "\\midrule"]
        for r in mc:
            lines.append(f"{r['severity']} & {r['brightness_factor']} & {pct(r['accuracy'])} & "
                         f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}] & "
                         f"{f(r['mean_cosine_to_clean'], 2)} & {f(r['mean_logit_margin'], 2)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_main.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: frequency dissociation ----------
    fq = read(f"{R}/frequency_tests.csv")
    if fq:
        lines = ["\\begin{tabular}{ccccc}", "\\toprule",
                 "Sev. & Low-light (\\%) & Low-pass (\\%) & High-pass (\\%) & $p$ (dark vs LP) \\\\", "\\midrule"]
        for r in fq:
            lines.append(f"{r['severity']} & {pct(r['acc_lowlight'])} & {pct(r['acc_lowpass'])} & "
                         f"{pct(r['acc_highpass'])} & {f(r['p_dark_vs_lowpass'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_freq.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: CKA by layer depth ----------
    ck = read(f"{R}/cka_by_layer.csv")
    if ck:
        lines = ["\\begin{tabular}{cccc}", "\\toprule",
                 "Block & CKA sev3 & CKA sev5 & Drop (0$\\to$5) \\\\", "\\midrule"]
        for r in ck:
            lines.append(f"{r['block']} & {f(r['cka_sev3'])} & {f(r['cka_sev5'])} & "
                         f"{f(r['drop_0_to_5'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_cka.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: mechanism (fixed vs severity-adapted probe) ----------
    mech = read(f"{R}/mechanism_adapted_probes.csv")
    if mech:
        lines = ["\\begin{tabular}{cccc}", "\\toprule",
                 "Sev. & Fixed probe (\\%) & Adapted probe (\\%) & Recovery (pp) \\\\", "\\midrule"]
        for r in mech:
            lines.append(f"{r['severity']} & {pct(r['acc_fixed'])} & {pct(r['acc_adapted'])} & "
                         f"{100 * float(r['recovered_gap']):+.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_mech.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: quantization ----------
    qz = read(f"{R}/quantization.csv")
    if qz:
        lines = ["\\begin{tabular}{ccccc}", "\\toprule",
                 "Sev. & Float (\\%) & uint8 (\\%) & Gap (pp) \\\\", "\\midrule"]
        for r in qz:
            gap = 100 * (float(r["acc_float_stage1"]) - float(r["acc_uint8_full"]))
            lines.append(f"{r['severity']} & {pct(r['acc_float_stage1'])} & {pct(r['acc_uint8_full'])} & {gap:+.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_quant.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: mitigation ----------
    mit = read(f"{R}/mitigation.csv")
    if mit:
        lines = ["\\begin{tabular}{ccccc}", "\\toprule",
                 "Sev. & None (\\%) & Gain (\\%) & Gamma (\\%) & CLAHE (\\%) \\\\", "\\midrule"]
        for r in mit:
            lines.append(f"{r['severity']} & {pct(r['acc_none'])} & {pct(r['acc_gain'])} & "
                         f"{pct(r['acc_gamma'])} & {pct(r['acc_clahe'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_mit.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: LoRA ablation (full runs only) ----------
    la = read(f"{R}/lora_ablation.csv")
    if la:
        lines = ["\\begin{tabular}{ccccc}", "\\toprule",
                 "Rank & Target & Params & Mean acc (\\%) & $\\Delta$ vs frozen (pp) \\\\", "\\midrule"]
        for r in la:
            lines.append(f"{r['rank']} & {r['target']} & {int(r['trainable_params']):,} & "
                         f"{pct(r['mean_acc'])} & {100 * float(r['delta_vs_frozen_mean']):+.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(f"{args.out}/table_lora.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: failure analysis (per class) ----------
    fb = read(f"{R}/failure_by_class.csv")
    if fb:
        lines = ["\\begin{tabular}{lcccc}", "\\toprule",
                 "Class & Clean & Sev 1 & Sev 3 & Sev 5 (\\%) \\\\", "\\midrule"]
        for r in fb:
            lines.append(f"{r['class']} & {pct(r['acc_clean'], 0)} & {pct(r['acc_sev1'], 0)} & "
                         f"{pct(r['acc_sev3'], 0)} & {pct(r['acc_sev5'], 0)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_failure.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: seed / hyperparameter sensitivity ----------
    ss = read_json(f"{R}/seed_sensitivity_summary.json")
    if ss:
        lines = ["\\begin{tabular}{cc}", "\\toprule",
                 "Severity & Accuracy (mean $\\pm$ std, $n$=9) \\\\", "\\midrule"]
        for k in sorted(ss, key=lambda x: int(x)):
            v = ss[k]
            lines.append(f"{k} & {pct(v['mean'])} $\\pm$ {pct(v['std'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(f"{args.out}/table_sensitivity.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: cross-dataset transfer ----------
    cd = read(f"{R}/cross_dataset.csv")
    if cd:
        lines = ["\\begin{tabular}{ccc}", "\\toprule",
                 "Severity & CIFAR-10 test (\\%) & STL-10 zero-shot (\\%) \\\\", "\\midrule"]
        cifar = {int(r["severity"]): r for r in (mc or [])}
        for r in cd:
            s = int(r["severity"])
            cif = pct(cifar[s]["accuracy"]) if s in cifar else "--"
            lines.append(f"{s} & {cif} & {pct(r['accuracy'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(f"{args.out}/table_cross.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: ViT-B generality (supplementary) ----------
    vbc = read(f"{R}/vitb_main_curve.csv")
    if vbc:
        lines = ["\\begin{tabular}{ccc}", "\\toprule",
                 "Sev. & ViT-B/14 acc (\\%) & cos to clean \\\\", "\\midrule"]
        for r in vbc:
            lines.append(f"{r['severity']} & {pct(r['accuracy'])} & "
                         f"{f(r['mean_cosine_to_clean'], 2)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(f"{args.out}/table_vitb_curve.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")
    vbd = read(f"{R}/vitb_cka_by_layer.csv")
    if vbd:
        lines = ["\\begin{tabular}{cccc}", "\\toprule",
                 "Block & CKA sev3 & CKA sev5 & Drop (0$\\to$5) \\\\", "\\midrule"]
        for r in vbd:
            lines.append(f"{r['block']} & {f(r['cka_sev3'])} & {f(r['cka_sev5'])} & "
                         f"{f(r['drop_0_to_5'])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_vitb_cka.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- Table: efficiency ----------
    ef = read(f"{R}/efficiency.csv")
    if ef:
        lines = ["\\begin{tabular}{lcccc}", "\\toprule",
                 "Arm & Backbone (M) & Trainable (M) & Latency (ms) & Acc (dark, \\%) \\\\", "\\midrule"]
        for r in ef:
            lat = f"{float(r['latency_ms']):.1f}" if r["latency_ms"] != "" else "--"
            acc = f"{pct(r['acc_dark_mean'])}" if r.get("acc_dark_mean") not in (None, "") else "--"
            arm = str(r["arm"]).replace("_", "\\_")
            lines.append(f"{arm} & {f(r['backbone_params_M'], 1)} & {f(r['trainable_params_M'], 2)} & "
                         f"{lat} & {acc} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines = ["\\resizebox{\\linewidth}{!}{%"] + lines + ["}"]
        with open(f"{args.out}/table_eff.tex", "w") as f_out:
            f_out.write("\n".join(lines) + "\n")

    # ---------- numbers.tex ----------
    n = read(f"{R}/main_curve.csv")
    cka = read_json(f"{R}/cka_summary.json")
    scale_note = ("Full-scale run" if "results_pilot" not in R else
                  "PILOT SCALE (reduced n; see configs/pilot_cpu.yaml) — provisional")
    with open(f"{args.out}/numbers.tex", "w") as f_out:
        f_out.write(f"% AUTO-GENERATED by make_paper_tables.py from {R}/ — {scale_note}\n")
        if n:
            clean = float(n[0]["accuracy"])
            floor = float(n[5]["accuracy"])
            cos1 = float(n[1]["mean_cosine_to_clean"])
            f_out.write(f"\\newcommand{{\\AccClean}}{{{pct(clean)}\\%}}\n")
            f_out.write(f"\\newcommand{{\\AccFloor}}{{{pct(floor)}\\%}}\n")
            f_out.write(f"\\newcommand{{\\AccCliffDroppp}}{{{100 * (clean - float(n[4]['accuracy'])):.0f}\\,pp}}\n")
            f_out.write(f"\\newcommand{{\\CosSevOne}}{{{cos1:.2f}}}\n")
            if len(n) > 5:
                f_out.write(f"\\newcommand{{\\GraceBump}}{{{100 * (float(n[1]['accuracy']) - float(n[0]['accuracy'])):+.1f}\\,pp}}\n")
                f_out.write(f"\\newcommand{{\\PRFloorRatio}}{{{float(n[5]['pr_ratio']):.2f}}}\n")
        nt = read_json(f"{R}/main_curve_nulltests.json")
        if nt:
            f_out.write(f"\\newcommand{{\\PNotNull}}{{{f(nt['p_not_linear'])}}}\n")
            f_out.write(f"\\newcommand{{\\PMidNotNull}}{{{f(nt['p_midpoint_asymmetric'])}}}\n")
        if cka:
            f_out.write(f"\\newcommand{{\\LateDrift}}{{{f(cka['late_mean_8_11'])}}}\n")
            f_out.write(f"\\newcommand{{\\EarlyDrift}}{{{f(cka['early_mean_0_3'])}}}\n")
        q5 = read_json(f"{R}/paired_tests.json")
        if q5 and "quant_sev5_p" in q5:
            f_out.write(f"\\newcommand{{\\QuantGapFiveP}}{{{f(q5['quant_sev5_p'])}}}\n")
        mech = read(f"{R}/mechanism_adapted_probes.csv")
        if mech and len(mech) > 3:
            gap3 = 100 * (float(mech[3]["acc_adapted"]) - float(mech[3]["acc_fixed"]))
            f_out.write(f"\\newcommand{{\\AdaptedGapThree}}{{{gap3:+.1f}\\,pp}}\n")
        if mech and len(mech) > 4:
            gap4 = 100 * (float(mech[4]["acc_adapted"]) - float(mech[4]["acc_fixed"]))
            f_out.write(f"\\newcommand{{\\AdaptedGapFour}}{{{gap4:+.1f}\\,pp}}\n")
        ft = read(f"{R}/frequency_tests.csv")
        if ft:
            sev3 = next((r for r in ft if int(r["severity"]) == 3), None)
            if sev3:
                f_out.write(f"\\newcommand{{\\AccDarkThree}}{{{pct(sev3['acc_lowlight'])}\\%}}\n")
                f_out.write(f"\\newcommand{{\\AccHighpassThree}}{{{pct(sev3['acc_highpass'])}\\%}}\n")
        fq = read_json(f"{R}/failure_summary.json")
        if fq and "persistence_sev1_to_max" in fq:
            f_out.write(f"\\newcommand{{\\FailPersist}}{{{fq['persistence_sev1_to_max']:.2f}}}\n")
        cdx = read(f"{R}/cross_dataset.csv")
        if cdx and len(cdx) > 5:
            f_out.write(f"\\newcommand{{\\StlClean}}{{{pct(cdx[0]['accuracy'])}\\%}}\n")
            f_out.write(f"\\newcommand{{\\StlFloor}}{{{pct(cdx[-1]['accuracy'])}\\%}}\n")
        pc = read(f"{R}/prediction_collapse.csv")
        if pc and len(pc) > 5:
            f_out.write(f"\\newcommand{{\\FrogShareFive}}{{{pct(pc[5]['top1_share'])}\\%}}\n")
            f_out.write(f"\\newcommand{{\\EntropyRatioFive}}{{{float(pc[5]['entropy_ratio_vs_uniform']):.2f}}}\n")
            f_out.write(f"\\newcommand{{\\EntropyRatioThree}}{{{float(pc[3]['entropy_ratio_vs_uniform']):.2f}}}\n")
        if mech:
            acc4 = next((r for r in mech if int(r["severity"]) == 4), None)
            if acc4:
                f_out.write(f"\\newcommand{{\\AccAdaptedFour}}{{{pct(acc4['acc_adapted'])}\\%}}\n")
        mitr = read(f"{R}/mitigation.csv")
        if mitr:
            g4 = next((r for r in mitr if int(r["severity"]) == 4), None)
            c0 = next((r for r in mitr if int(r["severity"]) == 0), None)
            if g4:
                f_out.write(f"\\newcommand{{\\AccGainFour}}{{{pct(g4['acc_gain'])}\\%}}\n")
            if c0:
                f_out.write(f"\\newcommand{{\\AccClaheClean}}{{{pct(c0['acc_clahe'])}\\%}}\n")
        if cka and "late_minus_early" in cka:
            f_out.write(f"\\newcommand{{\\LateMinusEarly}}{{{cka['late_minus_early']:+.2f}}}\n")
        # backbone-generality macros (present once run_vitb_generality.py ran)
        vb = read_json(f"{R}/vitb_cka_summary.json")
        vbm = read(f"{R}/vitb_main_curve.csv")
        if vb:
            f_out.write(f"\\newcommand{{\\ViTBLateMinusEarly}}{{{vb['late_minus_early']:+.2f}}}\n")
            f_out.write(f"\\newcommand{{\\ViTBLate}}{{{f(vb['late_mean_8_11'])}}}\n")
            f_out.write(f"\\newcommand{{\\ViTBEarly}}{{{f(vb['early_mean_0_3'])}}}\n")
        if vbm and len(vbm) > 5:
            f_out.write(f"\\newcommand{{\\ViTBAccClean}}{{{pct(vbm[0]['accuracy'])}\\%}}\n")
            f_out.write(f"\\newcommand{{\\ViTBAccFloor}}{{{pct(vbm[5]['accuracy'])}\\%}}\n")

    print(f"tables written to {args.out}/")


if __name__ == "__main__":
    main()
