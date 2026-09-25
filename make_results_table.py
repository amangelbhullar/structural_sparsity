#!/usr/bin/env python3
import argparse, sys
import pandas as pd

DATASET_ORDER = ["FB15k-237", "WN18RR", "YAGO3-10", "CoDEx-S", "CoDEx-M", "CoDEx-L"]
MODEL_ORDER = ["HS-GNN", "AS-GNN", "ASR-GNN"]
MODEL_RENAME = {"SD-GNN":"HS-GNN","SDGNN":"HS-GNN","HS-GNN":"HS-GNN",
                "AS-GNN":"AS-GNN","ASGNN":"AS-GNN","ASR-GNN":"ASR-GNN","ASRGNN":"ASR-GNN"}
DATASET_RENAME = {"fb15k237":"FB15k-237","FB15k-237":"FB15k-237","wn18rr":"WN18RR","WN18RR":"WN18RR",
                   "yago310":"YAGO3-10","YAGO3-10":"YAGO3-10","codex-s":"CoDEx-S","CoDEx-S":"CoDEx-S",
                   "codex-m":"CoDEx-M","CoDEx-M":"CoDEx-M","codex-l":"CoDEx-L","CoDEx-L":"CoDEx-L"}

def load(csv_path):
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].map(lambda m: MODEL_RENAME.get(str(m), str(m)))
    df["dataset"] = df["dataset"].map(lambda d: DATASET_RENAME.get(str(d), str(d)))
    mm = set(MODEL_ORDER) - set(df["model"].unique())
    md = set(DATASET_ORDER) - set(df["dataset"].unique())
    if mm: print(f"[warn] missing model(s): {sorted(mm)}", file=sys.stderr)
    if md: print(f"[warn] missing dataset(s): {sorted(md)}", file=sys.stderr)
    return df

def fmt(mean, std, pct=True, d=1):
    if pd.isna(mean): return "--"
    s = 100.0 if pct else 1.0
    m = mean*s
    if pd.isna(std): return f"{m:.{d}f}"
    return f"{m:.{d}f}\\std{{{std*s:.{d}f}}}"

def build_latex(df):
    L = [r"% Requires: \newcommand{\std}[1]{\tiny$\pm$#1} in the preamble",
         r"\begin{table*}[t]", r"\centering",
         r"\caption{Link prediction results (mean \std{std} over 10 seeds, as percentages) "
         r"for HS-GNN, AS-GNN, and ASR-GNN across six benchmarks. Best MRR per dataset in \textbf{bold}.}",
         r"\label{tab:main_results}", r"\resizebox{\textwidth}{!}{",
         r"\begin{tabular}{ll" + "c"*5 + "}", r"\toprule",
         r"Dataset & Model & MRR & Hits@1 & Hits@3 & Hits@10 & Density \\", r"\midrule"]
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty: continue
        best_m, best_v = None, -1
        for m in MODEL_ORDER:
            row = sub[sub["model"] == m]
            if row.empty: continue
            v = row["mrr_mean"].values[0]
            if pd.notna(v) and v > best_v: best_v, best_m = v, m
        for i, m in enumerate(MODEL_ORDER):
            row = sub[sub["model"] == m]
            if row.empty: continue
            r = row.iloc[0]
            mrr = fmt(r.get("mrr_mean"), r.get("mrr_std"))
            h1 = fmt(r.get("hits1_mean"), r.get("hits1_std"))
            h3 = fmt(r.get("hits3_mean"), r.get("hits3_std"))
            h10 = fmt(r.get("hits10_mean"), r.get("hits10_std"))
            dens = r.get("density_mean", r.get("final_density_mean", float("nan")))
            dens_s = f"{dens*100:.1f}\\%" if pd.notna(dens) else "--"
            if m == best_m: mrr = r"\textbf{" + mrr + "}"
            L.append(f"{ds if i==0 else ''} & {m} & {mrr} & {h1} & {h3} & {h10} & {dens_s} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(L)

def sanity(df):
    print("\n=== ASR-GNN vs AS-GNN (query-conditioning effect) ===")
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        a, r = sub[sub["model"]=="AS-GNN"], sub[sub["model"]=="ASR-GNN"]
        if a.empty or r.empty: continue
        am, rm = a["mrr_mean"].values[0], r["mrr_mean"].values[0]
        print(f"  {ds:10s}: AS-GNN={am:.4f}  ASR-GNN={rm:.4f}  delta={rm-am:+.4f}  ({'ASR-GNN better' if rm>am else 'AS-GNN better'})")
    print("\n=== AS-GNN vs HS-GNN (structural-conditioning effect) ===")
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        h, a = sub[sub["model"]=="HS-GNN"], sub[sub["model"]=="AS-GNN"]
        if h.empty or a.empty: continue
        hm, am = h["mrr_mean"].values[0], a["mrr_mean"].values[0]
        print(f"  {ds:10s}: HS-GNN={hm:.4f}  AS-GNN={am:.4f}  delta={am-hm:+.4f}")
    print("\n=== Relation-count effect (CoDEx vs FB15k-237, AS-GNN gain over HS-GNN) ===")
    for name, dss in [("FB15k-237", ["FB15k-237"]), ("CoDEx-*", ["CoDEx-S","CoDEx-M","CoDEx-L"])]:
        gains = []
        for ds in dss:
            sub = df[df["dataset"] == ds]
            h, a = sub[sub["model"]=="HS-GNN"], sub[sub["model"]=="AS-GNN"]
            if h.empty or a.empty: continue
            gains.append(a["mrr_mean"].values[0] - h["mrr_mean"].values[0])
        if gains: print(f"  {name:10s}: mean gain = {sum(gains)/len(gains):+.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/final/summary_all_metrics.csv")
    ap.add_argument("--out", default="results/final/results_tables.tex")
    args = ap.parse_args()
    df = load(args.csv)
    tex = build_latex(df)
    with open(args.out, "w") as f: f.write(tex + "\n")
    print(f"Wrote LaTeX table to {args.out}\n")
    print(tex)
    sanity(df)
