#!/usr/bin/env python3
"""
Aggregate ALL experiment results (PyKEEN + GNN models) into one
comparison table with stability metrics.

Run from the login node after all Slurm jobs finish:
    cd ~/kg_experiments/src
    source ~/venvs/kge-stability/bin/activate
    python aggregate_all_results.py --results-dir ../results

Produces:
  results/summary/combined_all_results.csv   ← every individual run
  results/summary/stability_summary.csv      ← per-(model, dataset) stats
  results/summary/model_ranking.csv          ← ranked by mean MRR
  results/figures/mrr_boxplot.png
  results/figures/sipd_barplot.png
"""

import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. Load results
# ──────────────────────────────────────────────

def load_all_csvs(results_dir: Path) -> pd.DataFrame:
    """Load every *.csv in results_dir (recursively) into one DataFrame."""
    files = list(results_dir.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {results_dir}")

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    # Drop duplicates (same seed+model+dataset)
    key_cols = [c for c in ["seed", "model", "dataset"] if c in combined.columns]
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined


# ──────────────────────────────────────────────
# 2. Stability metrics
# ──────────────────────────────────────────────

def sipd(series: pd.Series) -> float:
    """Seed-Induced Performance Difference = max − min MRR."""
    s = series.dropna()
    return float(s.max() - s.min()) if len(s) > 1 else float("nan")


def coefficient_of_variation(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.std() / s.mean()) if (len(s) > 1 and s.mean() != 0) else float("nan")


def compute_stability(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["model", "dataset"] if c in df.columns]

    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        mrr = grp["mrr"].dropna()

        row["n_seeds"]        = len(mrr)
        row["mean_mrr"]       = mrr.mean()
        row["std_mrr"]        = mrr.std()
        row["min_mrr"]        = mrr.min()
        row["max_mrr"]        = mrr.max()
        row["sipd"]           = sipd(mrr)
        row["cv"]             = coefficient_of_variation(mrr)

        for col in ["mr", "hits_at_1", "hits_at_3", "hits_at_10"]:
            if col in grp.columns:
                row[f"mean_{col}"] = grp[col].dropna().mean()

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("mean_mrr", ascending=False)
    return summary


# ──────────────────────────────────────────────
# 3. Plots
# ──────────────────────────────────────────────

def plot_mrr_boxplot(df: pd.DataFrame, out_path: Path):
    """Box plot of MRR distribution per model × dataset."""
    group_cols = [c for c in ["model", "dataset"] if c in df.columns]
    if not group_cols:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(df["model"].unique()) * 1.5), 5))
    groups = []
    labels = []
    for (model, *rest), grp in df.groupby(group_cols):
        mrr_vals = grp["mrr"].dropna().tolist()
        if mrr_vals:
            ds = rest[0] if rest else ""
            groups.append(mrr_vals)
            labels.append(f"{model}\n({ds})" if ds else model)

    ax.boxplot(groups, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    ax.set_ylabel("MRR (filtered)")
    ax.set_title("MRR Distribution Across Seeds — All Models")
    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_sipd_bar(summary: pd.DataFrame, out_path: Path):
    """Bar chart of SIPD per model × dataset."""
    if "sipd" not in summary.columns:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.2), 5))
    x = range(len(summary))
    bars = ax.bar(x, summary["sipd"], color="#DD8452", alpha=0.85, edgecolor="black")

    # Annotate with mean MRR
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0002,
                f"µ={row['mean_mrr']:.4f}",
                ha="center", va="bottom", fontsize=7)

    labels = []
    for _, row in summary.iterrows():
        model = row.get("model", "")
        ds    = row.get("dataset", "")
        labels.append(f"{model}\n({ds})" if ds else model)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("SIPD (max MRR − min MRR)")
    ax.set_title("Seed-Induced Performance Difference (SIPD) — All Models\n"
                 "Lower SIPD = more stable across seeds")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_mrr_vs_sipd(summary: pd.DataFrame, out_path: Path):
    """Scatter: mean MRR vs SIPD — ideal models are top-left."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in summary.iterrows():
        ax.scatter(row["sipd"], row["mean_mrr"], s=80, zorder=3)
        label = f"{row.get('model','')} ({row.get('dataset','')})"
        ax.annotate(label, (row["sipd"], row["mean_mrr"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=6)
    ax.set_xlabel("SIPD (instability ↑)")
    ax.set_ylabel("Mean MRR (performance ↑)")
    ax.set_title("Performance vs Stability\n(Ideal: high MRR, low SIPD — top-left)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ──────────────────────────────────────────────
# 4. Console report
# ──────────────────────────────────────────────

def print_report(summary: pd.DataFrame, df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("STABILITY SUMMARY")
    print("=" * 70)

    cols_to_show = ["model", "dataset", "n_seeds", "mean_mrr", "std_mrr",
                    "sipd", "cv", "mean_hits_at_10"]
    cols_present = [c for c in cols_to_show if c in summary.columns]
    print(summary[cols_present].to_string(index=False, float_format="{:.5f}".format))

    print("\n" + "=" * 70)
    print("PAPER INSIGHT")
    print("=" * 70)

    if "sipd" in summary.columns and "model" in summary.columns:
        best = summary.iloc[0]
        worst = summary.sort_values("sipd", ascending=False).iloc[0]
        print(f"\nMost stable model   : {best.get('model','')} "
              f"(SIPD={best['sipd']:.5f}, mean MRR={best['mean_mrr']:.5f})")
        print(f"Least stable model  : {worst.get('model','')} "
              f"(SIPD={worst['sipd']:.5f}, mean MRR={worst['mean_mrr']:.5f})")
        print(f"\nMax SIPD / Min SIPD ratio = "
              f"{summary['sipd'].max() / max(summary['sipd'].min(), 1e-9):.2f}x")
        print("\nNote: if SIPD ≈ typical KGE improvement margins (+0.001–0.003),")
        print("      reported improvements in the literature may not be significant.")

    print("=" * 70)


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str,
                        default="~/kg_experiments/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    summary_dir = results_dir / "summary"
    figures_dir = results_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    df = load_all_csvs(results_dir)
    print(f"  Loaded {len(df)} rows from {df['model'].nunique() if 'model' in df.columns else '?'} model(s)")

    summary = compute_stability(df)

    # ── Save CSV ──
    combined_out = summary_dir / "combined_all_results.csv"
    df.to_csv(combined_out, index=False)
    print(f"  Saved {combined_out}")

    summary_out = summary_dir / "stability_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"  Saved {summary_out}")

    ranking_out = summary_dir / "model_ranking.csv"
    ranking_cols = [c for c in ["model", "dataset", "mean_mrr", "sipd",
                                 "std_mrr", "mean_hits_at_10", "n_seeds"]
                    if c in summary.columns]
    summary[ranking_cols].to_csv(ranking_out, index=False)
    print(f"  Saved {ranking_out}")

    # ── Plots ──
    plot_mrr_boxplot(df,      figures_dir / "mrr_boxplot.png")
    plot_sipd_bar(summary,    figures_dir / "sipd_barplot.png")
    plot_mrr_vs_sipd(summary, figures_dir / "mrr_vs_sipd.png")

    print_report(summary, df)


if __name__ == "__main__":
    main()
