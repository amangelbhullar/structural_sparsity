"""
Comprehensive KGE Metrics for Survey Paper
Computes all metrics possible from existing results data.

Outputs:
  results/metrics/parameter_efficiency.csv
  results/metrics/training_efficiency.csv
  results/metrics/statistical_robustness.csv
  results/metrics/dataset_characteristics.csv
  results/metrics/full_metrics_table.csv   ← main table for paper

Usage (from ~/kg_experiments/):
    python src/compute_metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_DIR / "all_results.csv")

# Add RGCN if present
rgcn_files = list((RESULTS_DIR / "rgcn").glob("*.csv"))
if rgcn_files:
    rgcn = pd.concat([pd.read_csv(f) for f in rgcn_files], ignore_index=True)
    df = pd.concat([df, rgcn], ignore_index=True)

print(f"Loaded {len(df)} runs across {df['model'].nunique()} models, "
      f"{df['dataset'].nunique()} datasets\n")

# ── 1. Parameter Efficiency ───────────────────────────────────────────────────
# Known parameter counts per model (embedding dim d, entities E, relations R)
# Formula: total params = entity_emb + relation_emb + model_specific
# Using d=200 (your setting), E=14541 (FB15k-237), R=237

E, R, d = 14541, 237, 200

param_info = {
    # model: (total_params, emb_multiplier, notes)
    "TransE":    (E*d + R*d,               1,   "Real embeddings, same dim for e and r"),
    "TransH":    (E*d + R*d + R*d,         1,   "Extra normal vector per relation"),
    "TransR":    (E*d + R*d + R*d*d,       1,   "Relation-specific projection matrices"),
    "TransD":    (E*d + R*d + E*d + R*d,   1,   "Dynamic projection vectors"),
    "DistMult":  (E*d + R*d,               1,   "Diagonal bilinear"),
    "ComplEx":   (E*2*d + R*2*d,           2,   "Complex embeddings (2x real)"),
    "RotatE":    (E*2*d + R*d,             2,   "Complex entity, phase relation"),
    "QuatE":     (E*4*d + R*4*d,           4,   "Quaternion embeddings (4x real)"),
    "RESCAL":    (E*d + R*d*d,             1,   "Full relation matrix (expensive)"),
    "TuckER":    (E*d + R*d + d*d*d,       1,   "Tucker decomposition core tensor"),
    "SimplE":    (E*2*d + R*d,             2,   "Two entity embeddings"),
    "MuRE":      (E*d + R*d + R*d + R,     1,   "Diagonal + bias per relation"),
    "BoxE":      (E*2*d + R*2*d,           2,   "Box embeddings (center+offset)"),
    "ConvE":     (E*d + R*d + 32*d,        1,   "Conv filters + projection"),
    "PairRE":    (E*d + R*2*d,             1,   "Paired relation embeddings"),
    "AutoSF":    (E*d + R*4*d,             1,   "4 relation components"),
    "RGCN-DistMult": (E*d + R*d + 30*d*d + R*30, 1, "R-GCN basis decomp + DistMult"),
}

param_rows = []
for model, (params, emb_mult, note) in param_info.items():
    param_rows.append({
        "model":          model,
        "total_params":   params,
        "params_M":       round(params / 1e6, 2),
        "emb_multiplier": emb_mult,
        "notes":          note,
    })

param_df = pd.DataFrame(param_rows).sort_values("total_params")
param_df.to_csv(METRICS_DIR / "parameter_efficiency.csv", index=False)
print("── 1. Parameter Efficiency ───────────────────────")
print(param_df[["model","params_M","emb_multiplier","notes"]].to_string(index=False))

# ── 2. Training Efficiency ────────────────────────────────────────────────────
# Compute from existing runtime data
train_rows = []
for (dataset, model), grp in df.groupby(["dataset", "model"]):
    total_rt  = grp["runtime_seconds"].mean()
    epochs    = grp["epochs"].mode()[0]
    time_per_epoch = total_rt / epochs

    # Best seed performance
    best_mrr  = grp["mrr"].max()
    worst_mrr = grp["mrr"].min()
    mrr_range = best_mrr - worst_mrr

    train_rows.append({
        "dataset":          dataset,
        "model":            model,
        "epochs":           int(epochs),
        "total_runtime_s":  round(total_rt, 1),
        "time_per_epoch_s": round(time_per_epoch, 2),
        "time_per_epoch_m": round(time_per_epoch / 60, 3),
        "best_seed_mrr":    round(best_mrr, 4),
        "worst_seed_mrr":   round(worst_mrr, 4),
        "mrr_seed_range":   round(mrr_range, 4),
        "n_seeds":          len(grp),
    })

train_df = pd.DataFrame(train_rows).sort_values(["dataset", "time_per_epoch_s"])
train_df.to_csv(METRICS_DIR / "training_efficiency.csv", index=False)
print("\n── 2. Training Efficiency (FB15k-237) ────────────")
fb_train = train_df[train_df["dataset"] == "FB15k-237"]
print(fb_train[["model","epochs","time_per_epoch_s","best_seed_mrr","mrr_seed_range"]].to_string(index=False))

# ── 3. Statistical Robustness ─────────────────────────────────────────────────
stat_rows = []
for (dataset, model), grp in df.groupby(["dataset", "model"]):
    mrr_vals = grp["mrr"].values
    n = len(mrr_vals)
    mean = mrr_vals.mean()
    std  = mrr_vals.std()

    # 95% confidence interval
    if n > 1:
        ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=stats.sem(mrr_vals))
        ci_low, ci_high = round(ci[0], 4), round(ci[1], 4)
        ci_width = round(ci_high - ci_low, 4)
    else:
        ci_low = ci_high = ci_width = float("nan")

    stat_rows.append({
        "dataset":      dataset,
        "model":        model,
        "n_seeds":      n,
        "mrr_mean":     round(mean, 4),
        "mrr_std":      round(std, 4),
        "mrr_cv":       round(std / mean if mean > 0 else 0, 4),  # coefficient of variation
        "ci95_low":     ci_low,
        "ci95_high":    ci_high,
        "ci95_width":   ci_width,
        "best_seed":    round(mrr_vals.max(), 4),
        "worst_seed":   round(mrr_vals.min(), 4),
        "seed_range":   round(mrr_vals.max() - mrr_vals.min(), 4),
    })

stat_df = pd.DataFrame(stat_rows).sort_values(["dataset", "mrr_mean"], ascending=[True, False])
stat_df.to_csv(METRICS_DIR / "statistical_robustness.csv", index=False)
print("\n── 3. Statistical Robustness (FB15k-237) ─────────")
fb_stat = stat_df[stat_df["dataset"] == "FB15k-237"]
print(fb_stat[["model","mrr_mean","mrr_std","ci95_low","ci95_high","seed_range"]].to_string(index=False))

# ── 4. Dataset Characteristics ────────────────────────────────────────────────
# Known stats for standard benchmarks
dataset_info = pd.DataFrame([
    {
        "dataset":       "FB15k-237",
        "entities":      14541,
        "relations":     237,
        "train_triples": 272115,
        "valid_triples": 17535,
        "test_triples":  20466,
        "avg_degree":    37.4,
        "has_hierarchy": False,
        "symmetry":      "Low",
        "notes":         "Curated Freebase subset, general domain",
    },
    {
        "dataset":       "WN18RR",
        "entities":      40943,
        "relations":     11,
        "train_triples": 86835,
        "valid_triples": 3034,
        "test_triples":  3134,
        "avg_degree":    4.2,
        "has_hierarchy": True,
        "symmetry":      "High (hypernym/hyponym)",
        "notes":         "WordNet, hierarchical, sparse",
    },
    {
        "dataset":       "YAGO3-10",
        "entities":      123182,
        "relations":     37,
        "train_triples": 1079040,
        "valid_triples": 5000,
        "test_triples":  5000,
        "avg_degree":    17.5,
        "has_hierarchy": False,
        "symmetry":      "Medium",
        "notes":         "Large, real-world, multilingual entities",
    },
])
dataset_info.to_csv(METRICS_DIR / "dataset_characteristics.csv", index=False)
print("\n── 4. Dataset Characteristics ────────────────────")
print(dataset_info[["dataset","entities","relations","train_triples","avg_degree","has_hierarchy"]].to_string(index=False))

# ── 5. Full Metrics Table (for paper) ────────────────────────────────────────
# Merge performance + params + efficiency for FB15k-237
fb_perf = stat_df[stat_df["dataset"] == "FB15k-237"].copy()
fb_eff  = train_df[train_df["dataset"] == "FB15k-237"][
    ["model","time_per_epoch_s","epochs"]].copy()

# hits@10 mean
hits10 = df[df["dataset"]=="FB15k-237"].groupby("model")["hits_at_10"].mean().round(4)
hits10.name = "hits10_mean"

# hits@1 and hits@3 if available
h1 = df[df["dataset"]=="FB15k-237"].groupby("model")["hits_at_1"].mean().round(4) \
    if "hits_at_1" in df.columns else pd.Series(dtype=float)
h3 = df[df["dataset"]=="FB15k-237"].groupby("model")["hits_at_3"].mean().round(4) \
    if "hits_at_3" in df.columns else pd.Series(dtype=float)

param_lookup = param_df.set_index("model")[["params_M","emb_multiplier"]]

full = fb_perf.merge(fb_eff, on="model") \
              .merge(hits10, on="model") \
              .join(param_lookup, on="model") \
              .join(h1.rename("hits1_mean"), on="model") \
              .join(h3.rename("hits3_mean"), on="model")

# Format CI as string for paper
full["95%_CI"] = full.apply(
    lambda r: f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]", axis=1)

paper_cols = ["model","mrr_mean","mrr_std","95%_CI",
              "hits1_mean","hits3_mean","hits10_mean",
              "params_M","emb_multiplier",
              "time_per_epoch_s","epochs","seed_range","n_seeds"]
paper_cols = [c for c in paper_cols if c in full.columns]

full_sorted = full[paper_cols].sort_values("mrr_mean", ascending=False)
full_sorted.to_csv(METRICS_DIR / "full_metrics_table.csv", index=False)

print("\n── 5. Full Metrics Table (FB15k-237, sorted by MRR) ──")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(full_sorted.to_string(index=False))

print(f"\n✓ All metrics saved to {METRICS_DIR}/")
print("  parameter_efficiency.csv")
print("  training_efficiency.csv")
print("  statistical_robustness.csv")
print("  dataset_characteristics.csv")
print("  full_metrics_table.csv  ← use this in your paper")
