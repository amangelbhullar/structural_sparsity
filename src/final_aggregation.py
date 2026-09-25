import glob
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("../results")

RAW_OUTPUT = RESULTS_DIR / "all_seeds_all_models.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "summary_all_models.csv"

# =========================
# Load all experiment files
# =========================
files = glob.glob(str(RESULTS_DIR / "FB15k-237_*_seed*.csv"))

if not files:
    raise ValueError("No result files found")

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

# Combine all
all_df = pd.concat(dfs, ignore_index=True)

# Sort
all_df = all_df.sort_values(["model", "seed"]).reset_index(drop=True)

# Save raw combined
all_df.to_csv(RAW_OUTPUT, index=False)

print("\n=== RAW DATA (ALL SEEDS + MODELS) ===")
print(all_df.to_string(index=False))

# =========================
# Aggregation
# =========================
summary = (
    all_df.groupby(["dataset", "model"])
    .agg(
        num_seeds=("seed", "nunique"),
        mean_mrr=("mrr", "mean"),
        std_mrr=("mrr", "std"),
        min_mrr=("mrr", "min"),
        max_mrr=("mrr", "max"),
        mean_hits_at_10=("hits_at_10", "mean"),
        std_hits_at_10=("hits_at_10", "std"),
        mean_mr=("mr", "mean"),
        std_mr=("mr", "std"),
    )
    .reset_index()
)

# Add SIPD
summary["sipd"] = summary["max_mrr"] - summary["min_mrr"]

# Sort by performance
summary = summary.sort_values("mean_mrr", ascending=False).reset_index(drop=True)

# Save summary
summary.to_csv(SUMMARY_OUTPUT, index=False)

print("\n=== SUMMARY (PER MODEL) ===")
print(summary.to_string(index=False))

# =========================
# Rankings
# =========================
print("\n=== PERFORMANCE RANKING (HIGHER MRR BETTER) ===")
print(summary[["model", "mean_mrr", "std_mrr", "sipd"]].to_string(index=False))

print("\n=== STABILITY RANKING (LOWER SIPD BETTER) ===")
print(summary.sort_values("sipd")[["model", "sipd", "std_mrr", "mean_mrr"]].to_string(index=False))

print(f"\nSaved RAW file → {RAW_OUTPUT}")
print(f"Saved SUMMARY → {SUMMARY_OUTPUT}")
