import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("../results")
OUTPUT_FILE = RESULTS_DIR / "all_seeds_all_models.csv"

# Read all per-seed, per-model CSV files
files = glob.glob(str(RESULTS_DIR / "FB15k-237_*_seed*.csv"))

if not files:
    raise ValueError("No result files found in ../results")

dfs = []
for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Keep columns in a clean order if they exist
preferred_order = [
    "dataset",
    "model",
    "seed",
    "epochs",
    "mrr",
    "mr",
    "hits_at_10",
    "runtime_seconds",
    "device",
]
existing_cols = [c for c in preferred_order if c in combined.columns]
other_cols = [c for c in combined.columns if c not in existing_cols]
combined = combined[existing_cols + other_cols]

combined = combined.sort_values(by=["dataset", "model", "seed"]).reset_index(drop=True)

combined.to_csv(OUTPUT_FILE, index=False)

print("\n=== Combined all seeds and all models ===")
print(combined.to_string(index=False))
print(f"\nSaved to: {OUTPUT_FILE}")
