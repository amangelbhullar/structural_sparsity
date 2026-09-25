import glob
from pathlib import Path
import re
import pandas as pd

RESULTS_DIR = Path("../results")
OUT_ALL = RESULTS_DIR / "all_seeds_all_models_all_epochs.csv"
OUT_SUMMARY = RESULTS_DIR / "summary_all_models_all_epochs.csv"

files = glob.glob(str(RESULTS_DIR / "FB15k-237_*_seed*.csv"))

if not files:
    raise ValueError("No result CSV files found in ../results")

dfs = []
for f in files:
    df = pd.read_csv(f)

    name = Path(f).name

    # Extract epoch from filename if present, else use file content if available
    m = re.search(r"_ep(\d+)_seed(\d+)\.csv$", name)
    if m:
        df["epoch_budget"] = int(m.group(1))
        df["seed_from_name"] = int(m.group(2))
    else:
        df["epoch_budget"] = df["epochs"] if "epochs" in df.columns else None
        sm = re.search(r"_seed(\d+)\.csv$", name)
        df["seed_from_name"] = int(sm.group(1)) if sm else None

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

preferred = [
    "dataset", "model", "epoch_budget", "seed", "seed_from_name",
    "epochs", "mrr", "mr", "hits_at_10", "runtime_seconds", "device"
]
existing = [c for c in preferred if c in all_df.columns]
other = [c for c in all_df.columns if c not in existing]
all_df = all_df[existing + other]

all_df = all_df.sort_values(["dataset", "model", "epoch_budget", "seed"]).reset_index(drop=True)
all_df.to_csv(OUT_ALL, index=False)

summary = (
    all_df.groupby(["dataset", "model", "epoch_budget"], dropna=False)
    .agg(
        num_seeds=("seed", "nunique"),
        mean_mrr=("mrr", "mean"),
        std_mrr=("mrr", "std"),
        min_mrr=("mrr", "min"),
        max_mrr=("mrr", "max"),
        mean_mr=("mr", "mean"),
        std_mr=("mr", "std"),
        mean_hits_at_10=("hits_at_10", "mean"),
        std_hits_at_10=("hits_at_10", "std"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    )
    .reset_index()
)

summary["sipd"] = summary["max_mrr"] - summary["min_mrr"]
summary = summary.sort_values(["dataset", "epoch_budget", "mean_mrr"], ascending=[True, True, False]).reset_index(drop=True)
summary.to_csv(OUT_SUMMARY, index=False)

print("\n=== Saved seed-level combined file ===")
print(OUT_ALL)

print("\n=== Saved summary file ===")
print(OUT_SUMMARY)

print("\n=== Summary preview ===")
print(summary.to_string(index=False))
