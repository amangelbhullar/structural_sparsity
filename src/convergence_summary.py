from pathlib import Path
import pandas as pd

df = pd.read_csv("../results/all_seeds_all_models_all_epochs.csv")

summary = (
    df.groupby(["dataset", "model", "epoch_budget"], dropna=False)
    .agg(
        num_seeds=("seed", "nunique"),
        mean_mrr=("mrr", "mean"),
        std_mrr=("mrr", "std"),
        min_mrr=("mrr", "min"),
        max_mrr=("mrr", "max"),
        mean_hits_at_10=("hits_at_10", "mean"),
        mean_mr=("mr", "mean"),
    )
    .reset_index()
)

summary["sipd"] = summary["max_mrr"] - summary["min_mrr"]
summary = summary.sort_values(["model", "epoch_budget"]).reset_index(drop=True)

out = Path("../results/convergence_by_model_epoch.csv")
summary.to_csv(out, index=False)

print(summary.to_string(index=False))
print(f"\nSaved to: {out}")
