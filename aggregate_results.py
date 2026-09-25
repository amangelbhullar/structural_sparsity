import pandas as pd
from pathlib import Path

results_dir = Path.home() / "kg_experiments" / "results"

all_rows = []

skip_files = {
    "all_results.csv",
    "summary_results.csv",
    "all_model_seed_results.csv",
    "all_model_summary.csv",
    "all_seeds_all_models.csv",
    "all_seeds_all_models_all_epochs.csv",
    "convergence_by_model_epoch.csv",
    "summary_all_models_all_epochs.csv",
    "summary_Complex_FB15k237.csv",
}

for file in results_dir.glob("*.csv"):
    if file.name in skip_files:
        continue

    try:
        df = pd.read_csv(file)

        required = {"dataset", "model", "seed", "mrr", "mr", "hits_at_10"}
        if not required.issubset(df.columns):
            continue

        all_rows.append(df)

    except Exception as e:
        print(f"Skipping {file.name}: {e}")

if not all_rows:
    raise ValueError("No per-seed result CSV files found.")

all_data = pd.concat(all_rows, ignore_index=True)

all_data = all_data.sort_values(by=["dataset", "model", "seed"]).reset_index(drop=True)
all_data.to_csv(results_dir / "all_results.csv", index=False)
print("Saved:", results_dir / "all_results.csv")

summary = (
    all_data
    .groupby(["dataset", "model"])
    .agg(
        seeds=("seed", "count"),
        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", "std"),
        mr_mean=("mr", "mean"),
        mr_std=("mr", "std"),
        hits10_mean=("hits_at_10", "mean"),
        hits10_std=("hits_at_10", "std"),
        runtime_mean=("runtime_seconds", "mean"),
        runtime_std=("runtime_seconds", "std"),
    )
    .reset_index()
    .sort_values(by=["dataset", "mrr_mean"], ascending=[True, False])
)

summary.to_csv(results_dir / "summary_results.csv", index=False)
print("Saved:", results_dir / "summary_results.csv")

for dataset in sorted(all_data["dataset"].unique()):
    dataset_df = all_data[all_data["dataset"] == dataset]
    dataset_df.to_csv(results_dir / f"{dataset}_all_results.csv", index=False)

    dataset_summary = summary[summary["dataset"] == dataset]
    dataset_summary.to_csv(results_dir / f"{dataset}_summary_results.csv", index=False)

    print(f"Saved dataset files for {dataset}")
