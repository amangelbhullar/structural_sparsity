import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("../results")
OUTPUT_COMBINED = RESULTS_DIR / "all_model_seed_results.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "all_model_summary.csv"


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, model), group in df.groupby(["dataset", "model"]):
        group = group.sort_values("seed")

        mrr = group["mrr"]
        mr = group["mr"]
        hits10 = group["hits_at_10"]

        row = {
            "dataset": dataset,
            "model": model,
            "num_seeds": int(group["seed"].nunique()),
            "mean_mrr": mrr.mean(),
            "std_mrr": mrr.std(),
            "min_mrr": mrr.min(),
            "max_mrr": mrr.max(),
            "sipd": mrr.max() - mrr.min(),
            "mean_mr": mr.mean(),
            "std_mr": mr.std(),
            "mean_hits_at_10": hits10.mean(),
            "std_hits_at_10": hits10.std(),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["dataset", "mean_mrr"], ascending=[True, False])
    return summary


def main():
    pattern = str(RESULTS_DIR / "FB15k-237_*_seed*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No result files found matching: {pattern}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)
    df.to_csv(OUTPUT_COMBINED, index=False)

    print("\n=== Combined Per-Seed Results ===")
    print(df[["dataset", "model", "seed", "mrr", "mr", "hits_at_10"]].to_string(index=False))

    summary = compute_summary(df)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n=== Per-Model Summary ===")
    print(
        summary[
            [
                "dataset",
                "model",
                "num_seeds",
                "mean_mrr",
                "std_mrr",
                "min_mrr",
                "max_mrr",
                "sipd",
                "mean_hits_at_10",
            ]
        ].to_string(index=False)
    )

    print("\n=== Stability Ranking (lower SIPD is better) ===")
    stability = summary.sort_values(["dataset", "sipd"], ascending=[True, True]).reset_index(drop=True)
    print(
        stability[
            ["dataset", "model", "sipd", "std_mrr", "mean_mrr"]
        ].to_string(index=False)
    )

    print("\n=== Performance Ranking (higher mean MRR is better) ===")
    performance = summary.sort_values(["dataset", "mean_mrr"], ascending=[True, False]).reset_index(drop=True)
    print(
        performance[
            ["dataset", "model", "mean_mrr", "std_mrr", "sipd"]
        ].to_string(index=False)
    )

    print(f"\nSaved combined seed-level results to: {OUTPUT_COMBINED}")
    print(f"Saved model summary to: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
