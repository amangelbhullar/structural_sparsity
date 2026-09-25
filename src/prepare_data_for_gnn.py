#!/usr/bin/env python3
"""
Run this ONCE on the Rorqual LOGIN NODE to:
  1. Verify your PyKEEN-cached datasets (FB15k-237, WN18RR, YAGO3-10)
     are already downloaded in ~/pykeen_home.
  2. Convert them from PyKEEN's internal format into plain train/valid/test.txt
     files (tab-separated head\\trelation\\ttail) under ~/kg_experiments/data/.
     This is the format the GNN scripts expect.
  3. Print a summary so you can confirm everything is ready before submitting
     Slurm jobs.

Usage (login node only — NO training here):
    source ~/venvs/kge-stability/bin/activate
    export PYKEEN_HOME=$HOME/pykeen_home
    python ~/kg_experiments/src/prepare_data_for_gnn.py
"""

import os
from pathlib import Path

# Tell PyKEEN where the cache lives
os.environ.setdefault("PYKEEN_HOME", str(Path.home() / "pykeen_home"))

from pykeen.datasets import FB15k237, WN18RR, YAGO310

DATASETS = {
    "FB15k-237": FB15k237,
    "WN18RR":    WN18RR,
    "YAGO3-10":  YAGO310,
}

DATA_ROOT = Path.home() / "kg_experiments" / "data"


def tripleset_to_txt(triple_set, out_path: Path) -> int:
    """Write a PyKEEN TriplesFactory to a tab-separated text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    triples = triple_set.triples          # numpy array [N, 3] of strings
    with open(out_path, "w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    return len(triples)


def main():
    print("=" * 60)
    print("GNN Data Preparation — Rorqual Login Node")
    print("=" * 60)

    for ds_name, ds_class in DATASETS.items():
        print(f"\n── {ds_name} ──")
        try:
            ds = ds_class()
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            print("  → Run the download step first (see earlier instructions).")
            continue

        ds_dir = DATA_ROOT / ds_name
        splits = {
            "train": ds.training,
            "valid": ds.validation,
            "test":  ds.testing,
        }

        for split_name, factory in splits.items():
            out_file = ds_dir / f"{split_name}.txt"
            if out_file.exists():
                n = sum(1 for _ in open(out_file))
                print(f"  {split_name}.txt already exists ({n} triples) — skipping.")
                continue
            n = tripleset_to_txt(factory, out_file)
            print(f"  Wrote {out_file}  ({n} triples)")

        # Verify
        for split_name in ["train", "valid", "test"]:
            p = ds_dir / f"{split_name}.txt"
            n = sum(1 for _ in open(p))
            print(f"  ✓  {split_name}.txt  {n} triples")

    print("\n" + "=" * 60)
    print("Done. Data is ready under:")
    print(f"  {DATA_ROOT}/")
    print("\nYou can now submit GNN Slurm jobs:")
    print("  cd ~/kg_experiments/slurm")
    print('  sbatch --export=DATASET=FB15k-237 run_rgcn_kge.slurm')
    print('  sbatch --export=DATASET=FB15k-237 run_compgcn_kge.slurm')
    print("=" * 60)


if __name__ == "__main__":
    main()
