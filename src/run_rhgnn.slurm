#!/bin/bash
#SBATCH --account=def-kobti          # change to your Alliance account
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=rhgnn_%a
#SBATCH --output=/home/amangel/kg_experiments/logs/rhgnn_%a_%j.out
#SBATCH --error=/home/amangel/kg_experiments/logs/rhgnn_%a_%j.err
#SBATCH --array=0-3

# ── Activate environment ──────────────────────
source /home/amangel/venvs/kge-stability/bin/activate

# ── Dataset selection by array index ─────────
DATASETS=(ICEWS14 ICEWS18 GDELT YAGO3-10)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

BASE=/home/amangel/kg_experiments
DATA_DIR=$BASE/data/$DATASET
OUT_DIR=$BASE/results/$DATASET
SCRIPT=$BASE/src/Rhgnn_fixed.py

mkdir -p $OUT_DIR
mkdir -p $BASE/logs

echo "======================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Dataset:     $DATASET"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "======================================"

# ── Run RHGNN across 5 seeds ──────────────────
python $SCRIPT \
    --data_dir   $DATA_DIR \
    --output_dir $OUT_DIR \
    --seeds 42 0 1 2 3 \
    --embedding_dim 200 \
    --hidden_dim    256 \
    --curvature     1.0 \
    --dropout       0.1 \
    --batch_size    512 \
    --eval_batch_size 64 \
    --epochs        200 \
    --patience      10 \
    --lr            1e-3 \
    --weight_decay  1e-6 \
    --num_negatives 32 \
    --num_workers   4 \
    --device        cuda \
    --delta_hyp

echo "Done: $DATASET"
