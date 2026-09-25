#!/bin/bash
#SBATCH --account=def-kobti_gpu
#SBATCH --job-name=profile-efficiency
#SBATCH --time=0-00:10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/%u/kg_experiments/logs/%x-%A.out
#SBATCH --error=/home/%u/kg_experiments/logs/%x-%A.err

module purge
module load python/3.11.5 cuda
source $HOME/venvs/kge-stability/bin/activate
cd $HOME/kg_experiments/src
python3 profile_efficiency.py
