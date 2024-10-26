#!/bin/bash
#SBATCH --account=<project_id>
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00

module purge
module load gcc/11.3.0
module load python/3.11.3

pip install --user -r requirements.txt

python fingpt_evaluation.py