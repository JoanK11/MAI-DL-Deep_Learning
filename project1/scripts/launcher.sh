#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --gres gpu:1
#SBATCH --job-name="project1"
#SBATCH --chdir=/home/nct/nct01082/project1
#SBATCH --output=/home/nct/nct01082/project1/stdout.txt
#SBATCH --error=/home/nct/nct01082/project1/stderr.txt
module purge
module load anaconda/2023.07

python src/main.py
