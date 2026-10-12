#!/bin/bash
#SBATCH --cluster=faculty
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=hsal_prep
#SBATCH --output=logs/prep_%j.out
#SBATCH --requeue
bash ccr/run_in_container.sh ccr/prep.sh
