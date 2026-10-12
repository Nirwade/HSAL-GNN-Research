#!/bin/bash
#SBATCH --cluster=faculty
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# submit: sbatch --job-name=g_${V}_s${S} --output=logs/%x_%j.out --export=ALL,VARIANT=$V,SEED=$S,EPOCHS=20 ccr/train.sbatch
bash ccr/run_in_container.sh ccr/train.sh
