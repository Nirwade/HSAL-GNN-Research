#!/bin/bash
# usage: bash ccr/submit_all.sh "base rel_b5" "0 1 2 3 4"   (order of the ladder is the order you pass)
mkdir -p logs
for V in $1; do for S in $2; do
  sbatch --job-name=g_${V}_s${S} --output=logs/%x_%j.out --export=ALL,VARIANT=$V,SEED=$S,EPOCHS=${EPOCHS:-20} ccr/train.sbatch
done; done
