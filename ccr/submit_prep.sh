#!/bin/bash
# Step 1: baseline data (also builds the shared base graph). Step 2: the five relation variants in parallel once step 1 is done.
mkdir -p logs
J0=$(sbatch --parsable --job-name=prep_ev --output=logs/%x_%j.out --export=ALL,ONLY=_ev ccr/prep.sbatch); J0=${J0%%;*}
echo "baseline prep job $J0"
for t in _ev_rel_b5 _ev_rel_b2 _ev_rel_bu _ev_rand_b5 _ev_shuf_b5; do
  sbatch --job-name=prep$t --output=logs/%x_%j.out --dependency=afterok:$J0 --export=ALL,ONLY=$t ccr/prep.sbatch
done
