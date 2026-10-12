#!/bin/bash
source ccr/common.sh
set -e
python test_dgl_relation.py
export MAX_USERS=300 TAGSUF=_smoke
bash ccr/prep.sh
# 2-batch runs of the baseline and the richest variant (checkpoints go to the usual folders, tag them as smoke)
for v in base full; do
  TAGSUF=_smoke VARIANT=$v SEED=999 EPOCHS=1 EXTRA_ARGS="--max_batches 2" bash ccr/train.sh
done
echo SMOKE_OK
