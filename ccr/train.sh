#!/bin/bash
# env: VARIANT, SEED, EPOCHS ; test is only for the final report, selection uses validation
source ccr/common.sh
case $VARIANT in
  base)        TAG=_ev;         X="";;
  sertyped)    TAG=_ev;         X="--series_typed";;
  bidir)       TAG=_ev;         X="--series_mode bidir";;
  bidirtyped)  TAG=_ev;         X="--series_mode bidir --series_typed";;
  rel_b5)      TAG=_ev_rel_b5;  X="--use_rel";;
  rel_b2)      TAG=_ev_rel_b2;  X="--use_rel";;
  rel_bu)      TAG=_ev_rel_bu;  X="--use_rel";;
  rand_b5)     TAG=_ev_rand_b5; X="--use_rel";;
  shuf_b5)     TAG=_ev_shuf_b5; X="--use_rel";;
  full)        TAG=_ev_rel_b5;  X="--use_rel --series_mode bidir --series_typed";;
  *) echo "unknown VARIANT $VARIANT"; exit 2;;
esac
python -u new_main_BPR.py --data $DATASET --tag ${TAG}${TAGSUF} $X --batchSize 256 --item_max_length $ITEM_LEN \
  --user_max_length $USER_LEN --k_hop $KHOP --lr 0.001 --l2 0.0001 --layer_num 3 --hidden_size 50 \
  --gpu 0 --seed $SEED --epoch ${EPOCHS:-20} --val --model_record --resume $EXTRA_ARGS
