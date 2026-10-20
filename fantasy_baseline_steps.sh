# Baseline: original HSAL scripts on the single Fantasy CSV
source fantasy_common.sh
if [ -f markers/${DATASET}_prep_done ]; then
    step "preprocessing already done, skipping"
else
    # a stopped job may have left half-built outputs: start preprocessing from scratch
    rm -rf Data/${DATASET}_graph ${DATASET}_neg Series_table_sinusodial_Series_table_${DATASET}.csv Newdata/${DATASET}_${ITEM_LEN}_${USER_LEN}_${KHOP}
    step "generate_neg.py";  python generate_neg.py --data $DATASET | tail -1
    step "sinusoidal.py";    python sinusoidal.py --data $DATASET
    step "new_data1.py";     python new_data1.py --data $DATASET --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --job 16
    touch markers/${DATASET}_prep_done
    step "preprocessing done"
fi
step "training"
python -u new_main_BPR.py --data $DATASET $TRAIN_ARGS
step "done"
