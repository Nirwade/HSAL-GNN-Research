#!/bin/bash
# builds negatives, series table, base graph and every Newdata variant (skips what exists)
source ccr/common.sh
MAXU=${MAX_USERS:-}          # set MAX_USERS=300 for the smoke test
EXTRA=""; [ -n "$MAXU" ] && EXTRA="--max_users $MAXU"
[ -f ${DATASET}_neg ] || python generate_neg.py --data $DATASET
[ -f Series_table_sinusodial_Series_table_${DATASET}.csv ] || python sinusoidal.py --data $DATASET
python make_control_edges.py --data $DATASET --edges $EDGES
base="python new_data1.py --data $DATASET --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --exclude_val $EXTRA"
D() { local B=$1; local T=$1$TAGSUF; if [ -n "$ONLY" ] && [[ " $ONLY " != *" $B "* ]]; then return; fi; shift; set -- "$T" "$@"; [ -d Newdata/${DATASET}$1_${ITEM_LEN}_${USER_LEN}_${KHOP}/test ] && echo "skip $1" || { echo "build $1"; $base --tag $1 "${@:2}"; }; }
D _ev
D _ev_rel_b5   --rel_edges $EDGES --rel_budget 5
D _ev_rel_b2   --rel_edges $EDGES --rel_budget 2
D _ev_rel_bu   --rel_edges $EDGES --rel_budget -1
D _ev_rand_b5  --rel_edges ${EDGES%.csv}_random.csv  --rel_budget 5
D _ev_shuf_b5  --rel_edges ${EDGES%.csv}_shuffle.csv --rel_budget 5
for t in ${ONLY:-_ev _ev_rel_b5 _ev_rel_b2 _ev_rel_bu _ev_rand_b5 _ev_shuf_b5}; do t=$t$TAGSUF
  f=Newdata/${DATASET}${t}_${ITEM_LEN}_${USER_LEN}_${KHOP}/augmentation_stats.json; [ -f $f ] && { echo $f; cat $f; echo; }
done
