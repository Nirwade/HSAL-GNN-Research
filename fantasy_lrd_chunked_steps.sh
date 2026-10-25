# HSAL+LRD chunked: same code and settings, data added to the graph in 8 chunks (reuses that preprocessing)
source fantasy_common.sh
NAME=${DATASET}_chunked
if [ ! -f markers/${NAME}_prep_done ]; then
    step "preprocessing for $NAME not found: run run_fantasy_chunked.sbatch first"; exit 1
fi
if [ ! -f Data/${NAME}_llm_embeddings.pt ]; then
    step "Data/${NAME}_llm_embeddings.pt missing (make it with generate_llm_embeddings_chunked.py)"; exit 1
fi
step "training HSAL+LRD (bug-fixed, seeded)"
python -u new_main_lrd_chunked.py --data $NAME $TRAIN_ARGS
step "done"
