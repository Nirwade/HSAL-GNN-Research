# HSAL+LRD baseline: single-file Fantasy data, reusing the preprocessing of the core HSAL baseline
source fantasy_common.sh
if [ ! -f markers/${DATASET}_prep_done ]; then
    step "preprocessing for $DATASET not found: run run_fantasy_baseline.sbatch first"; exit 1
fi
if [ ! -f Data/${DATASET}_llm_embeddings.pt ]; then
    step "Data/${DATASET}_llm_embeddings.pt missing (make it with generate_llm_embeddings_v2.py)"; exit 1
fi
step "training HSAL+LRD (bug-fixed, seeded)"
python -u new_main_lrd_fixed.py --data $DATASET $TRAIN_ARGS
step "done"
