#!/bin/bash
#SBATCH --cluster=faculty
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --job-name=hsal_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=snirwade@buffalo.edu
#SBATCH --mail-type=ALL

# All Anime / ML-1M chunking jobs in one file. Put it in WORKDIR, pick the dataset and step when submitting:
#   sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=hsal_chunked --job-name=HSALchunk_Anime anime_ml1m_jobs.sbatch
#   sbatch --export=ALL,DATASET=MovieLens_all_v2,MODE=lrd_original,EPOCHS=5 --job-name=LRDorig_ML1M anime_ml1m_jobs.sbatch
# MODE (run in this order for each dataset):
#   hsal_chunked   split into 8 user-disjoint chunks, preprocess, pack samples, train HSAL
#   lrd_chunked    train HSAL+LRD on the chunked samples (waits for hsal_chunked preprocessing)
#   verify         compare the original and chunked samples one by one; exits 1 unless all match
#   hsal_original  original single-file preprocessing, pack samples, train HSAL
#   lrd_original   train HSAL+LRD on the original samples
#   ml1m_embeddings  run once, without sbatch, where sentence-transformers is installed:
#                    MODE=ml1m_embeddings bash anime_ml1m_jobs.sbatch
# Scavenger jobs can be stopped: --requeue restarts the job and --resume continues training.
# Samples live on the node's local disk (Newdata/ is bound there); archives/ keeps one tar per dataset.

WORKDIR=/projects/academic/haimonti/snirwade/HSAL-Chunking
: "${MODE:?set MODE, e.g. sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=hsal_chunked ...}"

# ---------- ML-1M item embeddings (same text and model as prepare_ml1m.py, row order of the HSAL CSV) ----------
if [ "$MODE" = ml1m_embeddings ]; then
    python - "${DATASET:-MovieLens_all_v2}" <<'PYEOF'
# Row i of the output = pd.factorize(item_id)[i], the numbering used by new_data1.py and finalize.
# Each CSV row is the same interaction as that row of ratings.dat, which gives item_id -> movie id.
import re, sys
import pandas as pd
from generate_llm_embeddings_v2 import default_device, embed_and_save
data, raw = sys.argv[1], 'Data/ml-1m'
df = pd.read_csv(f'Data/{data}.csv')
ratings = pd.read_csv(f'{raw}/ratings.dat', sep='::', engine='python', names=['user', 'movie', 'rating', 'ts'])
movies = pd.read_csv(f'{raw}/movies.dat', sep='::', engine='python', names=['movie', 'title', 'genres'], encoding='latin-1')
if len(df) != len(ratings):
    raise SystemExit(f'{len(df)} CSV rows, {len(ratings)} ratings')
pairs = pd.DataFrame({'item_id': df['item_id'].values, 'movie': ratings['movie'].values}).drop_duplicates()
if not (pairs['item_id'].is_unique and pairs['movie'].is_unique):
    raise SystemExit('item_id and movie id are not one to one')
item_to_movie = dict(zip(pairs['item_id'], pairs['movie']))
movie_to_text = {}
for row in movies.itertuples():
    title = re.sub(r'\s*\(\d{4}\)', '', row.title)
    movie_to_text[row.movie] = f"Title: {title}. Genres: {str(row.genres).replace('|', ', ')}."
_, unique_ids = pd.factorize(df['item_id'])
texts = [movie_to_text[item_to_movie[i]] for i in unique_ids]
print(f'{len(texts)} items; first: {texts[0]}')
device = default_device()
embed_and_save(texts, f'Data/{data}_llm_embeddings.pt', device)
embed_and_save(texts, f'Data/{data}_chunked_llm_embeddings.pt', device)
PYEOF
    exit $?
fi

# ---------- outside the container: start it and run this same file inside ----------
if [ -z "$IN_CONTAINER" ]; then
    echo "Job started : $(date)   host: $(hostname)   mode: $MODE"
    nvidia-smi
    export APPTAINER_CACHEDIR=$SLURMTMPDIR/apptainer_cache
    export APPTAINER_TMPDIR=$SLURMTMPDIR/apptainer_tmp
    export APPTAINERENV_IN_CONTAINER=1
    mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $SLURMTMPDIR/Newdata
    cd $WORKDIR
    apptainer exec --nv --writable-tmpfs -B /projects/academic/haimonti,$SLURMTMPDIR/Newdata:$WORKDIR/Newdata pytorch.sif bash $WORKDIR/anime_ml1m_jobs.sbatch
    status=$?
    echo "Job finished : $(date)   exit code: $status"
    exit $status
fi

# ---------- inside the container: shared settings (same values as the Goodreads jobs) ----------
set -eo pipefail
cd $WORKDIR
source venv/bin/activate
export LD_LIBRARY_PATH='/.singularity.d/libs'
mkdir -p save_models save_checkpoints markers logs archives verification
: "${DATASET:?set DATASET, e.g. sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=hsal_chunked ...}"
NAME=${DATASET}_chunked
N_CHUNKS=8
ITEM_LEN=50
USER_LEN=50
KHOP=2
EPOCHS=${EPOCHS:-10}
SEED=0
TRAIN_ARGS="--batchSize 256 --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --lr 0.001 --l2 0.0001 --layer_num 3 --hidden_size 50 --gpu 0 --seed $SEED --epoch $EPOCHS --model_record --resume"
step() { echo "[$(date +%F' '%T)] $*"; }
samples_dir() { echo "${1}_${ITEM_LEN}_${USER_LEN}_${KHOP}"; }
pack_samples() { local d=$(samples_dir $1); tar -cf archives/$d.tar.tmp -C Newdata $d && mv archives/$d.tar.tmp archives/$d.tar; step "packed $d: $(du -h archives/$d.tar | cut -f1)"; }
unpack_samples() { local d=$(samples_dir $1); [ -d Newdata/$d ] && return 0; [ -f archives/$d.tar ] || { step "archives/$d.tar missing"; exit 1; }; tar -xf archives/$d.tar -C Newdata; step "unpacked $d: $(find Newdata/$d -type f | wc -l) files"; }
wait_for_prep() {  # wait up to $2 minutes for another job's preprocessing marker
    for i in $(seq 1 $2); do
        [ -f markers/$1_prep_done ] && return 0
        [ $i -eq 1 ] && step "waiting for preprocessing of $1"
        sleep 60
    done
    step "preprocessing for $1 never finished"; exit 1
}
step "dataset $DATASET, mode $MODE, $EPOCHS epochs"

case "$MODE" in
hsal_chunked)
    if [ -f markers/${NAME}_prep_done ]; then
        step "preprocessing already done, skipping"
    else
        # a stopped job may have left a half-built chunked state: start it from scratch
        rm -rf Data/${DATASET}_chunks Data/${NAME}_chunked Data/${NAME}_graph ${NAME}_neg Series_table_sinusodial_Series_table_${NAME}.csv Newdata/$(samples_dir $NAME)
        step "split into $N_CHUNKS user-disjoint chunks"
        python split_hsal_csv_by_user.py --data $DATASET --n $N_CHUNKS
        for k in $(seq 1 $N_CHUNKS); do
            step "add chunk $k"
            python new_data1_chunked.py add-chunk --data $NAME --chunk Data/${DATASET}_chunks/${DATASET}_chunk_$k.csv
        done
        step "finalize";      python new_data1_chunked.py finalize --data $NAME --n_chunks $N_CHUNKS
        step "series table";  python sinusoidal_chunked.py --data $NAME --n_chunks $N_CHUNKS
        step "negatives";     python generate_neg_chunked.py --data $NAME --n_chunks $N_CHUNKS | grep -E "GB of|written"
        for k in $(seq 1 $N_CHUNKS); do
            step "samples for chunk $k"
            python new_data1_chunked.py samples --data $NAME --chunks $k --n_chunks $N_CHUNKS --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --job 16
        done
        pack_samples $NAME
        touch markers/${NAME}_prep_done
        step "preprocessing done"
    fi
    unpack_samples $NAME
    step "training HSAL"
    python -u new_main_BPR_chunked.py --data $NAME $TRAIN_ARGS
    ;;
lrd_chunked)
    wait_for_prep $NAME 240
    [ -f Data/${NAME}_llm_embeddings.pt ] || { step "Data/${NAME}_llm_embeddings.pt missing"; exit 1; }
    # the embeddings were made with the single-file numbering; finalize must give exactly the same book order
    python - <<PYEOF
import pandas as pd, torch
csv_order = list(pd.factorize(pd.read_csv('Data/${DATASET}.csv', dtype={'item_id': str})['item_id'])[1])
books = pd.read_csv('Data/${NAME}_chunked/books.csv', dtype={'raw_id': str}).sort_values('id')
emb = torch.load('Data/${NAME}_llm_embeddings.pt')
assert list(books['raw_id']) == csv_order, 'book order of the chunked ID map differs from the embeddings'
assert emb.shape[0] == len(books), f'embeddings have {emb.shape[0]} rows, ID map has {len(books)} books'
print(f'embedding check passed: {emb.shape[0]} books in the same order')
PYEOF
    unpack_samples $NAME
    step "training HSAL+LRD (bug-fixed, seeded)"
    python -u new_main_lrd_chunked.py --data $NAME $TRAIN_ARGS
    ;;
verify)
    # exits non-zero unless every check passes, so jobs that reuse the original outputs only start if it passes
    wait_for_prep $NAME 480
    unpack_samples $DATASET
    unpack_samples $NAME
    python verify_chunked_vs_original.py --orig $DATASET --data $NAME --n_check 100000000 2>&1 | grep -v -i warn | tee verification/${DATASET}.txt
    python - verification/${DATASET}.txt <<'PYEOF'
import re, sys
t = open(sys.argv[1]).read()
fails = []
for label in ['users: .*same set: True', 'books: .*same set: True', r'identical \(user, book, time\) edges: True',
              r'same \(user, position\) samples: True', 'series table: .*identical: True']:
    if not re.search(label, t):
        fails.append(label)
m = re.search(r'checked ([\d,]+) samples: same target ([\d,]+)\s+identical subgraph edges ([\d,]+)', t)
if not m or len({m.group(1), m.group(2), m.group(3)}) != 1:
    fails.append('samples: ' + (m.group(0) if m else 'line missing'))
n = re.search(r'negatives: identical unread-book sets for (\d+)/(\d+)', t)
if not n or n.group(1) != n.group(2):
    fails.append('negatives: ' + (n.group(0) if n else 'line missing'))
print('VERIFICATION', 'PASSED' if not fails else 'FAILED: ' + '; '.join(fails))
sys.exit(1 if fails else 0)
PYEOF
    step "verification passed"
    ;;
hsal_original)
    # new_data1.py cannot resume: if scavenger stops the job during preprocessing, preprocessing starts again
    if [ -f markers/${DATASET}_prep_done ]; then
        step "preprocessing already done, skipping"
    else
        rm -rf Data/${DATASET}_graph ${DATASET}_neg Series_table_sinusodial_Series_table_${DATASET}.csv Newdata/$(samples_dir $DATASET)
        step "generate_neg.py";  python generate_neg.py --data $DATASET | tail -1
        step "sinusoidal.py";    python sinusoidal.py --data $DATASET
        step "new_data1.py";     python new_data1.py --data $DATASET --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --job 16
        pack_samples $DATASET
        touch markers/${DATASET}_prep_done
        step "preprocessing done"
    fi
    unpack_samples $DATASET
    step "training HSAL"
    python -u new_main_BPR.py --data $DATASET $TRAIN_ARGS
    ;;
lrd_original)
    [ -f markers/${DATASET}_prep_done ] || { step "preprocessing for $DATASET not found"; exit 1; }
    [ -f Data/${DATASET}_llm_embeddings.pt ] || { step "Data/${DATASET}_llm_embeddings.pt missing"; exit 1; }
    unpack_samples $DATASET
    step "training HSAL+LRD (bug-fixed, seeded)"
    python -u new_main_lrd_fixed.py --data $DATASET $TRAIN_ARGS
    ;;
*)
    step "unknown MODE $MODE"; exit 1
    ;;
esac
step "done"
