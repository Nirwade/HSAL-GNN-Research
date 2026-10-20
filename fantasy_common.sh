# Settings shared by both Fantasy runs (sourced by fantasy_baseline_steps.sh and fantasy_chunked_steps.sh)
set -eo pipefail
cd /vscratch/grp-haimonti/snirwade/HSAL-Chunking
source venv/bin/activate
export LD_LIBRARY_PATH='/.singularity.d/libs'
mkdir -p save_models save_checkpoints markers logs
DATASET=Goodreads_Fantasy_HSAL_Final
ITEM_LEN=50
USER_LEN=50
KHOP=2
EPOCHS=10
SEED=0
TRAIN_ARGS="--batchSize 256 --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --lr 0.001 --l2 0.0001 --layer_num 3 --hidden_size 50 --gpu 0 --seed $SEED --epoch $EPOCHS --model_record --resume"
step() { echo "[$(date +%F' '%T)] $*"; }
