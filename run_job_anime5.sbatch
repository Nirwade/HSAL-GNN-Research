#!/bin/bash
#SBATCH --cluster=ub-hpc
#SBATCH --account=haimonti
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=MAL5ep
#SBATCH --output=run_job_%x_%j.out
#SBATCH --error=run_job_%x_%j.err
#SBATCH --mail-user=snirwade@buffalo.edu
#SBATCH --mail-type=ALL

echo "=================================================="
echo "Job started : $(date)"
echo "Hostname    : $(hostname)"
echo "Directory   : $(pwd)"
echo "=================================================="

echo "GPU Information"
nvidia-smi
echo "=================================================="

export APPTAINER_CACHEDIR=/vscratch/grp-haimonti/$USER/apptainer_cache
export APPTAINER_TMPDIR=/vscratch/grp-haimonti/$USER/apptainer_tmp

mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR

DATASET=${DATASET:-Goodreads_young_adult_HSAL_100k}
ITEM_LEN=50
USER_LEN=50
KHOP=2

NEG_FILE="${DATASET}_neg"
GRAPH_FILE="Data/${DATASET}_graph"
DATA_DIR="Newdata/${DATASET}_${ITEM_LEN}_${USER_LEN}_${KHOP}"

echo "Starting Apptainer..."
apptainer exec \
    --nv \
    --writable-tmpfs \
    pytorch.sif \
    bash -c "

cd /vscratch/grp-haimonti/\$USER/HSAL-GNN-Research

source venv/bin/activate

mkdir -p save_models

export LD_LIBRARY_PATH='/.singularity.d/libs'

echo '=================================================='
echo 'Inside container'

echo 'Current directory:'
pwd

echo
echo 'Python executable:'
which python

echo
echo 'Python version:'
python --version

echo
echo 'Torch version:'
python -c \"import torch; print(torch.__version__)\"

echo
echo 'CUDA available:'
python -c \"import torch; print(torch.cuda.is_available())\"

echo
echo 'Available datasets:'
ls Data

echo
echo 'Checking required datasets...'

for file in \
    MAL_HSAL_Final.csv \
    Goodreads_Fantasy_HSAL_Final.csv \
    Goodreads_Mystery_HSAL_Final.csv \
    Goodreads_Romance_HSAL_Final.csv \
    Goodreads_YoungAdult_HSAL_Final.csv \
    Goodreads_fantasy_paranormal_HSAL_100k.csv \
    Goodreads_mystery_HSAL_100k.csv \
    Goodreads_romance_HSAL_100k.csv \
    Goodreads_young_adult_HSAL_100k.csv
do
    if [ -f Data/\$file ]; then
        echo \"[FOUND] \$file\"
    else
        echo \"[MISSING] \$file\"
    fi
done

echo
echo '=================================================='
echo 'Checking preprocessing status...'
echo '=================================================='

if [ -f $NEG_FILE ] && [ -f $GRAPH_FILE ] && [ -d $DATA_DIR/train ] && [ -d $DATA_DIR/val ] && [ -d $DATA_DIR/test ]; then
    echo 'Preprocessed files found.'
    echo 'Skipping preprocessing.'
else
    echo 'Preprocessed files not found.'
    echo 'Running preprocessing...'
    python generate_neg.py --data $DATASET
    python sinusoidal.py --data $DATASET
    python new_data1.py \
        --data $DATASET \
        --item_max_length $ITEM_LEN \
        --user_max_length $USER_LEN \
        --k_hop $KHOP
    echo 'Preprocessing complete.'
fi
echo

echo '=================================================='
echo 'Starting training...'
echo 'Dataset: $DATASET'
echo '=================================================='

python -u new_main_BPR.py \
    --data $DATASET \
    --batchSize 256 \
    --item_max_length $ITEM_LEN \
    --user_max_length $USER_LEN \
    --k_hop $KHOP \
    --lr 0.001 \
    --l2 0.0001 \
    --layer_num 3 \
    --hidden_size 50 \
    --gpu 0 \
    --seed ${RUN_ID:-0} \
    --epoch 5 \
    --model_record \

"

echo "=================================================="
echo "Training completed successfully."
echo "Job finished : $(date)"
echo "=================================================="
