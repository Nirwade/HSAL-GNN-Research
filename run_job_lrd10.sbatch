#!/bin/bash
#SBATCH --cluster=ub-hpc
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=LRD10ep
#SBATCH --output=run_job_%x_%j.out
#SBATCH --error=run_job_%x_%j.err
#SBATCH --mail-user=snirwade@buffalo.edu
#SBATCH --mail-type=ALL

echo "=================================================="
echo "Job started : $(date)"
echo "Hostname    : $(hostname)"
echo "=================================================="

nvidia-smi
echo "=================================================="

export APPTAINER_CACHEDIR=/vscratch/grp-haimonti/$USER/apptainer_cache
export APPTAINER_TMPDIR=/vscratch/grp-haimonti/$USER/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR
mkdir -p $APPTAINER_TMPDIR

cd /vscratch/grp-haimonti/$USER/HSAL-GNN-Research

apptainer exec --nv --writable-tmpfs pytorch.sif bash -c "
cd /vscratch/grp-haimonti/\$USER/HSAL-GNN-Research
source venv/bin/activate
export LD_LIBRARY_PATH='/.singularity.d/libs'

mkdir -p save_checkpoints save_models

python -u new_main_lrd.py \
    --data MAL_HSAL_Final \
    --batchSize 16 \
    --item_max_length 50 \
    --user_max_length 50 \
    --k_hop 2 \
    --lr 0.001 \
    --l2 0.0001 \
    --layer_num 3 \
    --hidden_size 50 \
    --gpu 0 \
    --epoch 10 \
    --model_record \
    --resume
"

echo "=================================================="
echo "Job finished : $(date)"
echo "=================================================="
