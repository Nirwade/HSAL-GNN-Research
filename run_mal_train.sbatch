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
#SBATCH --time=9:45:00
#SBATCH --job-name=MAL_train
#SBATCH --output=mal_train_%j.out
#SBATCH --error=mal_train_%j.err

apptainer exec --nv pytorch.sif bash -c "
cd /vscratch/grp-haimonti/\$USER/HSAL-GNN-Research
source venv/bin/activate
mkdir -p save_models
python -u new_main_BPR.py --data MAL_HSAL_Final --batchSize 256 --item_max_length 50 --user_max_length 50 --k_hop 2 --lr 0.001 --l2 0.0001 --layer_num 3 --hidden_size 50 --gpu 0 --epoch 50 --model_record --resume
"
