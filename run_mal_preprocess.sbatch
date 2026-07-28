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
#SBATCH --time=08:00:00
#SBATCH --job-name=MAL_preprocess
#SBATCH --output=mal_preprocess_%j.out
#SBATCH --error=mal_preprocess_%j.err

apptainer exec --nv pytorch.sif bash -c "
cd /vscratch/grp-haimonti/\$USER/HSAL-GNN-Research
source venv/bin/activate
python new_data1.py --data MAL_HSAL_Final --item_max_length 50 --user_max_length 50 --k_hop 2
"
