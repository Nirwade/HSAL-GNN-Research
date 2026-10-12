#!/bin/bash
# usage: run_in_container.sh <script-inside-repo> [args...]   (called from the sbatch files)
set -e
export APPTAINER_CACHEDIR=/vscratch/grp-haimonti/$USER/apptainer_cache
export APPTAINER_TMPDIR=/vscratch/grp-haimonti/$USER/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
cd /vscratch/grp-haimonti/$USER/HSAL-GNN-Research
apptainer exec --nv --writable-tmpfs --bind /projects/academic/haimonti pytorch.sif bash -c "
set -e
cd /vscratch/grp-haimonti/\$USER/HSAL-GNN-Research
source venv/bin/activate
export LD_LIBRARY_PATH='/.singularity.d/libs'
mkdir -p save_models save_checkpoints results logs
bash $*
"
