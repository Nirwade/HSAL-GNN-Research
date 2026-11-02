#!/bin/bash
#SBATCH --cluster=faculty
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --job-name=sequel_llama_A
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

REPO_DIR="/vscratch/grp-haimonti/snirwade/llm_relation_pipeline_ccr/llm_relation_pipeline"
OLLAMA_SIF="/vscratch/grp-haimonti/snirwade/HSAL-GNN-Research/ollama.sif"
PORT=11444

cd "$REPO_DIR"
export APPTAINER_CACHEDIR="/vscratch/grp-haimonti/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/vscratch/grp-haimonti/$USER/apptainer_tmp"
export OLLAMA_URL="http://127.0.0.1:${PORT}/api/chat"

apptainer exec --nv --env OLLAMA_NUM_PARALLEL=4 --env OLLAMA_HOST=127.0.0.1:${PORT} "$OLLAMA_SIF" ollama serve &
SERVER_PID=$!

cleanup() {
    kill $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:${PORT}/api/tags" > /dev/null 2>&1; then
        echo "Ollama server is up on port ${PORT}"
        break
    fi
    sleep 2
done

python3 extract_sequel.py --version A --model llama3.1:8b --workers 4
