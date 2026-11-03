#!/bin/bash
#SBATCH --cluster=faculty
#SBATCH --account=haimonti
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=prequel_mistral_B
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/vscratch/grp-haimonti/snirwade/llm_relation_pipeline_ccr/llm_relation_pipeline}"
OLLAMA_SIF="${OLLAMA_SIF:-/vscratch/grp-haimonti/snirwade/HSAL-GNN-Research/ollama.sif}"
PORT=11437
SERVER_LOG="${SERVER_LOG:-${REPO_DIR}/slurm_logs/ollama_server_${SLURM_JOB_ID}.log}"

export APPTAINER_CACHEDIR="/vscratch/grp-haimonti/$USER/apptainer_cache"
export APPTAINER_TMPDIR="/vscratch/grp-haimonti/$USER/apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

cd "$REPO_DIR"
export OLLAMA_URL="http://127.0.0.1:${PORT}/api/chat"

echo "Job started on $(hostname) at $(date), port $PORT"
nvidia-smi

apptainer exec --nv --env OLLAMA_NUM_PARALLEL=4 --env OLLAMA_HOST=127.0.0.1:${PORT} \
  "$OLLAMA_SIF" ollama serve > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Stopping Ollama server (pid $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 30); do
  if curl -s "http://127.0.0.1:${PORT}/api/tags" > /dev/null; then
    echo "Ollama server is up on port $PORT."
    break
  fi
  sleep 2
done

python3 src/extract_prequel.py --version B --model mistral --workers 4

echo "Job finished at $(date)"
