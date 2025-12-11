#!/bin/bash

#SBATCH --job-name=squai_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=capella
#SBATCH --time=00:30:00
#SBATCH --output=squai_output_%j.log

# --- 1. Load Modules (From your README) ---
echo "Loading modules..."
module purge
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2

# Fix for the "libbz2" error you saw earlier (just in case)
module load bzip2

# --- 2. Activate Environment ---
# Assumes the folder 'env' exists in the current directory
if [ -f "env/bin/activate" ]; then
    source env/bin/activate
else
    echo "Error: 'env/bin/activate' not found. Please run 'python -m venv env' first."
    exit 1
fi

# Optional: Ensure dependencies are installed (uncomment if needed)
# pip install -r requirements.txt

# --- 3. Run the Command ---
echo "Starting SQuAI Inference..."

python run_SQuAI.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n 0.5 \
    --alpha 0.65 \
    --top_k 20 \
    --single_question "What are the measurable performance differences between dense retrieval and sparse BM25 retrieval when searching for exact LaTeX equations in a multi-million document corpus?"

echo "Job finished."