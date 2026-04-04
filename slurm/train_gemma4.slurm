#!/bin/bash
#SBATCH --job-name=eden-gemma4-sft
#SBATCH --account=YOUR_UMRCP_ACCOUNT    # Replace with your SLURM account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1               # 1x A100 80GB for QLoRA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eden-gemma4-%j.log
#SBATCH --error=logs/eden-gemma4-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu

# ============================================================
# Experiment 1: Fine-tune Gemma 4 E4B for tool calling
# Base: google/gemma-4-E4B-it (Apache 2.0, native function calling)
# Target: Great Lakes HPC — 1x A100 80GB
# Duration: ~8-16 hours
# ============================================================

echo "=== Experiment 1: Gemma 4 E4B LoRA SFT ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Load modules
module load cuda/12.2
module load python/3.11

# Activate virtual environment
source ~/eden-models/venv/bin/activate

# Set environment
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export WANDB_PROJECT=eden-models
export TOKENIZERS_PARALLELISM=false

# Create output directories
mkdir -p /scratch/$USER/eden-models/checkpoints
mkdir -p /scratch/$USER/eden-models/logs

# Run training
python training/sft.py \
    --config configs/gemma4-e4b-lora.yaml \
    --data_path /scratch/$USER/eden-models/data/combined_sft.jsonl \
    --output_dir /scratch/$USER/eden-models/checkpoints/gemma4-e4b-eden-v1 \
    --wandb_run_name "gemma4-e4b-eden-sft-$(date +%Y%m%d)" \
    2>&1 | tee logs/gemma4_$(date +%Y%m%d_%H%M%S).log

echo "=== Training complete ==="
echo "Date: $(date)"

# Copy final checkpoint to Turbo for persistence
echo "Copying checkpoint to Turbo storage..."
cp -r /scratch/$USER/eden-models/checkpoints/gemma4-e4b-eden-v1/final \
      /nfs/turbo/$USER/eden-models/checkpoints/gemma4-e4b-eden-v1-final/

echo "Done. Checkpoint saved to Turbo."
