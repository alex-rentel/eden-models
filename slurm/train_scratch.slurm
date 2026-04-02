#!/bin/bash
#SBATCH --job-name=eden-scratch-500m
#SBATCH --account=YOUR_UMRCP_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                    # 1x RTX PRO 6000 Blackwell
#SBATCH --constraint=rtx6000_blackwell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00                 # 3 days max
#SBATCH --output=logs/eden-scratch-%j.log
#SBATCH --error=logs/eden-scratch-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu

# ============================================================
# Experiment 3: Train Eden-500M from scratch (1.58-bit)
# Target: RTX PRO 6000 Blackwell — FP4 native Tensor Cores
# Duration: ~24-48 hours
# Purpose: Is purpose-built 1-bit viable for tool calling?
# ============================================================

echo "=== Experiment 3: From-Scratch 1-bit 500M ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load cuda/12.4
module load python/3.11
source ~/eden-models/venv/bin/activate

export HF_HOME=/scratch/$USER/hf_cache
export WANDB_PROJECT=eden-models
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/$USER/eden-models/checkpoints/eden-500m-v1
mkdir -p logs

# Stage 1: Pre-training on general data (1B tokens)
echo "=== Stage 1: Pre-training ==="
python training/bitnet.py \
    --config configs/eden-500m-scratch.yaml \
    --stage pretrain \
    --data_path /scratch/$USER/eden-models/data/slimpajama_sample_1B.jsonl \
    --output_dir /scratch/$USER/eden-models/checkpoints/eden-500m-v1/pretrain \
    --wandb_run_name "eden-500m-pretrain-$(date +%Y%m%d)" \
    2>&1 | tee logs/scratch_pretrain_$(date +%Y%m%d_%H%M%S).log

# Stage 2: SFT on tool-calling data
echo "=== Stage 2: Tool-calling SFT ==="
python training/sft.py \
    --config configs/eden-500m-scratch.yaml \
    --stage sft \
    --model_path /scratch/$USER/eden-models/checkpoints/eden-500m-v1/pretrain/final \
    --data_path /scratch/$USER/eden-models/data/combined_sft.jsonl \
    --output_dir /scratch/$USER/eden-models/checkpoints/eden-500m-v1/sft \
    --wandb_run_name "eden-500m-sft-$(date +%Y%m%d)" \
    2>&1 | tee logs/scratch_sft_$(date +%Y%m%d_%H%M%S).log

echo "=== Experiment 3 complete ==="
cp -r /scratch/$USER/eden-models/checkpoints/eden-500m-v1 \
      /nfs/turbo/$USER/eden-models/checkpoints/eden-500m-v1/
echo "All checkpoints saved to Turbo."
