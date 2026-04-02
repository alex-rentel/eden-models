#!/bin/bash
#SBATCH --job-name=eden-bonsai-lora
#SBATCH --account=YOUR_UMRCP_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                    # 1x RTX PRO 6000 Blackwell
#SBATCH --constraint=rtx6000_blackwell  # Request specific GPU type
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eden-bonsai-%j.log
#SBATCH --error=logs/eden-bonsai-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu

# ============================================================
# Experiment 2: Fine-tune Bonsai-1.7B (1-bit) for tool calling
# Target: RTX PRO 6000 Blackwell — FP4 native Tensor Cores
# Duration: ~4 hours
# Purpose: Can 1-bit match full-precision on tool calling?
# ============================================================

echo "=== Experiment 2: Bonsai 1-bit LoRA ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load cuda/12.4
module load python/3.11
source ~/eden-models/venv/bin/activate

export HF_HOME=/scratch/$USER/hf_cache
export WANDB_PROJECT=eden-models
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/$USER/eden-models/checkpoints/bonsai-eden-v1
mkdir -p logs

python training/sft.py \
    --config configs/bonsai-1bit-lora.yaml \
    --data_path /scratch/$USER/eden-models/data/combined_sft.jsonl \
    --output_dir /scratch/$USER/eden-models/checkpoints/bonsai-eden-v1 \
    --wandb_run_name "bonsai-1bit-eden-sft-$(date +%Y%m%d)" \
    2>&1 | tee logs/bonsai_$(date +%Y%m%d_%H%M%S).log

echo "=== Experiment 2 complete ==="
cp -r /scratch/$USER/eden-models/checkpoints/bonsai-eden-v1/final \
      /nfs/turbo/$USER/eden-models/checkpoints/bonsai-eden-v1-final/
echo "Checkpoint saved to Turbo."
