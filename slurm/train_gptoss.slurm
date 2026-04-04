#!/bin/bash
#SBATCH --job-name=eden-gptoss-sft
#SBATCH --account=YOUR_UMRCP_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1               # 1x A100 80GB for QLoRA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eden-gptoss-%j.log
#SBATCH --error=logs/eden-gptoss-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu

# ============================================================
# Experiment 4: Fine-tune GPT-OSS 20B for tool calling
# Base: openai/gpt-oss-20b (Apache 2.0, MoE, tool-use optimized)
# Target: Great Lakes HPC — 1x A100 80GB
# Duration: ~16-24 hours
# ============================================================

echo "=== Experiment 4: GPT-OSS 20B LoRA SFT ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load cuda/12.2
module load python/3.11
source ~/eden-models/venv/bin/activate

export HF_HOME=/scratch/$USER/hf_cache
export WANDB_PROJECT=eden-models
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/$USER/eden-models/checkpoints/gptoss-20b-eden-v1
mkdir -p logs

python training/sft.py \
    --config configs/gpt-oss-20b-lora.yaml \
    --data_path /scratch/$USER/eden-models/data/combined_sft.jsonl \
    --output_dir /scratch/$USER/eden-models/checkpoints/gptoss-20b-eden-v1 \
    --wandb_run_name "gptoss-20b-eden-sft-$(date +%Y%m%d)" \
    2>&1 | tee logs/gptoss_$(date +%Y%m%d_%H%M%S).log

echo "=== Experiment 4 complete ==="
cp -r /scratch/$USER/eden-models/checkpoints/gptoss-20b-eden-v1/final \
      /nfs/turbo/$USER/eden-models/checkpoints/gptoss-20b-eden-v1-final/
echo "Checkpoint saved to Turbo."
