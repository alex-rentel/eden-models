#!/bin/bash
#SBATCH --job-name=eden-eval-all
#SBATCH --account=YOUR_UMRCP_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eden-eval-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@umich.edu

# ============================================================
# Evaluate all 3 experiments side-by-side
# Run after all experiments complete
# ============================================================

echo "=== Eden Model Evaluation ==="
echo "Date: $(date)"

module load cuda/12.4
module load python/3.11
source ~/eden-models/venv/bin/activate

RESULTS_DIR=/scratch/$USER/eden-models/eval_results/$(date +%Y%m%d)
mkdir -p $RESULTS_DIR

CHECKPOINTS=(
    "qwen3-4b-eden-v1/final:Qwen3-4B-LoRA"
    "bonsai-eden-v1/final:Bonsai-1bit-LoRA"
    "eden-500m-v1/sft/final:Eden-500M-Scratch"
)

for entry in "${CHECKPOINTS[@]}"; do
    IFS=':' read -r ckpt name <<< "$entry"
    CKPT_PATH="/scratch/$USER/eden-models/checkpoints/$ckpt"
    
    if [ ! -d "$CKPT_PATH" ]; then
        echo "SKIP: $name — checkpoint not found at $CKPT_PATH"
        continue
    fi
    
    echo ""
    echo "=== Evaluating: $name ==="
    
    # Eden-Eval (400 custom test cases)
    echo "  Running Eden-Eval..."
    python eval/eden_eval.py \
        --model_path "$CKPT_PATH" \
        --test_cases eval/test_cases/eden_eval_400.jsonl \
        --output "$RESULTS_DIR/${name}_eden_eval.json" \
        2>&1 | tee -a "$RESULTS_DIR/${name}_log.txt"
    
    # BFCL benchmark
    echo "  Running BFCL..."
    python eval/bfcl_eval.py \
        --model_path "$CKPT_PATH" \
        --output "$RESULTS_DIR/${name}_bfcl.json" \
        2>&1 | tee -a "$RESULTS_DIR/${name}_log.txt"
    
    # Speed benchmark
    echo "  Running speed benchmark..."
    python eval/speed_bench.py \
        --model_path "$CKPT_PATH" \
        --output "$RESULTS_DIR/${name}_speed.json" \
        2>&1 | tee -a "$RESULTS_DIR/${name}_log.txt"
    
    echo "  Done: $name"
done

# Generate comparison report
echo ""
echo "=== Generating comparison report ==="
python eval/compare_experiments.py \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/comparison_report.md"

echo ""
echo "=== All evaluations complete ==="
echo "Results: $RESULTS_DIR"
echo "Report: $RESULTS_DIR/comparison_report.md"

# Copy to Turbo
cp -r "$RESULTS_DIR" /nfs/turbo/$USER/eden-models/eval_results/
echo "Results copied to Turbo."
