# Eden Models — Overnight Training Results

**Date:** 2026-04-03  
**Machine:** M1 Max 64GB, macOS  
**Framework:** mlx-lm 0.31.1  
**Base model:** mlx-community/Qwen3-1.7B-4bit (~1GB)

## Summary

Fine-tuned Qwen3-1.7B-4bit on tool-calling data using MLX LoRA. Three iterations of increasing refinement:

| Model | Overall | Single Tool | Arg Format | Multi-Tool | No Tool | Wrong Tool |
|-------|---------|-------------|------------|------------|---------|------------|
| **Base (no adapter)** | **72% (36/50)** | 70% (14/20) | 90% (9/10) | 40% (2/5) | 60% (6/10) | 100% (5/5) |
| v1 (aggressive) | 48% (24/50) | 25% (5/20) | 40% (4/10) | 0% (0/5) | 100% (10/10) | 100% (5/5) |
| v2 (rebalanced) | 36% (18/50) | 0% (0/20) | 30% (3/10) | 0% (0/5) | 100% (10/10) | 100% (5/5) |
| **v3 (conservative)** | **78% (39/50)** | 60% (12/20) | **100% (10/10)** | **100% (5/5)** | 70% (7/10) | 100% (5/5) |

**Best result: v3 at 78% (+6 points over base)**

## Training Configs

### v1 — Aggressive (catastrophic forgetting)
```yaml
rank: 16, scale: 20.0, num_layers: 16
lr: 1e-5, iters: 1000, batch_size: 4
data: 113K examples (50% tool-calling, 50% no-tool/refusal)
```
- Val loss: 2.142 → 0.506 (best at iter 600) → 0.570 (final)
- Peak memory: 33.8 GB
- **Result:** Model learned to refuse tool calls. Catastrophic forgetting of base tool-calling ability.

### v2 — Rebalanced data (still too aggressive)
```yaml
rank: 16, scale: 20.0, num_layers: 16
lr: 1e-5, iters: 1000, batch_size: 4
data: 64K examples (80% tool-calling, 10% refusal, 10% no-tool)
```
- Val loss: 2.646 → 0.211 (best at iter 800) → 0.465 (final)
- Peak memory: 33.8 GB
- **Result:** Still refused tool calls despite rebalanced data. High rank + LR + iters caused forgetting.

### v3 — Conservative (preserved + improved)
```yaml
rank: 8, scale: 10.0, num_layers: 8
lr: 2e-6, iters: 200, batch_size: 4
data: 51K examples (100% tool-calling)
```
- Val loss: 3.239 → 1.125 → 0.715 → 0.549 → 0.469 (monotonically decreasing)
- Peak memory: 9.0 GB
- Training time: ~12 minutes
- **Result:** +6% overall, +60 points on multi-tool, +10 points on arg formatting. Slight regression on single-tool selection.

## Loss Curves

### v1
| Iter | Train Loss | Val Loss |
|------|-----------|----------|
| 1    | —         | 2.142    |
| 100  | 0.437     | —        |
| 200  | 0.297     | 0.551    |
| 400  | 0.530     | 0.543    |
| 600  | 0.414     | 0.506    |
| 800  | 0.223     | 0.519    |
| 1000 | 0.246     | 0.570    |

### v3 (best)
| Iter | Train Loss | Val Loss |
|------|-----------|----------|
| 1    | —         | 3.239    |
| 50   | 0.987     | 1.125    |
| 100  | 0.803     | 0.715    |
| 150  | —         | 0.549    |
| 200  | 0.218     | 0.469    |

## Inference Speed

With LoRA adapter loaded:
- ~55 tok/sec during training
- ~40 tok/sec generation with batch=1
- Adapter size: ~10MB (v3, rank 8) / ~40MB (v1/v2, rank 16)

## Key Learnings

### What worked
1. **Conservative LoRA is critical for small models.** Rank 8, 8 layers, LR 2e-6 preserved the base model's reasoning while improving tool formatting.
2. **Only train on what you want to improve.** 100% tool-calling data for v3 avoided teaching refusal patterns.
3. **Qwen3's native tool-calling format is excellent.** The base model already uses `<tool_call>` XML tags via its chat template — fine-tuning should enhance this, not replace it.
4. **mlx_lm.lora's ChatDataset with `tools` key works perfectly.** The `tools` field is passed to `apply_chat_template()` and renders Qwen3-native format automatically.

### What didn't work
1. **Glaive dataset has too many refusal examples.** ~50% of examples are "I can't do that" responses, which teaches the model to refuse.
2. **Aggressive LoRA destroys small model capabilities.** Rank 16, LR 1e-5, 1000 iters on a 1.7B model causes catastrophic forgetting.
3. **Overfitting appears around 600-800 iters** even with 50K+ training examples.
4. **Data composition matters more than quantity.** 51K pure tool-calling examples (v3) beat 113K mixed examples (v1).

### What to try next
1. **Higher quality tool-calling datasets** — filter for examples where the assistant actually reasons about which tool to use (not just responds)
2. **Qwen3-4B or Qwen3-8B** — larger models may be more resilient to catastrophic forgetting
3. **Synthetic data generation** — use a large model to generate Qwen3-native format training data with the Eden tool schema
4. **DPO/preference tuning** — train the model to prefer tool calls over refusals
5. **Scale to Great Lakes** — use the configs in `configs/qwen3-4b-lora.yaml` for A100 training with FlashAttention
6. **Extended v3 training** — the val loss was still decreasing at 200 iters; try 500 iters with the same conservative settings

## Reproduction

```bash
# Generate training data
python3 data/processing/format_for_mlx_lora.py
python3 data/processing/rebalance_data.py

# Train v3 (best model)
python3 -m mlx_lm.lora -c configs/overnight-m1max-v3.yaml

# Evaluate
python3 eval/tool_calling_eval.py \
  --model mlx-community/Qwen3-1.7B-4bit \
  --output eval/results/base_results.json

python3 eval/tool_calling_eval.py \
  --model mlx-community/Qwen3-1.7B-4bit \
  --adapter-path adapters/qwen3-1.7b-tools-v3 \
  --output eval/results/v3_results.json
```

## File Structure

```
eden-models/
├── configs/
│   ├── overnight-m1max.yaml      # v1 config
│   ├── overnight-m1max-v2.yaml   # v2 config  
│   └── overnight-m1max-v3.yaml   # v3 config (best)
├── data/
│   ├── processing/
│   │   ├── format_for_mlx_lora.py   # Glaive → ChatDataset converter
│   │   └── rebalance_data.py        # Data rebalancing script
│   └── schemas/eden_tools.json      # Eden tool definitions
├── eval/
│   ├── tool_calling_eval.py         # 50-case eval suite
│   └── results/                     # JSON results per model
├── adapters/
│   ├── qwen3-1.7b-tools-v1/        # v1 adapter (40MB)
│   ├── qwen3-1.7b-tools-v2/        # v2 adapter (40MB)
│   └── qwen3-1.7b-tools-v3/        # v3 adapter (10MB, best)
├── logs/                            # Training logs
├── docs/
│   └── MLX_LORA_API.md             # mlx_lm.lora API documentation
└── RESULTS.md                       # This file
```
