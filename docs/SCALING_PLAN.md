# Eden Models — Scaling Plan for Great Lakes HPC

## Best Configuration Found (M1 Max, 1.7B model)

### Winning config: v4a
```yaml
model: Qwen3-1.7B-4bit
rank: 8
scale: 10.0
num_layers: 8
learning_rate: 2e-6
iters: 400
batch_size: 4
mask_prompt: true
data: 15K Eden-specific synthetic examples (template-generated)
```

### Results Summary

| Model | Overall | Single | Args | Multi | No-Tool | Wrong |
|-------|---------|--------|------|-------|---------|-------|
| Base (no adapter) | 72% | 70% | 90% | 40% | 60% | 100% |
| v3 (generic xlam) | 78% | 60% | 100% | 100% | 70% | 100% |
| v4 (Eden, 200 iter) | 78% | 65% | 100% | 100% | 60% | 100% |
| **v4a (Eden, 400 iter)** | **80%** | 60% | 100% | 100% | **80%** | 100% |
| v4b (Eden, rank=16) | 68% | 55% | 70% | 20% | 100% | 100% |

### Key Findings

1. **Rank 8 >> rank 16** for 1.7B model. Higher rank causes catastrophic forgetting.
2. **400 iters > 200 iters** with this data. Val loss still dropping at 400.
3. **Eden-specific data** improves no-tool judgment (+20 over base) while maintaining tool-calling.
4. **Template-based data works** — 15K synthetic examples match 51K generic examples.
5. **mask_prompt=true is critical** — trains only on completions, not prompt tokens.

## Data Pipeline

### What Worked
- Template-based generation (zero API cost, infinite scale)
- Distribution: 50% single-tool, 15% multi-tool, 15% no-tool, 10% error-recovery, 10% clarification
- All 9 Eden tools covered with realistic args and results
- Qwen3-native format via `tokenizer.apply_chat_template(messages, tools=tools)`

### Data Format (mlx_lm.lora ChatDataset)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
    {"role": "tool", "name": "bash", "content": "file1.py\nfile2.py"},
    {"role": "assistant", "content": "Here are the files."}
  ],
  "tools": [{"type": "function", "function": {"name": "bash", ...}}]
}
```

Note: `arguments` must be a JSON **string**, not an object.

## Great Lakes Scaling Plan

### Phase 1: Qwen3-4B (1x A100 80GB)

```yaml
model: Qwen/Qwen3-4B   # full precision on A100
rank: 16                 # can go higher with 4B model
scale: 20.0
num_layers: 16
learning_rate: 1e-5      # can be more aggressive with larger model
iters: 1000
batch_size: 8
grad_accumulation_steps: 4  # effective batch=32
max_seq_length: 4096
mask_prompt: true
optimizer: adamw
```

Expected: ~4 hours on A100, ~$12 at Great Lakes rates.

### Phase 2: Qwen3-8B (1x A100 80GB, QLoRA)

```yaml
model: Qwen/Qwen3-8B
quantization: 4bit       # QLoRA to fit in 80GB
rank: 32
num_layers: -1            # all layers
learning_rate: 5e-6
iters: 2000
batch_size: 4
grad_accumulation_steps: 8
```

Expected: ~8-12 hours on A100.

### Phase 3: Scale Data (50K+ examples)

1. Expand template generator with more diverse scenarios
2. Add multi-turn conversations (3+ tool calls)
3. Add domain-specific data (web dev, data science, DevOps)
4. Consider Claude API generation for highest-quality examples ($100-200 for 50K)

### Phase 4: Multi-Model Comparison

Train same data on:
- Qwen3-1.7B, 4B, 8B
- Llama-3.2-1B, 3B
- Phi-3.5-mini (3.8B)
- Gemma-2-2B, 9B

Compare: accuracy vs model size vs training time vs inference speed.

## Remaining Failure Modes

### Persistent failures across all versions:
1. **read_file confusion** — model calls `list_files` instead of `file_read` for file reading
2. **run_command refusal** — refuses to use `bash` for pip install, git status
3. **web_search avoidance** — answers from knowledge instead of searching
4. **translate bypass** — translates directly instead of using translate tool
5. **command_with_pipe** — uses `list_files` instead of `bash` for complex commands

### Fixes for next iteration:
1. Add more `file_read` examples with explicit "read", "show contents", "cat" queries
2. Add bash examples for pip, git, and complex piped commands
3. Add web_search examples for current events and version queries
4. Add translate examples where using the tool is clearly better than direct answer

## Resource Estimates

| Model | Hardware | Data | Time | Cost |
|-------|----------|------|------|------|
| Qwen3-1.7B | M1 Max 64GB | 15K | 30 min | $0 |
| Qwen3-4B | A100 80GB | 15K | 4 hr | $12 |
| Qwen3-8B | A100 80GB | 50K | 12 hr | $36 |
| Full sweep (6 models) | 2x A100 | 50K | 48 hr | $150 |

## Reproduction

```bash
# Generate data
python3 data/generation/generate_eden_data.py --num 15000 --output data/eden_15k.jsonl

# Split
python3 -c "
import json, random; random.seed(42)
data = [json.loads(l) for l in open('data/eden_15k.jsonl')]
random.shuffle(data); split = int(len(data)*0.9)
open('data/train.jsonl','w').writelines(json.dumps(d)+'\n' for d in data[:split])
open('data/valid.jsonl','w').writelines(json.dumps(d)+'\n' for d in data[split:])
"

# Train (best config)
python3 -m mlx_lm.lora -c configs/eden-v4a-400iter.yaml

# Evaluate
python3 eval/tool_calling_eval.py \
  --model mlx-community/Qwen3-1.7B-4bit \
  --adapter-path adapters/qwen3-1.7b-tools-v4a \
  --output eval/results/v4a_results.json
```
