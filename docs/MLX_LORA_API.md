# mlx_lm.lora API Analysis

> mlx-lm v0.31.1 — Apple MLX LoRA/QLoRA fine-tuning

## CLI Usage

```bash
python3 -m mlx_lm lora [args]    # preferred
mlx_lm.lora [args]               # also works
python3 -m mlx_lm.lora [args]    # deprecated but functional
```

## Data Format

mlx_lm.lora expects `{train,valid,test}.jsonl` files in the `--data` directory. Each line must be one of three formats, auto-detected from the first example:

### 1. Chat Format (what we use for tool-calling)

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "tools": [...]}
```

- The `"tools"` key is **optional** and gets passed directly to `tokenizer.apply_chat_template(messages, tools=tools)`
- Qwen3's tokenizer handles tool formatting natively via its chat template
- `--mask-prompt` works: masks all tokens before the final assistant turn

### 2. Completions Format

```json
{"prompt": "...", "completion": "..."}
```

### 3. Text Format

```json
{"text": "full text to train on"}
```

## LoRA Configuration

Default parameters (from `CONFIG_DEFAULTS`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `rank` | 8 | LoRA rank (r) |
| `scale` | 20.0 | Direct multiplier on LoRA output (NOT alpha/rank) |
| `dropout` | 0.0 | LoRA dropout |
| `keys` | auto-detected | Which layers to apply LoRA to |

Override via YAML config:

```yaml
lora_parameters:
  rank: 16
  scale: 20.0
  dropout: 0.0
  keys:
    - self_attn.q_proj
    - self_attn.k_proj
    - self_attn.v_proj
    - self_attn.o_proj
```

If `keys` is omitted, LoRA is applied to ALL Linear/QuantizedLinear layers found in the last `num_layers` transformer blocks.

## Training Defaults

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| `model` | Qwen/Qwen3-0.6b | `--model` |
| `num_layers` | 16 | `--num-layers` |
| `batch_size` | 4 | `--batch-size` |
| `iters` | 1000 | `--iters` |
| `learning_rate` | 1e-5 | `--learning-rate` |
| `steps_per_report` | 10 | `--steps-per-report` |
| `steps_per_eval` | 200 | `--steps-per-eval` |
| `save_every` | 100 | `--save-every` |
| `max_seq_length` | 2048 | `--max-seq-length` |
| `optimizer` | adam | `--optimizer` |
| `grad_checkpoint` | false | `--grad-checkpoint` |

## Adapter Output

Saved to `--adapter-path` directory:
- `adapters.safetensors` — the LoRA weights
- `adapter_config.json` — full config for reproducibility

## Loading Fine-tuned Model

```bash
python3 -m mlx_lm.generate \
  --model mlx-community/Qwen3-1.7B-4bit \
  --adapter-path adapters/qwen3-1.7b-tools-v1 \
  --prompt "..."
```

## Key Architectural Notes

1. **LoRA scale**: The `scale` parameter is a direct multiplier (`y + scale * (x @ A) @ B`), not alpha/rank. Default 20.0 is tuned for rank=8. For rank=16, may want to lower to ~10.0.
2. **QLoRA native**: When the base model is quantized (4-bit), LoRA automatically wraps `QuantizedLinear` layers — no special config needed.
3. **Auto layer detection**: If no `keys` specified, finds all Linear-like layers in each transformer block.
4. **Chat template integration**: The `tools` field in JSONL data is passed to `tokenizer.apply_chat_template()`, so Qwen3's native tool-calling format is automatically applied.

## Training Plan for Eden v1

```yaml
model: mlx-community/Qwen3-1.7B-4bit
train: true
data: data/
num_layers: 16
batch_size: 4
iters: 1000
learning_rate: 1e-5
steps_per_eval: 200
save_every: 200
max_seq_length: 2048
mask_prompt: true
adapter_path: adapters/qwen3-1.7b-tools-v1
lora_parameters:
  rank: 16
  scale: 20.0
  dropout: 0.0
```
