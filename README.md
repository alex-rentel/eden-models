# Eden Models — Local Tool-Calling LLMs

Training pipeline for purpose-built language models optimized for local agentic tool calling at low-bit precision. Fine-tuned models deploy locally on Apple Silicon via Ollama (MLX backend) or mlx-lm.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Goal

Train open-source language models specifically optimized for local agentic tool calling:
- **95%+ tool-calling accuracy** on Eden's 33 tools
- **200+ tok/s** on Apple Silicon (1-bit variant)
- **120MB model size** at 1.58-bit precision
- **Apache 2.0** — fully open-source

## April 2026 Landscape

The open-source model landscape shifted massively, opening new experiment paths:

| Model | Why It Matters |
|---|---|
| **Gemma 4 E4B** (Google, April 2) | Apache 2.0, native function calling, multimodal, 128K context — better base than Qwen3-4B |
| **GPT-OSS 20B** (OpenAI) | Apache 2.0, MoE (21B total/3.6B active), specifically optimized for tool use |
| **Qwen 3.6 Plus** | Free on OpenRouter — zero-cost synthetic training data generation |
| **TurboQuant** | 4x less KV cache memory — enables bigger models on 16GB Mac Mini |

## Approach: Four Parallel Experiments

Run four approaches at small scale, compare results, then scale the winner.

| Experiment | Base Model | Hardware | Time | What We Learn |
|---|---|---|---|---|
| **Exp 1:** Gemma 4 E4B QLoRA | google/gemma-4-E4B-it | 1x A100 80GB | ~8-16 hrs | Native tool-calling base — upper bound accuracy |
| **Exp 2:** Bonsai 1-bit LoRA | prism-ml/Bonsai-1.7B-1bit | 1x RTX PRO 6000 | ~4 hrs | Can 1-bit match full precision? |
| **Exp 3:** From-scratch 500M | BitNet b1.58 architecture | 1x RTX PRO 6000 | ~48 hrs | Is purpose-built viable? |
| **Exp 4:** GPT-OSS 20B QLoRA | openai/gpt-oss-20b | 1x A100 80GB | ~16-24 hrs | Best open-source tool-calling base? |

### Decision Tree After Experiments

```
IF Gemma4 LoRA >> all others:
  → Scale Gemma4 LoRA to 50K dataset (production model)
  → Native function calling = less fine-tuning needed

IF GPT-OSS LoRA ≈ Gemma4 LoRA:
  → MoE architecture = faster inference at same quality
  → Scale GPT-OSS (3.6B active params fits on 16GB)

IF Bonsai LoRA ≈ Gemma4/GPT-OSS:
  → 1-bit matches full precision — major finding
  → Scale Bonsai fine-tune (120MB production model)

IF Scratch 500M > 80% accuracy:
  → Purpose-built is viable — scale to 1B, 4B, 8B
  → That's a major paper
```

## Full Pipeline

```
training-flywheel (captures real tool-calling sessions)
        ↓
eden-models (trains on Great Lakes HPC)
        ↓
mlx-nanochat (validates locally on Apple Silicon)
        ↓
deploy to Ollama (MLX backend)
        ↓
mlx-turboquant (validates compression, 4x KV cache savings)
```

## Training Infrastructure

- **Cluster:** University of Michigan Great Lakes HPC ([UMRCP](https://arc.umich.edu/umrcp/))
- **GPUs:** A100 80GB (Exp 1, 4) + RTX PRO 6000 Blackwell 96GB (Exp 2, 3)
- **Storage:** 10TB Turbo (active) + 100TB Data Den (archive)

### Why Two GPU Types

| GPU | Best For | Why |
|---|---|---|
| A100 80GB | QLoRA fine-tuning (Exp 1, 4) | HBM2e at 2 TB/s — fastest for gradient updates on larger models |
| RTX PRO 6000 Blackwell | 1-bit training (Exp 2, 3) | Native FP4 Tensor Cores — 2-4x faster for sub-4-bit math |

## Training Data

### Primary: training-flywheel captures
Real user↔Eden sessions in ChatML format with tool calls.

### Supplemental: Synthetic generation
When training-flywheel hasn't captured enough data, generate synthetic conversations:

```bash
export OPENROUTER_API_KEY=...
python scripts/generate_training_data.py \
    --count 1000 \
    --output data/synthetic_sft.jsonl \
    --tools configs/tool_schemas.json \
    --difficulty mixed
```

Uses Qwen 3.6 Plus on OpenRouter (free tier) to generate validated tool-calling conversations.

### Layer 1: Open-Source Tool-Calling Datasets (~590K examples)

| Dataset | Size | License |
|---|---|---|
| [ToolMind](https://arxiv.org/abs/2511.15718) | 360K | Apache 2.0 |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60K | CC-BY-4.0 |
| [Glaive function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113K | Apache 2.0 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 26K | Apache 2.0 |
| [NVIDIA When2Call](https://github.com/NVIDIA/When2Call) | 10K | Apache 2.0 |
| [Gorilla/BFCL](https://gorilla.cs.berkeley.edu/) | 14K | Apache 2.0 |

### Layer 2: Eden-Specific Synthetic Data (~50-100K examples)

Generated covering all 33 Eden tools with Claude Code behavioral patterns.

### Layer 3: Eden Session Replay (growing)

Real conversations from Eden usage — the training flywheel.

## Repository Structure

```
eden-models/
├── configs/                         # Training configurations
│   ├── gemma4-e4b-lora.yaml         # Exp 1: Gemma 4 E4B QLoRA on A100
│   ├── bonsai-1bit-lora.yaml        # Exp 2: Bonsai 1-bit LoRA on RTX PRO 6000
│   ├── eden-500m-scratch.yaml       # Exp 3: From-scratch BitNet on RTX PRO 6000
│   ├── gpt-oss-20b-lora.yaml        # Exp 4: GPT-OSS 20B QLoRA on A100
│   └── tool_schemas.json            # Tool schemas for synthetic data gen
├── scripts/
│   ├── generate_training_data.py    # Synthetic data via OpenRouter (Qwen 3.6+)
│   └── compare_experiments.py       # Side-by-side experiment comparison
├── data/
│   ├── generation/                  # Claude API data generation
│   │   ├── generate_eden_tools.py
│   │   └── claude_code_patterns.py
│   └── schemas/
│       └── eden_tools.json          # All 33 Eden tools with examples
├── training/
│   ├── sft.py                       # Supervised fine-tuning (LoRA + full)
│   └── bitnet.py                    # 1-bit training (from scratch)
├── eval/
│   └── eden_eval.py                 # Custom 400-case Eden eval
├── convert/
│   └── to_mlx.py                    # → MLX format (Apple Silicon)
├── slurm/                           # Great Lakes job scripts
│   ├── train_gemma4.slurm           # Exp 1
│   ├── sft_bonsai.slurm            # Exp 2
│   ├── train_scratch.slurm         # Exp 3
│   ├── train_gptoss.slurm          # Exp 4
│   └── eval.slurm
├── docs/
│   └── RESEARCH-PLAN.md
└── LICENSE                          # Apache 2.0
```

## Running on Great Lakes HPC

```bash
# 1. SSH into Great Lakes
ssh YOUR_UNIQNAME@greatlakes.arc-ts.umich.edu

# 2. Clone and set up
git clone https://github.com/alex-rentel/eden-models.git
cd eden-models
module load python/3.11
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Run experiments
sbatch slurm/train_gemma4.slurm     # Exp 1: Gemma 4 E4B
sbatch slurm/sft_bonsai.slurm       # Exp 2: Bonsai 1-bit
sbatch slurm/train_scratch.slurm    # Exp 3: From-scratch 500M
sbatch slurm/train_gptoss.slurm     # Exp 4: GPT-OSS 20B

# 4. Monitor
squeue -u $USER
tail -f logs/eden-gemma4-*.log

# 5. Evaluate all
sbatch slurm/eval.slurm

# 6. Compare results
python scripts/compare_experiments.py \
    --experiments results/gemma4.json results/bonsai.json results/scratch.json results/gptoss.json \
    --output results/comparison.md
```

## Evaluation

| Benchmark | What It Measures | Target |
|---|---|---|
| [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) | General tool-calling accuracy | >90% |
| Eden-Eval (custom, 400 cases) | Eden's 33 specific tools | >95% |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | Multi-tool planning | >85% |
| Speed (tok/s) | Inference on M1 Max 64GB | >200 (1-bit) |
| MT-Bench | General conversation quality | >6.5 |

## Related Projects

- [training-flywheel](https://github.com/alex-rentel/training-flywheel) — Captures real tool-calling sessions for training data
- [mlx-nanochat](https://github.com/alex-rentel/mlx-nanochat) — Local validation on Apple Silicon
- [mlx-turboquant](https://github.com/alex-rentel/mlx-turboquant) — Compression validation, 4x KV cache savings
- [PrismML/Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) — 1-bit model architecture reference
- [Microsoft BitNet](https://github.com/microsoft/BitNet) — 1.58-bit training methodology
- [Salesforce xLAM](https://github.com/SalesforceAIResearch/xLAM) — Action model research
- [Berkeley BFCL](https://gorilla.cs.berkeley.edu/) — Tool-calling evaluation standard

## License

Apache 2.0 — all code, data generation scripts, and model weights.
