# Eden Models — Local Tool-Calling LLMs

Training pipeline for the Eden model family: purpose-built language models optimized for agentic tool calling at low-bit precision.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Goal

Train the first open-source language model specifically optimized for local agentic tool calling:
- **95%+ tool-calling accuracy** on Eden's 33 tools
- **200+ tok/s** on Apple Silicon (1-bit variant)
- **120MB model size** at 1.58-bit precision
- **Apache 2.0** — fully open-source

## Approach: Three Parallel Experiments

We run three approaches at small scale (5K examples, <0.5% UMRCP quota), compare results, then scale the winner.

| Experiment | Base Model | Hardware | Time | What We Learn |
|---|---|---|---|---|
| **Exp 1:** LoRA fine-tune Qwen3-4B | Qwen3-4B (97.5% tool-call baseline) | 1x A100 80GB | ~2 hrs | Upper bound accuracy |
| **Exp 2:** 1-bit LoRA fine-tune Bonsai | Bonsai-1.7B (1-bit, Apache 2.0) | 1x RTX PRO 6000 Blackwell | ~4 hrs | Can 1-bit match full precision? |
| **Exp 3:** Train from scratch (1-bit) | None — BitNet b1.58 architecture | 1x RTX PRO 6000 Blackwell | ~48 hrs | Is purpose-built viable? |

### Decision Tree After Experiments

```
IF Qwen3 LoRA >> Bonsai >> Scratch:
  → Scale Qwen3 LoRA to 50K dataset (production model)
  → Publish Bonsai results for comparison

IF Bonsai LoRA ≈ Qwen3 LoRA:
  → 1-bit matches full precision — major finding
  → Scale Bonsai fine-tune (120MB production model)

IF Scratch 500M > 80% accuracy:
  → Purpose-built is viable — scale to 1B, 4B, 8B
  → That's a major paper

IF Scratch 500M < 60%:
  → Focus on fine-tuning paths
  → Try scratch again at larger scale
```

## Training Infrastructure

- **Cluster:** University of Michigan Great Lakes HPC ([UMRCP](https://arc.umich.edu/umrcp/))
- **GPUs:** A100 80GB (Exp 1) + RTX PRO 6000 Blackwell 96GB (Exp 2, 3)
- **Storage:** 10TB Turbo (active) + 100TB Data Den (archive)
- **Total compute budget:** ~300 GPU hours for experiments (<0.5% UMRCP)

### Why Two GPU Types

| GPU | Best For | Why |
|---|---|---|
| A100 80GB | LoRA fine-tuning (Exp 1) | HBM2e at 2 TB/s — fastest for gradient updates |
| RTX PRO 6000 Blackwell | 1-bit training (Exp 2, 3) | Native FP4 Tensor Cores — 2-4x faster for sub-4-bit math |

## Training Data

### Layer 1: Open-Source Tool-Calling Datasets (~590K examples)

| Dataset | Size | Purpose | License |
|---|---|---|---|
| [ToolMind](https://arxiv.org/abs/2511.15718) | 360K | Multi-turn with reasoning traces | Apache 2.0 |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60K | High-quality single-turn | CC-BY-4.0 |
| [Glaive function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113K | Diverse conversations | Apache 2.0 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 26K | Verified multi-domain | Apache 2.0 |
| [NVIDIA When2Call](https://github.com/NVIDIA/When2Call) | 10K | When NOT to call tools | Apache 2.0 |
| [Gorilla/BFCL](https://gorilla.cs.berkeley.edu/) | 14K | Berkeley function calling | Apache 2.0 |

### Layer 2: Eden-Specific Synthetic Data (~50-100K examples)

Generated using Claude API, covering all 33 Eden tools with Claude Code behavioral patterns:

| Category | % | Description |
|---|---|---|
| Single tool call | 30% | One tool, correct params |
| Multi-tool sequential | 20% | Tool A → use result → Tool B |
| Multi-tool parallel | 10% | Multiple independent tools |
| Error recovery | 10% | Tool fails → model tries alternative |
| No tool needed | 10% | Model answers directly |
| Complex chains | 15% | 3+ tools with reasoning |
| Refusal | 5% | Dangerous commands blocked |

### Layer 3: Eden Session Replay (growing)

Real conversations from Eden usage — the training flywheel.

### Claude Code Behavioral Patterns

Training data is aligned with Claude Code's observable behavioral patterns:
- **Explore before modify** — always read/grep before editing files
- **Error recovery** — tool fails → try alternative approach
- **Permission model** — auto-execute safe tools, ask for write ops
- **Response format** — summarize results, don't dump raw output
- **Multi-step chains** — find → read → edit → verify

## Repository Structure

```
eden-models/
├── configs/                    # Training configurations
│   ├── qwen3-4b-lora.yaml     # Exp 1: Qwen3 QLoRA on A100
│   ├── bonsai-1bit-lora.yaml  # Exp 2: Bonsai 1-bit LoRA on RTX PRO 6000
│   └── eden-500m-scratch.yaml # Exp 3: From-scratch BitNet on RTX PRO 6000
├── data/
│   ├── generation/             # Synthetic data generation scripts
│   │   ├── generate_eden_tools.py      # Claude API → 50K examples
│   │   ├── generate_hard_negatives.py  # Refusals, security, no-tool
│   │   └── claude_code_patterns.py     # Behavioral pattern templates
│   ├── processing/             # Data cleaning and formatting
│   │   ├── format_for_sft.py
│   │   ├── merge_datasets.py
│   │   └── quality_filter.py
│   └── schemas/
│       └── eden_tools.json     # All 33 Eden tools with examples
├── training/
│   ├── sft.py                  # Supervised fine-tuning (LoRA + full)
│   ├── dpo.py                  # Direct preference optimization
│   ├── bitnet.py               # 1-bit training (from scratch)
│   └── utils.py
├── eval/
│   ├── bfcl_eval.py            # Berkeley Function Calling benchmark
│   ├── eden_eval.py            # Custom 400-case Eden eval
│   ├── speed_bench.py          # tok/s across hardware
│   └── test_cases/
├── convert/
│   ├── to_mlx.py               # → MLX format (Apple Silicon)
│   ├── to_gguf.py              # → GGUF format (llama.cpp)
│   └── to_1bit.py              # → 1-bit quantization
├── slurm/                      # Great Lakes job scripts
│   ├── sft_qwen3.slurm        # Exp 1
│   ├── sft_bonsai.slurm       # Exp 2
│   ├── train_scratch.slurm    # Exp 3
│   └── eval.slurm
├── docs/
│   └── RESEARCH-PLAN.md
└── LICENSE                     # Apache 2.0
```

## Timeline

```
Week 1: Data Generation
├── Generate 5K test examples (verify quality)
├── Generate remaining 45K examples
├── Download + process open-source datasets
├── Set up Great Lakes environment
└── Parallel: Format data for all 3 experiments

Week 2: Run All 3 Experiments (small scale)
├── Monday:    Exp 1 — Qwen3 LoRA on A100 (2 hrs, 5K examples)
├── Tuesday:   Exp 2 — Bonsai LoRA on RTX PRO 6000 (4 hrs)
├── Wed-Thu:   Exp 3 — Scratch 500M on RTX PRO 6000 (48 hrs)
├── Friday:    Evaluate all three on Eden-Eval + BFCL
└── Friday PM: DECIDE which path to scale

Week 3-4: Scale the Winner
├── Full 50K dataset training
├── Multiple model sizes if from-scratch won
├── Comprehensive benchmarks
├── Convert to MLX, test on Apple Silicon
└── Compare: Eden-Eval, BFCL, tok/s, model size

Week 5: Write + Release
├── arXiv paper (all 3 experiments + scaled results)
├── HuggingFace model release
├── Blog post
└── Integration into Eden framework
```

## Evaluation

| Benchmark | What It Measures | Target |
|---|---|---|
| [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) | General tool-calling accuracy | >90% |
| Eden-Eval (custom, 400 cases) | Eden's 33 specific tools | >95% |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | Multi-tool planning | >85% |
| Speed (tok/s) | Inference on M1 Max 64GB | >200 (1-bit) |
| MT-Bench | General conversation quality | >6.5 |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/alex-rentel/eden-models.git
cd eden-models

# 2. Generate training data (needs ANTHROPIC_API_KEY)
pip install aiohttp
export ANTHROPIC_API_KEY=sk-...
python data/generation/generate_eden_tools.py \
    --num_examples 1000 \
    --output data/test_1k.jsonl \
    --parallel 3

# 3. Inspect quality
head -5 data/test_1k.jsonl | python -m json.tool

# 4. Train (on Great Lakes)
sbatch slurm/sft_qwen3.slurm
```

## Related Projects

- [Eden](https://github.com/alex-rentel/eden) — The local AI agent framework these models are built for
- [PrismML/Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) — 1-bit model architecture reference
- [Microsoft BitNet](https://github.com/microsoft/BitNet) — 1.58-bit training methodology
- [Salesforce xLAM](https://github.com/SalesforceAIResearch/xLAM) — Action model research
- [Berkeley BFCL](https://gorilla.cs.berkeley.edu/) — Tool-calling evaluation standard

## License

Apache 2.0 — all code, data generation scripts, and model weights.
