# Eden Models — Local Tool-Calling LLMs

Training pipeline for the Eden model family: purpose-built language models optimized for agentic tool calling at low-bit precision.

## Goal

Train the first open-source language model specifically optimized for local agentic tool calling, targeting:
- **95%+ tool-calling accuracy** on Eden's 33 tools
- **200+ tok/s** on Apple Silicon (Eden-1B)
- **120MB model size** at 1.58-bit precision (Eden-1B)
- **Apache 2.0** — fully open-source

## Approach

**Phase 1:** Fine-tune Qwen3-4B on tool-calling data (proven, fast, low-risk)
**Phase 2:** Train Eden-1B from scratch at 1.58-bit (novel, publishable)

## Training Infrastructure

- **Cluster:** University of Michigan Great Lakes HPC (UMRCP)
- **GPUs:** A100 80GB (Phase 1), targeting 4x for Phase 2
- **Storage:** 10TB Turbo
- **Framework:** PyTorch + Hugging Face Transformers + TRL

## Repository Structure

```
eden-models/
├── configs/              # Model architecture and training configs
│   ├── qwen3-4b-lora.yaml
│   └── eden-1b.yaml
├── data/
│   ├── generation/       # Scripts to generate synthetic training data
│   │   ├── generate_eden_tools.py    # Eden-specific tool conversations
│   │   ├── generate_hard_negatives.py # Refusals, security, no-tool
│   │   └── claude_code_patterns.py   # Match Claude Code behavior
│   ├── processing/       # Data cleaning, filtering, formatting
│   │   ├── format_for_sft.py
│   │   ├── merge_datasets.py
│   │   └── quality_filter.py
│   └── schemas/          # Eden tool definitions for data generation
│       └── eden_tools.json
├── training/
│   ├── sft.py            # Supervised fine-tuning
│   ├── dpo.py            # Direct preference optimization
│   ├── bitnet.py         # 1-bit training (Phase 2)
│   └── utils.py          # Training utilities
├── eval/
│   ├── bfcl_eval.py      # Berkeley Function Calling benchmark
│   ├── eden_eval.py      # Custom Eden tool-calling eval (400 cases)
│   ├── speed_bench.py    # tok/s on different hardware
│   └── test_cases/       # Eval test cases
├── convert/
│   ├── to_mlx.py         # Convert weights to MLX format
│   ├── to_gguf.py        # Convert to GGUF for llama.cpp
│   └── to_1bit.py        # Quantize to 1-bit
├── slurm/                # Great Lakes job scripts
│   ├── sft_qwen3.slurm
│   ├── train_eden1b.slurm
│   └── eval.slurm
├── docs/
│   └── RESEARCH-PLAN.md  # Full research proposal
├── LICENSE               # Apache 2.0
└── pyproject.toml
```

## Datasets Used

| Dataset | Size | Purpose | License |
|---|---|---|---|
| [ToolMind](https://arxiv.org/abs/2511.15718) | 360K | Multi-turn tool calling with reasoning | Apache 2.0 |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60K | High-quality single-turn tool calling | CC-BY-4.0 |
| [Glaive function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113K | Diverse tool-calling conversations | Apache 2.0 |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 26K | Verified multi-domain tool calling | Apache 2.0 |
| [NVIDIA When2Call](https://github.com/NVIDIA/When2Call) | 10K | When NOT to call tools | Apache 2.0 |
| [Gorilla/BFCL](https://gorilla.cs.berkeley.edu/) | 14K | Berkeley function calling data | Apache 2.0 |
| Eden Synthetic (ours) | 50-100K | Eden's 33 tools, Claude Code patterns | Apache 2.0 |
| Eden Session Replay (ours) | Growing | Real user interactions with Eden | Apache 2.0 |

## Training Recipe

### Phase 1: Qwen3-4B LoRA (fast baseline)
```bash
# On Great Lakes — 1x A100 80GB, ~8-16 hours
sbatch slurm/sft_qwen3.slurm
```

### Phase 2: Eden-1B from scratch (research contribution)
```bash
# On Great Lakes — 4x A100 80GB, ~5-7 days  
sbatch slurm/train_eden1b.slurm
```

## Related Projects

- [Eden](https://github.com/alex-rentel/eden) — The local AI agent framework these models are built for
- [PrismML/Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) — 1-bit model architecture reference
- [Microsoft BitNet](https://github.com/microsoft/BitNet) — 1.58-bit training methodology
- [Salesforce xLAM](https://github.com/SalesforceAIResearch/xLAM) — Action model research

## License

Apache 2.0 — all code, data generation scripts, and model weights.
