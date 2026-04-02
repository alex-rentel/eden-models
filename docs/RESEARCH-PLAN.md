# Eden-1B: Training a 1-Bit Tool-Calling Language Model
## Research Proposal — University of Michigan ARC / UMRCP

**Principal Investigator:** Alexis Castellanos
**Affiliation:** University of Michigan
**Computing Resources:** UMRCP + ARC-supported RTX PRO 6000 Blackwell GPUs
**License:** Apache 2.0 (all outputs open-sourced)

---

## 1. Executive Summary

We propose training **Eden-1B**, the first open-source language model specifically optimized for agentic tool calling at 1.58-bit precision. At ~120MB, Eden-1B will run at 200+ tokens/second on consumer Apple Silicon hardware while achieving 95%+ tool-calling accuracy — enabling local AI agents on devices with as little as 4GB RAM.

The model family (Eden-1B, Eden-4B, Eden-8B) will be trained on NVIDIA RTX PRO 6000 Blackwell GPUs through UMich's Advanced Research Computing infrastructure, leveraging the GPU's native FP4 Tensor Cores for efficient sub-4-bit training.

**Key deliverables:**
- Open-source model weights on HuggingFace (Apache 2.0)
- 50-100K synthetic tool-calling training dataset
- Training code and reproduction scripts
- Benchmark results on BFCL and custom agentic evaluation
- arXiv publication

---

## 2. Research Justification

### 2.1 The Problem

Current local AI agents face a fundamental tradeoff: small models (1-4B) are fast but poor at tool calling; large models (7B+) are capable but slow on consumer hardware. No existing model is specifically trained for agentic tool calling at low-bit precision.

### 2.2 Why RTX PRO 6000 Blackwell

| Capability | Why It Matters |
|---|---|
| 96GB GDDR7 ECC | Train up to 8B models without multi-node complexity |
| 5th-gen Tensor Cores with FP4 | Native hardware support for 1-bit/1.58-bit weight simulation |
| 1,792 GB/s bandwidth | Fast gradient updates during training |
| 752 Tensor Cores | High throughput for transformer matrix operations |
| Blackwell architecture | Latest CUDA optimizations for mixed-precision training |
| ECC memory | Ensures training stability over multi-day runs |

**Critical:** The FP4 native support on Blackwell Tensor Cores is not available on our existing A100 or V100 hardware. Simulating 1.58-bit training in FP16 on A100 is 2-4x slower than native FP4 on Blackwell. This hardware is uniquely suited to our research.

### 2.3 Why This Hasn't Been Done

- **PrismML/Bonsai** trained 1-bit models for general intelligence (cost: millions on TPU v4 pods)
- **Microsoft/BitNet** demonstrated 1.58-bit architectures but released no tool-calling variants
- **Qwen/Llama** families train for general capabilities, not tool-call specialists
- No one has combined 1-bit precision + tool-calling specialization + open-source release

---

## 3. Technical Approach

### 3.1 Architecture: BitNet b1.58 Transformer

Based on Microsoft Research's "The Era of 1-bit LLMs" (Ma et al., 2024), using ternary weights {-1, 0, +1} with 1.58-bit effective precision.

```
Architecture: Transformer decoder-only
Precision: 1.58-bit weights (ternary), FP4 activations on Blackwell
Vocabulary: 32,000 tokens (shared with Qwen tokenizer for compatibility)
Context: 8,192 tokens (sufficient for tool-calling conversations)
Attention: Grouped-Query Attention (GQA), 8 KV heads
Normalization: RMSNorm (pre-norm)
Activation: SiLU/Swish
Position encoding: RoPE
```

**Model family:**

| Model | Params | Layers | Hidden | Heads | Size (1-bit) | Target Speed (M1) |
|---|---|---|---|---|---|---|
| Eden-1B | 1.0B | 24 | 2048 | 16 | ~120MB | 200+ tok/s |
| Eden-4B | 4.0B | 32 | 3584 | 28 | ~500MB | 80+ tok/s |
| Eden-8B | 8.0B | 40 | 4096 | 32 | ~1.0GB | 40+ tok/s |

### 3.2 Training Data Pipeline

**Phase A: Synthetic Tool-Calling Dataset (50-100K examples)**

Generated using Claude API with structured prompting:

```json
{
  "id": "eden-tool-0001",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI assistant with access to these tools: [bash, file_read, file_write, file_edit, glob, grep, python_exec, web_fetch, web_search, ...33 total]. Call tools using <tool_call> tags."
    },
    {
      "role": "user",
      "content": "Find all Python files larger than 100KB in the project"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"bash\", \"input\": {\"command\": \"find . -name '*.py' -size +100k -exec ls -lh {} ;\"}}</tool_call>"
    },
    {
      "role": "tool",
      "name": "bash",
      "content": "-rw-r--r--  1 user  staff  142K  eden/tree.py\n-rw-r--r--  1 user  staff  118K  eden/security.py"
    },
    {
      "role": "assistant",
      "content": "Found 2 Python files over 100KB:\n- eden/tree.py (142K)\n- eden/security.py (118K)"
    }
  ],
  "metadata": {
    "tools_used": ["bash"],
    "turns": 2,
    "complexity": "single_tool",
    "category": "file_search"
  }
}
```

**Dataset composition (100K examples):**

| Category | Examples | Description |
|---|---|---|
| Single tool call | 30,000 | One tool, correct params |
| Multi-tool sequential | 20,000 | Tool A → use result → Tool B |
| Multi-tool parallel | 10,000 | Multiple tools in one response |
| Error recovery | 10,000 | Tool fails → model tries alternative |
| No tool needed | 10,000 | Model answers directly (negative examples) |
| Complex reasoning | 10,000 | Multi-step plans with 3+ tools |
| Edge cases | 5,000 | Ambiguous queries, partial info |
| Refusal | 5,000 | Dangerous commands → model refuses |

**Coverage:** All 33 Eden tools, 50+ coding languages, 20+ project structures.

**Phase B: General Instruction Data (500K examples)**

To prevent catastrophic forgetting, mix tool-calling data with general instruction following:
- SlimOrca (filtered, Apache 2.0)
- OpenHermes 2.5 (filtered)
- Code Alpaca (coding instructions)

**Final mix:** 60% tool-calling, 30% general instruction, 10% code generation.

### 3.3 Training Procedure

**Stage 1: Pre-training (Eden-1B only if training from scratch)**

```
Data: 10B tokens from RedPajama v2 (Apache 2.0)
Hardware: 4x RTX PRO 6000 Blackwell
Method: BitNet b1.58 — ternary quantization on weights during forward pass
Optimizer: AdamW (1-bit compatible variant)
LR: 3e-4, cosine schedule with warmup
Batch: 256 sequences × 8192 tokens = ~2M tokens/batch
Duration: ~3-5 days
Checkpoints: Every 2 hours
```

**Stage 2: Supervised Fine-Tuning (SFT)**

```
Data: 100K tool-calling + 500K general = 600K examples
Hardware: 2x RTX PRO 6000 Blackwell
Method: Full fine-tune (model is small enough)
Optimizer: AdamW, lr=2e-5
Batch: 64 sequences
Duration: ~12-24 hours
Eval: Every 1000 steps on held-out tool-calling test set
```

**Stage 3: Direct Preference Optimization (DPO)**

```
Data: 10K preference pairs (correct tool call vs wrong tool call)
Hardware: 1x RTX PRO 6000 Blackwell
Method: DPO with β=0.1
Duration: ~2-4 hours
Purpose: Sharpen tool selection, reduce hallucinated parameters
```

### 3.4 Alternative Path: LoRA Fine-Tune Existing Model

If training from scratch proves too costly, we can LoRA fine-tune an existing model:

```
Base: Qwen3-4B or Bonsai-1.7B (both Apache 2.0)
Method: QLoRA (4-bit base + LoRA adapters)
Rank: r=64, alpha=128
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Data: 100K tool-calling examples
Hardware: 1x RTX PRO 6000 Blackwell
Duration: ~8-16 hours
```

This produces a tool-calling specialist adapter that can be merged into the base model.

---

## 4. Hardware Configuration

### 4.1 Requested Configuration

```
Primary: 4x NVIDIA RTX PRO 6000 Blackwell Server Edition
├── 4 × 96GB GDDR7 ECC = 384GB total GPU memory
├── 4 × 1,792 GB/s = 7,168 GB/s aggregate bandwidth
├── 4 × 752 Tensor Cores = 3,008 Tensor Cores total
├── 4 × 600W TDP = 2,400W peak power
├── PCIe Gen 5 x16 interconnect
└── FP4 native Tensor Core support

Host system:
├── CPU: Dual AMD EPYC 9474F or Intel Xeon w9-3595X
├── RAM: 512GB DDR5 ECC
├── Storage: 4TB NVMe SSD (local scratch)
├── Network: 100GbE to Turbo storage
└── OS: Rocky Linux 8/9 (matching Great Lakes)
```

### 4.2 Integration Options

**Option A: Lighthouse cluster addition**
- Purchase hardware, integrate into Lighthouse
- Shared with other researchers after our project
- ARC manages maintenance

**Option B: Dedicated research node**
- Purchased through ARC Research Purchased Hardware program
- Dedicated to our research group for project duration
- Integrated into Great Lakes SLURM scheduler

**Option C: Leased cloud instances**
- RTX PRO 6000 available on Spheron ($1.65/hr) and Nebius
- 4 GPUs × $1.65/hr × 720 hours/month = ~$4,752/month
- Less ideal — UMRCP free allocation is better

### 4.3 Storage Requirements

| Storage | Amount | Purpose |
|---|---|---|
| Turbo (active) | 2TB | Training data, checkpoints, code |
| Scratch (local NVMe) | 4TB | Fast I/O during training |
| Data Den (archive) | 10TB | Final models, all checkpoints, datasets |

---

## 5. Training Timeline

```
Week 0: Preparation
├── Generate synthetic training dataset (50K examples) — MacBook Pro
├── Validate dataset quality with held-out eval
├── Set up training codebase on Great Lakes
├── Test with existing A100s (small LoRA run as validation)
└── ARC hardware procurement / allocation approval

Week 1-2: LoRA Baseline (on existing A100s)
├── LoRA fine-tune Qwen3-4B on tool-calling data
├── Evaluate on BFCL benchmark + custom Eden eval
├── This proves the data pipeline works
├── Uses <5% of UMRCP allocation
└── Publish baseline results

Week 3-4: Eden-1B Training (on RTX PRO 6000)
├── Day 1-3: Stage 1 pre-training (if from scratch)
├── Day 4: Stage 2 SFT on tool-calling + general data
├── Day 5: Stage 3 DPO on preference pairs
├── Day 5-7: Evaluation, ablation studies
└── Convert to MLX format, test on Apple Silicon

Week 5-6: Eden-4B Training
├── Same pipeline, larger model
├── ~5-7 days on 4x RTX PRO 6000
├── Compare against Eden-1B and Qwen3-4B
└── Determine optimal size/accuracy tradeoff

Week 7-8: Eden-8B Training (stretch goal)
├── Full 8B model, ~2-3 weeks on 4x RTX PRO 6000
├── Only if results from 1B/4B are promising
└── Could use extended ARC allocation if needed

Week 9-10: Evaluation + Paper
├── Comprehensive benchmarks (BFCL, ToolBench, custom)
├── Comparison with Qwen3.5-4B, Nemotron Nano, GPT-OSS
├── Ablation studies (data size, model size, precision)
├── Write arXiv paper
└── Release everything on HuggingFace + GitHub
```

---

## 6. Evaluation Plan

### 6.1 Benchmarks

| Benchmark | What It Measures | Target |
|---|---|---|
| BFCL (Berkeley Function Calling) | Tool calling accuracy across formats | >90% |
| ToolBench | Multi-tool planning and execution | >85% |
| Eden-Eval (custom) | Accuracy on Eden's 33 specific tools | >95% |
| MT-Bench | General conversation quality | >6.5 |
| HumanEval | Code generation | >50% |
| IFEval | Instruction following | >80% |

### 6.2 Custom Eden-Eval Suite

400 test cases across Eden's tool categories:

```
bash (60 cases): file ops, git, system info, piping
file_read/write/edit (60 cases): create, modify, search-replace
glob + grep (40 cases): file discovery, content search
python_exec (40 cases): calculations, data processing
web tools (30 cases): fetch pages, search queries
multi-tool chains (80 cases): 2-5 tool sequences
error recovery (40 cases): handle failures gracefully
refusal (30 cases): dangerous commands blocked
no-tool (20 cases): answer directly without tools
```

### 6.3 Speed Benchmarks

| Device | Eden-1B Target | Eden-4B Target | Eden-8B Target |
|---|---|---|---|
| M1 (8GB) | 200+ tok/s | 60+ tok/s | N/A (too large) |
| M1 Max (64GB) | 250+ tok/s | 100+ tok/s | 50+ tok/s |
| M4 (16GB) | 300+ tok/s | 80+ tok/s | 35+ tok/s |
| Mac Mini M4 (16GB) | 300+ tok/s | 80+ tok/s | 35+ tok/s |

---

## 7. Resource Budget

### 7.1 UMRCP Allocation Usage

| Phase | Resource | GPU Hours | % of UMRCP |
|---|---|---|---|
| LoRA baseline | 4x A100 | ~200 | 0.25% |
| Data preprocessing | 1x A40 | ~100 | 0.12% |
| Eden-1B training | 4x RTX PRO 6000 | ~2,000 | 2.5% |
| Eden-4B training | 4x RTX PRO 6000 | ~5,000 | 6.25% |
| Eden-8B training | 4x RTX PRO 6000 | ~15,000 | 18.75% |
| Evaluation runs | 1x RTX PRO 6000 | ~500 | 0.6% |
| **Total** | | **~22,800** | **28.5%** |

Leaves 71.5% of UMRCP allocation for other research needs.

### 7.2 Additional Costs

| Item | Cost | Funding |
|---|---|---|
| Claude API (data gen) | ~$100-200 | Out of pocket |
| HuggingFace storage | Free (open model) | N/A |
| RTX PRO 6000 hardware | ~$34,000 (4x $8,500) | ARC purchase program |
| Electricity | Covered by UMich | N/A |

---

## 8. Open-Source Deliverables

All outputs released under Apache 2.0:

```
huggingface.co/eden-ai/
├── Eden-1B/               # 1B param, 1.58-bit, ~120MB
│   ├── model.safetensors
│   ├── tokenizer/
│   ├── config.json
│   └── README.md          # Model card with benchmarks
├── Eden-4B/               # 4B param, 1.58-bit, ~500MB
├── Eden-8B/               # 8B param, 1.58-bit, ~1.0GB
├── Eden-1B-MLX/           # MLX-converted for Apple Silicon
├── Eden-4B-MLX/
├── Eden-8B-MLX/
└── Eden-ToolCall-100K/    # Training dataset

github.com/alex-rentel/eden-training/
├── configs/               # Model architecture configs
├── data/                  # Data generation scripts
├── train/                 # Training code (PyTorch + BitNet)
├── eval/                  # Evaluation suite
├── convert/               # Weight conversion (PT → MLX → GGUF)
├── paper/                 # LaTeX source for arXiv paper
└── reproduce.sh           # One-command reproduction
```

---

## 9. Impact Statement

Eden-1B would be the first model at the intersection of:
- **1-bit precision** (120MB, runs on any device)
- **Tool-calling specialization** (95%+ accuracy on 33 agentic tools)
- **Open-source** (Apache 2.0, fully reproducible)
- **Apple Silicon optimized** (MLX native format)

This enables local AI agents on devices that cannot run existing 4B+ models, including older iPhones, iPads, Raspberry Pi, and IoT devices. The research contributes to UMich's AI/ML research portfolio and demonstrates practical use of Blackwell GPU architecture for novel training workloads.

---

## 10. Next Steps

1. **Immediate:** Meet with ARC to discuss RTX PRO 6000 availability/procurement
2. **Week 1:** Begin synthetic data generation pipeline
3. **Week 2:** LoRA baseline on existing A100 allocation
4. **Week 3+:** Full training on RTX PRO 6000 Blackwell
5. **Week 9-10:** Paper submission to arXiv, model release on HuggingFace

---

*Prepared by: Alexis Castellanos*
*University of Michigan — Data Science*
*Contact: [your_uniqname]@umich.edu*
*Date: April 2026*
