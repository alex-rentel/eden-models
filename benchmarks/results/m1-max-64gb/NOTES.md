# Benchmark Run Notes — 2026-04-02

## Hardware
- Apple M1 Max, 64GB RAM, macOS 26.3.1

## Models Run

| # | Model | Status | Score | Avg t/s | Notes |
|---|-------|--------|-------|---------|-------|
| 1 | Bonsai 8B | OK (prior run) | 60/72 (83.3%) | 42 t/s | Results from earlier run, copied to bonsai-8b/ |
| 2 | Bonsai 1.7B | OK | 54/72 (75.0%) | 96 t/s | Auto-downloaded from HF. Surprisingly strong for 240 MB |
| 3 | Bonsai 4B | OK | 50/72 (69.4%) | 53 t/s | Auto-downloaded from HF. Tool Calling scored 0/8 — anomalous |
| 4 | Qwen 2.5 1.5B | OK | 40/72 (55.6%) | 111 t/s | Used PrismML MLX fork (worked fine, no separate venv needed) |
| 5 | Qwen 2.5 3B | OK | 47/72 (65.3%) | 81 t/s | Same venv as above |
| 6 | Qwen 2.5 7B | OK | 65/72 (90.3%) | 45 t/s | Top scorer overall |
| 7 | Llama 3.2 1B | OK | 43/72 (59.7%) | 203 t/s | Fastest model at 203 t/s avg |
| 8 | Gemma 2 2B | OK | 41/72 (56.9%) | 69 t/s | RAG scored 0/7 (all errors) |
| 9 | Llama 3.2 3B | OK | 52/72 (72.2%) | 87 t/s | Strong code and writing |
| 10 | Phi-3.5 Mini | OK | 41/72 (56.9%) | 82 t/s | Code 0/7, Writing 1/7 — weak spots |

## Issues Encountered

- **Bonsai 4B Tool Calling 0/8**: The 4B model doesn't emit the `<tool_call>` opening tag token. See BONSAI_4B_TOOLS_INVESTIGATION.md for full root cause analysis.
- **Gemma 2B RAG 0/7**: All RAG tests returned errors (0 tok/s reported). Likely a context length or formatting issue with the Gemma 2 chat template for long-context RAG prompts.
- **Phi-3.5 Code 0/7**: All code generation tests failed. The model generated verbose explanations but failed to produce executable code blocks that passed validation.
- **All models worked with PrismML fork**: No separate venv was needed — all 10 models loaded and served correctly through the PrismML MLX fork at ~/Bonsai-demo/.venv.
- **No models skipped**: All 10 models completed the full 72-test suite.

## Timing

- Bonsai 8B: results from prior run (not timed)
- Bonsai 1.7B: ~2.5 min
- Bonsai 4B: ~5.3 min
- Qwen 1.5B: ~2.4 min
- Qwen 3B: ~2.8 min
- Qwen 7B: ~4.5 min
- Llama 1B: ~1.5 min
- Gemma 2B: ~2.5 min
- Llama 3B: ~2 min
- Phi-3.5: ~5 min
- **Total wall clock:** ~35 min benchmark time + ~15 min server startup/download = ~50 min

## Key Observations

1. **Bonsai 1-bit models are remarkably size-efficient**: Bonsai 8B at 1.28 GB scores 83.3% vs Qwen 7B at 4.4 GB scoring 90.3% — only 7 points behind at 3.4x smaller
2. **Bonsai 1.7B (0.24 GB) outperforms everything under 2 GB**: 75% beats Llama 3B (72.2%), Qwen 3B (65.3%), Phi-3.5 (56.9%), and Gemma 2B (56.9%)
3. **Speed scales inversely with size** as expected: Llama 1B tops 203 t/s, larger models in the 40-90 range
4. **1-bit quantization preserves quality** better than expected: Bonsai 8B 7/7 on reasoning, Bonsai 1.7B also 7/7
5. **Tool calling remains hard for most models**: Only Bonsai 8B and Qwen 7B score 8/8
