# Contributing to eden-models

This repo is the training pipeline for the Eden model family — purpose-built 1-bit LLMs for local agentic tool calling. Most "contributing" here is **running experiments and submitting results**, not code patches; this file mostly orients you.

## What's here

| Directory | Purpose |
|---|---|
| `training/` | SFT / pretraining scripts. Currently a placeholder — the real training entry points are in `slurm/*.slurm`. |
| `benchmarks/` | Eval harness: agentic, classification, code, math, reasoning, RAG, tools, writing, multilingual. Has its own `requirements.txt`. |
| `eval/` | Standalone eval scripts that run benchmarks on a finished checkpoint. |
| `scripts/` | Synthetic data generation, experiment comparison helpers. |
| `slurm/` | Great Lakes job scripts — the actual experiment submissions. |
| `configs/` | Tool schemas, run configs, depth/lr presets. |
| `data/` | Local data staging (gitignored except for tiny manifests). |
| `adapters/` | LoRA adapter outputs (gitignored). |
| `logs/` | Training-run logs (gitignored except for headers). |
| `RESULTS.md` | Where experiment results land for cross-run comparison. |

## Local install

```bash
git clone https://github.com/alex-rentel/eden-models.git
cd eden-models
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The default `requirements.txt` is the lightweight inference + benchmark set. For training on Great Lakes, additionally:

```bash
pip install -r benchmarks/requirements.txt
# Plus whatever training framework the slurm script needs (transformers, peft, etc.)
```

## Adding a benchmark

1. New file `benchmarks/bench_<name>.py`.
2. Use `BenchmarkSuite` from `bench_utils.py` — it handles model loading, query, scoring, and result writing.
3. Include the new bench in `benchmarks/run_all.py` so it runs in the standard sweep.
4. Append your results to `RESULTS.md` under a dated heading.

## Adding an experiment

1. New `slurm/<exp_name>.slurm` script. Match the structure of the existing four (gemma4, bonsai, scratch, gptoss).
2. Document the experiment in the README's "Experiments" section, with hardware, hyperparameters, expected runtime, and what success looks like.
3. Cross-link to `RESULTS.md`.

## Style

- Match existing surrounding code; the training scripts and benchmarks intentionally use direct, scriptable patterns over heavy abstractions.
- `.editorconfig` enforces 4-space py, 2-space yaml/json, LF endings.
- No CI runs Python tests on this repo — there are no unit tests; the training scripts are the contract. If you add `tests/`, also wire up a `.github/workflows/test.yml` (the eden-turboquant or eden-nanochat workflow is a fine template).

## Issues / questions

Open an issue at https://github.com/alex-rentel/eden-models/issues. For Great Lakes / SLURM scheduler issues, contact UMich ARC at `arc-support@umich.edu` instead — those are infra concerns, not eden-models bugs.
