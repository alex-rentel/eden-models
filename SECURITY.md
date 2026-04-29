# Security policy

## Reporting a vulnerability

Email **alex@renaissanceintelligence.ai** with the details. Avoid filing a public GitHub issue for anything you believe could be exploited — open a private channel first.

A useful report includes:

- Commit SHA you reproduced against.
- A minimal repro: training script invocation, dataset path, the failure mode.
- Cluster vs. local, GPU type, Python + framework versions.

You should expect a first reply within a few days.

## Supported versions

Only `main` gets security fixes. Branches and historical commits are unsupported.

## Scope

In scope: anything that lets crafted training data / SFT JSONL crash the pipeline, leak data across runs, or escape the configured output directory. Same for the synthetic-data generation scripts (`scripts/generate_training_data.py`) — model-injection-via-OpenRouter-prompt is a real concern.

Out of scope:

- Training instabilities, divergence, NaN losses on a particular config — these are bugs or hyperparameter issues, not security issues. Open a public GitHub issue.
- SLURM / Great Lakes scheduler issues — report to UMich ARC support (`arc-support@umich.edu`).
- Issues in MLX, HuggingFace transformers, OpenRouter, or model weights themselves — report upstream.
- Anything that requires the attacker to already have write access to the output directory or `$SLURM_TMPDIR`.
