#!/usr/bin/env python3
"""Compare experiment results side-by-side and produce a markdown report.

Reads WandB-style metrics JSON (or training log summaries) and generates
a markdown comparison table.

Usage:
    python scripts/compare_experiments.py \
        --experiments results/gemma4.json results/bonsai.json results/scratch.json results/gptoss.json \
        --output results/comparison.md
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    """Load experiment results JSON."""
    with open(path) as f:
        return json.load(f)


def get_metric(data: dict, key: str, default: str = "—") -> str:
    """Safely extract a metric, returning default if missing."""
    val = data.get(key)
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.4f}" if val < 1.0 else f"{val:.2f}"
    return str(val)


def generate_comparison(experiments: list[dict], output_path: str):
    """Generate markdown comparison table from experiment results."""
    lines = [
        "# Experiment Comparison Report",
        "",
        f"Generated from {len(experiments)} experiment results.",
        "",
        "## Summary Table",
        "",
        "| Metric | " + " | ".join(e.get("name", f"Exp {i+1}") for i, e in enumerate(experiments)) + " |",
        "|---|" + "|".join("---" for _ in experiments) + "|",
    ]

    metrics = [
        ("Base Model", "base_model"),
        ("Final Train Loss", "final_train_loss"),
        ("Final Eval Loss", "final_eval_loss"),
        ("Tool-Call Accuracy", "tool_call_accuracy"),
        ("BFCL Score", "bfcl_score"),
        ("Eden-Eval Score", "eden_eval_score"),
        ("Inference Speed (tok/s, M1)", "inference_tok_s_m1"),
        ("Inference Speed (tok/s, M4)", "inference_tok_s_m4"),
        ("Model Size (weights)", "model_size_mb"),
        ("Adapter Size", "adapter_size_mb"),
        ("Training Time (hrs)", "training_hours"),
        ("GPU Type", "gpu_type"),
        ("GPU Hours", "gpu_hours"),
        ("Peak VRAM (GB)", "peak_vram_gb"),
        ("Total Parameters", "total_params"),
        ("Active Parameters", "active_params"),
    ]

    for label, key in metrics:
        row = f"| {label} | " + " | ".join(
            get_metric(e, key) for e in experiments
        ) + " |"
        lines.append(row)

    lines.extend([
        "",
        "## Decision Matrix",
        "",
        "| Criteria | Weight | " + " | ".join(e.get("name", f"Exp {i+1}") for i, e in enumerate(experiments)) + " |",
        "|---|---|" + "|".join("---" for _ in experiments) + "|",
    ])

    criteria = [
        ("Tool-call accuracy", "40%"),
        ("Inference speed (Apple Silicon)", "25%"),
        ("Model size", "20%"),
        ("Training cost", "15%"),
    ]

    for name, weight in criteria:
        row = f"| {name} | {weight} | " + " | ".join("TBD" for _ in experiments) + " |"
        lines.append(row)

    lines.extend([
        "",
        "## Notes",
        "",
        "- Tool-call accuracy measured on Eden-Eval 400-case test set",
        "- Inference speed measured on Apple Silicon with MLX backend",
        "- Model size is the exported MLX/GGUF artifact size",
        "- Training cost is GPU hours × estimated $/hr on Great Lakes",
        "",
        "## Recommendation",
        "",
        "_Fill in after reviewing results._",
        "",
    ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Comparison written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments", nargs="+", required=True, help="Paths to result JSON files")
    parser.add_argument("--output", type=str, default="results/comparison.md", help="Output markdown path")
    args = parser.parse_args()

    experiments = []
    for path in args.experiments:
        try:
            data = load_results(path)
            experiments.append(data)
            print(f"Loaded: {path} ({data.get('name', 'unnamed')})")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {path}: {e}", file=sys.stderr)
            continue

    if not experiments:
        print("Error: No valid experiment results loaded.", file=sys.stderr)
        sys.exit(1)

    generate_comparison(experiments, args.output)


if __name__ == "__main__":
    main()
