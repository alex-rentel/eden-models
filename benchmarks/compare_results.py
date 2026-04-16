#!/usr/bin/env python3
"""
Compare benchmark results across all models.
Reads CSVs from results/<model>/ subfolders and builds a comparison table.
"""

import argparse
import csv
import os
from pathlib import Path

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "bonsai-8b":  {"label": "Bonsai 8B",  "family": "Bonsai", "format": "1-bit", "size": "~1.28 GB", "repo": "prism-ml/Bonsai-8B-mlx-1bit"},
    "bonsai-4b":  {"label": "Bonsai 4B",  "family": "Bonsai", "format": "1-bit", "size": "~0.57 GB", "repo": "prism-ml/Bonsai-4B-mlx-1bit"},
    "bonsai-1.7b":{"label": "Bonsai 1.7B","family": "Bonsai", "format": "1-bit", "size": "~0.24 GB", "repo": "prism-ml/Bonsai-1.7B-mlx-1bit"},
    "qwen-7b":    {"label": "Qwen 7B",    "family": "Qwen",   "format": "4-bit", "size": "~4.4 GB",  "repo": "mlx-community/Qwen2.5-7B-Instruct-4bit"},
    "qwen-3b":    {"label": "Qwen 3B",    "family": "Qwen",   "format": "4-bit", "size": "~1.8 GB",  "repo": "mlx-community/Qwen2.5-3B-Instruct-4bit"},
    "qwen-1.5b":  {"label": "Qwen 1.5B",  "family": "Qwen",   "format": "4-bit", "size": "~0.9 GB",  "repo": "mlx-community/Qwen2.5-1.5B-Instruct-4bit"},
    "llama-3b":   {"label": "Llama 3B",   "family": "Llama",  "format": "4-bit", "size": "~1.8 GB",  "repo": "mlx-community/Llama-3.2-3B-Instruct-4bit"},
    "llama-1b":   {"label": "Llama 1B",   "family": "Llama",  "format": "4-bit", "size": "~0.7 GB",  "repo": "mlx-community/Llama-3.2-1B-Instruct-4bit"},
    "phi-3.5":    {"label": "Phi-3.5",    "family": "Phi",    "format": "4-bit", "size": "~2.2 GB",  "repo": "mlx-community/Phi-3.5-mini-instruct-4bit"},
    "gemma-2b":   {"label": "Gemma 2B",   "family": "Gemma",  "format": "4-bit", "size": "~1.4 GB",  "repo": "mlx-community/gemma-2-2b-it-4bit"},
}

# Canonical category order
CATEGORIES = ["Math", "Reasoning", "Code", "Writing", "Tool Calling",
              "Instruction Following", "RAG / Grounded QA", "Multilingual",
              "Classification", "Agentic Planning"]


def load_model_results(model_dir: Path) -> dict:
    """Load all CSVs from a model dir, return {category: {passed, total, tps_values}}."""
    results = {}
    for csv_file in sorted(model_dir.glob("bench_*.csv")):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row["category"]
                if cat not in results:
                    results[cat] = {"passed": 0, "total": 0, "tps_values": []}
                results[cat]["total"] += 1
                if row["passed"] == "True":
                    results[cat]["passed"] += 1
                tps = float(row["tok_per_sec"]) if row["tok_per_sec"] else 0
                if tps > 0:
                    results[cat]["tps_values"].append(tps)
    return results


def find_results_dir(base: Path) -> Path:
    """Auto-detect results dir: check for model subdirs directly or inside hardware subdirs."""
    for mk in MODELS:
        if (base / mk).is_dir():
            return base
    for sd in sorted(base.iterdir()):
        if sd.is_dir() and not sd.name.startswith("."):
            for mk in MODELS:
                if (sd / mk).is_dir():
                    return sd
    return base


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results across models")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Path to results directory")
    args = parser.parse_args()

    results_dir = find_results_dir(args.results_dir)

    all_data = {}
    for model_key, info in MODELS.items():
        model_dir = results_dir / model_key
        if model_dir.exists():
            all_data[model_key] = load_model_results(model_dir)
        else:
            all_data[model_key] = {}

    # Build markdown table
    model_keys = list(MODELS.keys())
    headers = ["Category"] + [MODELS[k]["label"] for k in model_keys]

    rows = []
    totals = {k: {"passed": 0, "total": 0, "tps_values": []} for k in model_keys}

    for cat in CATEGORIES:
        row = [cat]
        for mk in model_keys:
            data = all_data[mk].get(cat, {"passed": 0, "total": 0, "tps_values": []})
            p, t = data["passed"], data["total"]
            avg_tps = sum(data["tps_values"]) / len(data["tps_values"]) if data["tps_values"] else 0
            totals[mk]["passed"] += p
            totals[mk]["total"] += t
            totals[mk]["tps_values"].extend(data["tps_values"])
            row.append(f"{p}/{t} ({avg_tps:.0f} t/s)")
        rows.append(row)

    # Overall row
    overall_row = ["**Overall**"]
    for mk in model_keys:
        p, t = totals[mk]["passed"], totals[mk]["total"]
        pct = (p / t * 100) if t > 0 else 0
        avg_tps = sum(totals[mk]["tps_values"]) / len(totals[mk]["tps_values"]) if totals[mk]["tps_values"] else 0
        overall_row.append(f"**{p}/{t} ({pct:.1f}%) @ {avg_tps:.0f} t/s**")
    rows.append(overall_row)

    # Print markdown table
    print()
    print("## Benchmark Results: Bonsai 1-bit vs Qwen 2.5 4-bit")
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()

    # Models tested table
    print("## Models Tested")
    print()
    print("| Model | Family | HuggingFace Repo | Format | Size |")
    print("| --- | --- | --- | --- | --- |")
    for mk in model_keys:
        info = MODELS[mk]
        print(f"| {info['label']} | {info['family']} | `{info['repo']}` | {info['format']} | {info['size']} |")
    print()

    # Save combined CSV
    csv_path = results_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category"] + [MODELS[k]["label"] + " Score" for k in model_keys] + [MODELS[k]["label"] + " Avg t/s" for k in model_keys])
        for cat in CATEGORIES:
            score_cols = []
            tps_cols = []
            for mk in model_keys:
                data = all_data[mk].get(cat, {"passed": 0, "total": 0, "tps_values": []})
                score_cols.append(f"{data['passed']}/{data['total']}")
                avg_tps = sum(data["tps_values"]) / len(data["tps_values"]) if data["tps_values"] else 0
                tps_cols.append(f"{avg_tps:.1f}")
            writer.writerow([cat] + score_cols + tps_cols)
        # Overall
        score_cols = []
        tps_cols = []
        for mk in model_keys:
            p, t = totals[mk]["passed"], totals[mk]["total"]
            score_cols.append(f"{p}/{t}")
            avg_tps = sum(totals[mk]["tps_values"]) / len(totals[mk]["tps_values"]) if totals[mk]["tps_values"] else 0
            tps_cols.append(f"{avg_tps:.1f}")
        writer.writerow(["Overall"] + score_cols + tps_cols)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
