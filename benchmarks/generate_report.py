#!/usr/bin/env python3
"""
Generate publication-quality visual report from benchmark results.
Produces 5 chart types and a markdown report embedding them.

Usage:
    python generate_report.py                          # Auto-detect results
    python generate_report.py --results-dir results/m1-max-64gb
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Install with: pip install matplotlib numpy")
    sys.exit(1)

# ─── Shared Config ───────────────────────────────────────
MODELS = {
    "bonsai-8b":   {"label": "Bonsai 8B",   "family": "Bonsai", "size_gb": 1.28},
    "bonsai-4b":   {"label": "Bonsai 4B",   "family": "Bonsai", "size_gb": 0.57},
    "bonsai-1.7b": {"label": "Bonsai 1.7B", "family": "Bonsai", "size_gb": 0.24},
    "qwen-7b":     {"label": "Qwen 7B",     "family": "Qwen",   "size_gb": 4.4},
    "qwen-3b":     {"label": "Qwen 3B",     "family": "Qwen",   "size_gb": 1.8},
    "qwen-1.5b":   {"label": "Qwen 1.5B",   "family": "Qwen",   "size_gb": 0.9},
    "llama-3b":    {"label": "Llama 3B",     "family": "Llama",  "size_gb": 1.8},
    "llama-1b":    {"label": "Llama 1B",     "family": "Llama",  "size_gb": 0.7},
    "phi-3.5":     {"label": "Phi-3.5",      "family": "Phi",    "size_gb": 2.2},
    "gemma-2b":    {"label": "Gemma 2B",     "family": "Gemma",  "size_gb": 1.4},
}

CATEGORIES = [
    "Math", "Reasoning", "Code", "Writing", "Tool Calling",
    "Instruction Following", "RAG / Grounded QA", "Multilingual",
    "Classification", "Agentic Planning",
]

FAMILY_COLORS = {
    "Bonsai": "#0077B6",
    "Qwen":   "#E07A00",
    "Llama":  "#8B5CF6",
    "Phi":    "#059669",
    "Gemma":  "#DC2626",
}

FAMILY_CMAP = {
    "Bonsai": ["#90E0EF", "#0096C7", "#023E8A"],
    "Qwen":   ["#FBBF24", "#F59E0B", "#D97706"],
    "Llama":  ["#C4B5FD", "#8B5CF6"],
    "Phi":    ["#6EE7B7"],
    "Gemma":  ["#FCA5A5"],
}

DPI = 150
FIGSIZE = (12, 7)


# ─── Data Loading ────────────────────────────────────────
def find_results_dir(base: Path) -> Path:
    """Find the results directory, checking for hardware subdirs."""
    # If subdirectories contain model dirs, use base directly
    for mk in MODELS:
        if (base / mk).is_dir():
            return base
    # Otherwise look for a single hardware subdir
    subdirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for sd in subdirs:
        for mk in MODELS:
            if (sd / mk).is_dir():
                return sd
    return base


def load_model_results(model_dir: Path) -> dict:
    results = {}
    for csv_file in sorted(model_dir.glob("bench_*.csv")):
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                cat = row["category"]
                if cat not in results:
                    results[cat] = {"passed": 0, "total": 0, "tps_values": []}
                results[cat]["total"] += 1
                if row["passed"] == "True":
                    results[cat]["passed"] += 1
                tps = float(row["tok_per_sec"]) if row.get("tok_per_sec") else 0
                if tps > 0:
                    results[cat]["tps_values"].append(tps)
    return results


def load_all(results_dir: Path):
    data = {}
    available = []
    for mk, info in MODELS.items():
        d = results_dir / mk
        if d.is_dir() and list(d.glob("bench_*.csv")):
            data[mk] = load_model_results(d)
            available.append(mk)
        else:
            data[mk] = {}
    return data, available


def model_summary(data: dict, mk: str):
    d = data[mk]
    passed = sum(v["passed"] for v in d.values())
    total = sum(v["total"] for v in d.values())
    pct = (passed / total * 100) if total else 0
    all_tps = [t for v in d.values() for t in v["tps_values"]]
    avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0
    return passed, total, pct, avg_tps


def get_color(mk: str, idx: int = 0) -> str:
    family = MODELS[mk]["family"]
    cmap = FAMILY_CMAP.get(family, ["#888888"])
    return cmap[min(idx, len(cmap) - 1)]


def assign_colors(available: list[str]) -> dict:
    family_counter = {}
    colors = {}
    for mk in available:
        fam = MODELS[mk]["family"]
        idx = family_counter.get(fam, 0)
        colors[mk] = get_color(mk, idx)
        family_counter[fam] = idx + 1
    return colors


# ─── Chart 1: Overall Comparison Bar ────────────────────
def chart_overall(data, available, colors, out_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = [MODELS[mk]["label"] for mk in available]
    pcts = [model_summary(data, mk)[2] for mk in available]
    bar_colors = [colors[mk] for mk in available]

    bars = ax.bar(labels, pcts, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, mk, pct in zip(bars, available, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.text(bar.get_x() + bar.get_width()/2, -4,
                MODELS[mk]["size_gb"].__str__() + " GB",
                ha="center", va="top", fontsize=7, color="#666")

    ax.set_ylabel("Score (%)")
    ax.set_title("Overall Benchmark Score by Model", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "overall_comparison.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Generated: {path}")


# ─── Chart 2: Category Heatmap ──────────────────────────
def chart_heatmap(data, available, out_dir):
    labels = [MODELS[mk]["label"] for mk in available]
    matrix = []
    annots = []
    for cat in CATEGORIES:
        row_vals, row_annots = [], []
        for mk in available:
            d = data[mk].get(cat, {"passed": 0, "total": 0})
            p, t = d["passed"], d["total"]
            pct = (p / t * 100) if t else 0
            row_vals.append(pct)
            row_annots.append(f"{p}/{t}")
        matrix.append(row_vals)
        annots.append(row_annots)

    fig, ax = plt.subplots(figsize=(max(10, len(available) * 1.4), 7))
    arr = np.array(matrix)
    im = ax.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES, fontsize=9)

    for i in range(len(CATEGORIES)):
        for j in range(len(available)):
            ax.text(j, i, annots[i][j], ha="center", va="center", fontsize=8,
                    color="white" if arr[i, j] < 50 else "black")

    fig.colorbar(im, ax=ax, label="Pass Rate (%)", shrink=0.8)
    ax.set_title("Category Performance Heatmap", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = out_dir / "category_heatmap.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Generated: {path}")


# ─── Chart 3: Intelligence Density ──────────────────────
def chart_density(data, available, colors, out_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sizes, pcts, labels_list = [], [], []
    for mk in available:
        _, _, pct, _ = model_summary(data, mk)
        sz = MODELS[mk]["size_gb"]
        sizes.append(sz)
        pcts.append(pct)
        labels_list.append(MODELS[mk]["label"])

        marker = "*" if MODELS[mk]["family"] == "Bonsai" else "o"
        ax.scatter(sz, pct, c=colors[mk], s=180, marker=marker, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(MODELS[mk]["label"], (sz, pct), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    # Pareto frontier
    points = sorted(zip(sizes, pcts), key=lambda p: p[0])
    frontier = []
    best_pct = -1
    for s, p in points:
        if p > best_pct:
            frontier.append((s, p))
            best_pct = p
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "--", color="#999", linewidth=1, alpha=0.7, label="Pareto frontier")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}" if x < 1 else f"{x:.0f}"))
    ax.set_xlabel("Model Size (GB, log scale)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Intelligence Density: Score vs Model Size", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "intelligence_density.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Generated: {path}")


# ─── Chart 4: Speed vs Accuracy ─────────────────────────
def chart_speed_accuracy(data, available, colors, out_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for mk in available:
        _, _, pct, avg_tps = model_summary(data, mk)
        sz = MODELS[mk]["size_gb"]
        ax.scatter(avg_tps, pct, c=colors[mk], s=sz * 80 + 40, zorder=5,
                   edgecolors="black", linewidths=0.5, alpha=0.85)
        ax.annotate(MODELS[mk]["label"], (avg_tps, pct), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    ax.set_xlabel("Average tok/s")
    ax.set_ylabel("Score (%)")
    ax.set_title("Speed vs Accuracy (point size = model size)", fontweight="bold", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = out_dir / "speed_vs_accuracy.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Generated: {path}")


# ─── Chart 5: Radar / Spider ────────────────────────────
def chart_radar(data, available, colors, out_dir):
    N = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for mk in available:
        values = []
        for cat in CATEGORIES:
            d = data[mk].get(cat, {"passed": 0, "total": 0})
            p, t = d["passed"], d["total"]
            values.append((p / t * 100) if t else 0)
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=MODELS[mk]["label"], color=colors[mk])
        ax.fill(angles, values, alpha=0.08, color=colors[mk])

    ax.set_xticks(angles[:-1])
    short_cats = [c.replace(" / Grounded QA", "").replace("Instruction Following", "Instruction") for c in CATEGORIES]
    ax.set_xticklabels(short_cats, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Category Performance Radar", fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    path = out_dir / "category_radar.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated: {path}")


# ─── Report Markdown ────────────────────────────────────
def generate_report_md(data, available, out_dir):
    lines = [
        "# MLX Bonsai Benchmarks — Visual Report\n",
        "## Overall Comparison\n",
        "![Overall Comparison](overall_comparison.png)\n",
        "Grouped bar chart showing each model's overall benchmark score. ",
        "Model sizes annotated below each bar.\n",
        "## Category Heatmap\n",
        "![Category Heatmap](category_heatmap.png)\n",
        "Pass rate by category for every model. Green = strong, red = weak.\n",
        "## Intelligence Density\n",
        "![Intelligence Density](intelligence_density.png)\n",
        "Score vs model size on a log scale. Stars = Bonsai (1-bit), circles = others (4-bit). ",
        "Models above the Pareto frontier offer the best quality-per-byte.\n",
        "## Speed vs Accuracy\n",
        "![Speed vs Accuracy](speed_vs_accuracy.png)\n",
        "Trade-off between inference speed and benchmark accuracy. ",
        "Point size proportional to model disk size.\n",
        "## Category Radar\n",
        "![Category Radar](category_radar.png)\n",
        "Spider chart overlaying all models across the 10 benchmark categories.\n",
    ]

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines))
    print(f"  Generated: {path}")


# ─── Main ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate visual benchmark report")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results",
                        help="Path to results directory")
    args = parser.parse_args()

    results_dir = find_results_dir(args.results_dir)
    print(f"Reading results from: {results_dir}")

    data, available = load_all(results_dir)
    if not available:
        print("ERROR: No model results found.")
        sys.exit(1)

    print(f"Found results for: {', '.join(available)}")

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)

    colors = assign_colors(available)

    print("\nGenerating charts...")
    chart_overall(data, available, colors, out_dir)
    chart_heatmap(data, available, out_dir)
    chart_density(data, available, colors, out_dir)
    chart_speed_accuracy(data, available, colors, out_dir)
    chart_radar(data, available, colors, out_dir)
    generate_report_md(data, available, out_dir)

    print(f"\nAll reports saved to: {out_dir}/")


if __name__ == "__main__":
    main()
