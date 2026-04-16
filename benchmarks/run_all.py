#!/usr/bin/env python3
"""
Run all MLX Bonsai benchmarks and generate a combined report.

Usage:
    python run_all.py              # Run all benchmarks
    python run_all.py math code    # Run specific benchmarks
    python run_all.py --list       # List available benchmarks
    python run_all.py --runs 3     # Run each benchmark 3 times (variance mode)
"""

import sys
import os
import time
import csv
import importlib
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))

# ─── Colors ───────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ─── Available benchmark modules ──────────────────────────
BENCHMARKS = {
    "math":           ("benchmarks.bench_math",           "Math"),
    "reasoning":      ("benchmarks.bench_reasoning",      "Reasoning"),
    "code":           ("benchmarks.bench_code",           "Code"),
    "writing":        ("benchmarks.bench_writing",        "Writing"),
    "tools":          ("benchmarks.bench_tools",          "Tool Calling"),
    "instruction":    ("benchmarks.bench_instruction",    "Instruction Following"),
    "rag":            ("benchmarks.bench_rag",            "RAG / Grounded QA"),
    "multilingual":   ("benchmarks.bench_multilingual",   "Multilingual"),
    "classification": ("benchmarks.bench_classification", "Classification"),
    "agentic":        ("benchmarks.bench_agentic",        "Agentic Planning"),
}


def run_benchmarks(selected: list[str] = None, num_runs: int = 1):
    """Run selected (or all) benchmark suites."""
    if selected is None:
        selected = list(BENCHMARKS.keys())

    if num_runs > 1:
        return run_benchmarks_variance(selected, num_runs)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = os.environ.get("BONSAI_MODEL", "models/Bonsai-8B-mlx")

    print(f"\n{BOLD}{'═' * 70}")
    print(f"  MLX BONSAI BENCHMARKS — FULL SUITE")
    print(f"  {timestamp}")
    print(f"  Model: {model}")
    print(f"  Suites: {', '.join(selected)}")
    print(f"{'═' * 70}{RESET}\n")

    all_results = {}
    overall_start = time.perf_counter()

    for key in selected:
        if key not in BENCHMARKS:
            print(f"{RED}  Unknown benchmark: {key}{RESET}")
            print(f"  Available: {', '.join(BENCHMARKS.keys())}")
            continue

        module_path, display_name = BENCHMARKS[key]

        try:
            mod = importlib.import_module(module_path)
            results = mod.suite.run()
            all_results[key] = results
        except Exception as e:
            print(f"{RED}  Error running {display_name}: {e}{RESET}")
            all_results[key] = []

    overall_time = time.perf_counter() - overall_start

    # ── Combined Summary ──────────────────────────────────
    print(f"\n{BOLD}{'═' * 70}")
    print(f"  COMBINED RESULTS")
    print(f"{'═' * 70}{RESET}\n")

    total_passed = 0
    total_tests = 0
    total_tokens = 0
    total_tps_vals = []

    for key in selected:
        if key not in all_results:
            continue
        results = all_results[key]
        if not results:
            continue

        display_name = BENCHMARKS[key][1]
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_passed += passed
        total_tests += total
        total_tokens += sum(r.completion_tokens for r in results)
        total_tps_vals.extend([r.tok_per_sec for r in results if r.tok_per_sec > 0])

        color = GREEN if passed == total else (YELLOW if passed > 0 else RED)
        bar = "█" * passed + "░" * (total - passed)
        pct = (passed / total * 100) if total > 0 else 0

        print(f"  {display_name:25s}  {color}{bar}  {passed}/{total}  ({pct:.0f}%){RESET}")

    avg_tps = sum(total_tps_vals) / len(total_tps_vals) if total_tps_vals else 0
    overall_pct = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"\n  {'─' * 55}")

    overall_color = GREEN if overall_pct >= 80 else (YELLOW if overall_pct >= 60 else RED)
    print(f"  {BOLD}Overall Score:       {overall_color}{total_passed}/{total_tests} ({overall_pct:.1f}%){RESET}")
    print(f"  Total tokens:       {total_tokens:,}")
    print(f"  Total time:         {overall_time:.1f}s")
    print(f"  Avg tok/s:          {avg_tps:.1f}")
    if total_tps_vals:
        print(f"  Min tok/s:          {min(total_tps_vals):.1f}")
        print(f"  Max tok/s:          {max(total_tps_vals):.1f}")

    print(f"\n  {DIM}Results saved per-suite in results/{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}\n")


def run_benchmarks_variance(selected: list[str], num_runs: int):
    """Run each benchmark suite N times and report variance."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = os.environ.get("BONSAI_MODEL", "models/Bonsai-8B-mlx")
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    print(f"\n{BOLD}{'═' * 70}")
    print(f"  MLX BONSAI BENCHMARKS — VARIANCE MODE ({num_runs} runs)")
    print(f"  {timestamp}")
    print(f"  Model: {model}")
    print(f"  Suites: {', '.join(selected)}")
    print(f"{'═' * 70}{RESET}\n")

    overall_start = time.perf_counter()

    # {key: {test_name: [results_per_run]}}
    all_runs = {}

    for run_idx in range(1, num_runs + 1):
        print(f"\n{BOLD}{CYAN}  ── Run {run_idx}/{num_runs} ──{RESET}\n")

        for key in selected:
            if key not in BENCHMARKS:
                continue
            module_path, display_name = BENCHMARKS[key]
            if key not in all_runs:
                all_runs[key] = {}
            try:
                # Reimport to reset state
                mod = importlib.import_module(module_path)
                importlib.reload(mod)
                results = mod.suite.run()
                for r in results:
                    if r.name not in all_runs[key]:
                        all_runs[key][r.name] = []
                    all_runs[key][r.name].append(r)
            except Exception as e:
                print(f"{RED}  Error running {display_name} (run {run_idx}): {e}{RESET}")

    overall_time = time.perf_counter() - overall_start

    # ── Variance Summary ──────────────────────────────────
    print(f"\n{BOLD}{'═' * 70}")
    print(f"  VARIANCE RESULTS ({num_runs} runs)")
    print(f"{'═' * 70}{RESET}\n")

    total_passed_avg = 0
    total_tests = 0

    for key in selected:
        if key not in all_runs or not all_runs[key]:
            continue

        display_name = BENCHMARKS[key][1]
        test_names = list(all_runs[key].keys())
        total = len(test_names)
        total_tests += total

        pass_counts = []
        run_scores = [0] * num_runs
        tps_all = []

        for tname in test_names:
            runs = all_runs[key][tname]
            n_pass = sum(1 for r in runs if r.passed)
            pass_counts.append(n_pass)
            for i, r in enumerate(runs):
                if r.passed:
                    run_scores[i] += 1
                if r.tok_per_sec > 0:
                    tps_all.append(r.tok_per_sec)

        # Majority vote: passed if >50% of runs pass
        majority_passed = sum(1 for pc in pass_counts if pc > num_runs / 2)
        total_passed_avg += majority_passed
        avg_score = sum(run_scores) / num_runs

        avg_tps = sum(tps_all) / len(tps_all) if tps_all else 0
        std_tps = (sum((t - avg_tps)**2 for t in tps_all) / len(tps_all))**0.5 if len(tps_all) > 1 else 0

        run_str = ",".join(str(s) for s in run_scores)
        color = GREEN if majority_passed == total else (YELLOW if majority_passed > 0 else RED)
        bar = "█" * majority_passed + "░" * (total - majority_passed)

        print(f"  {display_name:25s}  {color}{bar}  {avg_score:.1f}/{total} avg ({run_str} across runs)  {avg_tps:.1f} ± {std_tps:.1f} tok/s{RESET}")

        # Save variance CSV
        _save_variance_csv(key, all_runs[key], num_runs, results_dir, timestamp)

    overall_pct = (total_passed_avg / total_tests * 100) if total_tests > 0 else 0
    overall_color = GREEN if overall_pct >= 80 else (YELLOW if overall_pct >= 60 else RED)

    print(f"\n  {'─' * 55}")
    print(f"  {BOLD}Overall (majority vote): {overall_color}{total_passed_avg}/{total_tests} ({overall_pct:.1f}%){RESET}")
    print(f"  Total time:            {overall_time:.1f}s")
    print(f"\n{BOLD}{'═' * 70}{RESET}\n")


def _save_variance_csv(key, test_data, num_runs, results_dir, timestamp):
    """Save a variance CSV with per-run results plus aggregates."""
    os.makedirs(results_dir, exist_ok=True)
    slug = f"bench_{key}_variance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = os.path.join(results_dir, f"{slug}.csv")

    fieldnames = ["test_name", "difficulty", "num_runs"]
    for i in range(1, num_runs + 1):
        fieldnames.extend([f"run{i}_passed", f"run{i}_tok_per_sec", f"run{i}_elapsed_sec"])
    fieldnames.extend(["pass_rate", "majority_pass", "mean_tok_per_sec", "std_tok_per_sec"])

    rows = []
    for tname, runs in test_data.items():
        row = {
            "test_name": tname,
            "difficulty": runs[0].difficulty if runs else "",
            "num_runs": num_runs,
        }
        tps_vals = []
        pass_count = 0
        for i, r in enumerate(runs, 1):
            row[f"run{i}_passed"] = r.passed
            row[f"run{i}_tok_per_sec"] = round(r.tok_per_sec, 1)
            row[f"run{i}_elapsed_sec"] = round(r.elapsed_sec, 2)
            if r.passed:
                pass_count += 1
            if r.tok_per_sec > 0:
                tps_vals.append(r.tok_per_sec)

        row["pass_rate"] = f"{pass_count}/{num_runs}"
        row["majority_pass"] = pass_count > num_runs / 2
        row["mean_tok_per_sec"] = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0
        std = (sum((t - row["mean_tok_per_sec"])**2 for t in tps_vals) / len(tps_vals))**0.5 if len(tps_vals) > 1 else 0
        row["std_tok_per_sec"] = round(std, 1)
        rows.append(row)

    if rows:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {GREEN}Saved variance: {filepath}{RESET}")


def main():
    args = sys.argv[1:]

    if "--list" in args or "-l" in args:
        print(f"\n{BOLD}Available benchmarks:{RESET}")
        for key, (_, name) in BENCHMARKS.items():
            print(f"  {key:20s}  {name}")
        print(f"\n  Usage: python run_all.py [bench1] [bench2] ...")
        print(f"  Example: python run_all.py math code tools\n")
        return

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # Parse --runs N
    num_runs = 1
    if "--runs" in args:
        idx = args.index("--runs")
        try:
            num_runs = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        except (IndexError, ValueError):
            print(f"{RED}  --runs requires an integer argument{RESET}")
            return

    selected = args if args else None
    run_benchmarks(selected, num_runs)


if __name__ == "__main__":
    main()
