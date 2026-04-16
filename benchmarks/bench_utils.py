"""
Shared benchmark utilities for MLX Bonsai Benchmarks.
Handles API calls, performance tracking, display, and CSV export.
"""

import requests
import time
import json
import csv
import subprocess
import tempfile
import os
import sys
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Configuration ────────────────────────────────────────
DEFAULT_API_URL = os.environ.get("BONSAI_API_URL", "http://localhost:8081/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("BONSAI_MODEL", "models/Bonsai-8B-mlx")
DEFAULT_MAX_TOKENS = 1024
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# ─── Colors ───────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class TestResult:
    """Result of a single benchmark test."""
    test_num: int = 0
    category: str = ""
    name: str = ""
    difficulty: str = ""
    passed: bool = False
    note: str = ""
    elapsed_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tok_per_sec: float = 0.0
    response: str = ""
    tool_calls: list = field(default_factory=list)
    exec_output: str = ""
    error: str = ""


def query(
    messages: list,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tools: list = None,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> dict:
    """Send a request to the MLX server and return parsed response with timing."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    start = time.perf_counter()
    try:
        resp = requests.post(api_url, json=payload, timeout=180)
        elapsed = time.perf_counter() - start
        data = resp.json()
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"error": str(e), "elapsed_sec": round(elapsed, 2)}

    if "error" in data:
        return {"error": data["error"], "elapsed_sec": round(elapsed, 2)}

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    choice = data["choices"][0]
    content = choice["message"].get("content", "")
    tool_calls = choice["message"].get("tool_calls", [])
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return {
        "content": content,
        "tool_calls": tool_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_sec": round(elapsed, 2),
        "tok_per_sec": round(tok_per_sec, 1),
        "finish_reason": choice.get("finish_reason", ""),
    }


def run_python(code: str, timeout: int = 10) -> dict:
    """Execute Python code in a temp file and return stdout/stderr/success."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[len("```"):]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "TIMEOUT"}
        finally:
            os.unlink(f.name)


class BenchmarkSuite:
    """
    Manages test registration, execution, display, and CSV export.

    Usage:
        suite = BenchmarkSuite("Math", "bench_math")
        
        @suite.test("Addition word problem", "Easy")
        def test_add():
            r = query([...])
            passed = "42" in r.get("content", "")
            return r, passed, "Expected 42"
        
        suite.run()
    """

    def __init__(self, category: str, slug: str, description: str = ""):
        self.category = category
        self.slug = slug
        self.description = description
        self.tests = []

    def test(self, name: str, difficulty: str = "Medium"):
        """Decorator to register a test function."""
        def wrapper(fn):
            self.tests.append({"name": name, "difficulty": difficulty, "fn": fn})
            return fn
        return wrapper

    def run(self) -> list[TestResult]:
        """Execute all registered tests and return results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = len(self.tests)

        print(f"\n{BOLD}{'=' * 70}")
        print(f"  {self.category.upper()} BENCHMARK")
        print(f"  {timestamp}")
        print(f"  Model: {DEFAULT_MODEL}  |  Tests: {total}")
        if self.description:
            print(f"  {self.description}")
        print(f"{'=' * 70}{RESET}\n")

        results = []

        for i, t in enumerate(self.tests, 1):
            name = t["name"]
            difficulty = t["difficulty"]
            diff_color = {"Easy": GREEN, "Medium": YELLOW, "Hard": RED}.get(difficulty, RESET)

            print(f"{BOLD}[{i}/{total}] {name} {diff_color}[{difficulty}]{RESET}")
            print(f"{DIM}{'─' * 70}{RESET}")

            try:
                raw, passed, note = t["fn"]()
            except Exception as e:
                raw = {"content": "", "elapsed_sec": 0, "completion_tokens": 0, "prompt_tokens": 0, "tok_per_sec": 0}
                passed = False
                note = f"EXCEPTION: {e}"

            # Build TestResult
            tr = TestResult(
                test_num=i,
                category=self.category,
                name=name,
                difficulty=difficulty,
                passed=passed,
                note=note[:300],
                elapsed_sec=raw.get("elapsed_sec", 0),
                prompt_tokens=raw.get("prompt_tokens", 0),
                completion_tokens=raw.get("completion_tokens", 0),
                total_tokens=raw.get("total_tokens", 0),
                tok_per_sec=raw.get("tok_per_sec", 0),
                response=raw.get("content", "")[:1000],
                tool_calls=raw.get("tool_calls", []),
                exec_output=raw.get("exec_output", ""),
                error=raw.get("error", ""),
            )
            if tr.tok_per_sec == 0 and tr.elapsed_sec > 0 and tr.completion_tokens > 0:
                tr.tok_per_sec = round(tr.completion_tokens / tr.elapsed_sec, 1)

            results.append(tr)

            # Display
            status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
            print(f"\n  {status}  {DIM}{note[:120]}{RESET}")

            preview = tr.response[:300].replace("\n", "\n  ") if tr.response else ""
            if preview:
                print(f"\n  {DIM}{preview}{'...' if len(tr.response) > 300 else ''}{RESET}")

            if tr.tool_calls:
                print(f"\n  {CYAN}Tool Calls: {json.dumps(tr.tool_calls, indent=2)[:400]}{RESET}")

            if tr.exec_output:
                print(f"\n  {CYAN}Exec: {tr.exec_output[:200]}{RESET}")

            print(f"\n  {DIM}⏱ {tr.elapsed_sec}s | 🔤 {tr.completion_tokens} tok | ⚡ {tr.tok_per_sec} tok/s{RESET}\n")

        # Summary
        self._print_summary(results)
        self._save_csv(results, timestamp)
        return results

    def _print_summary(self, results: list[TestResult]):
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_tokens = sum(r.completion_tokens for r in results)
        total_time = sum(r.elapsed_sec for r in results)
        tps_vals = [r.tok_per_sec for r in results if r.tok_per_sec > 0]
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0

        color = GREEN if passed == total else (YELLOW if passed > 0 else RED)
        bar = "█" * passed + "░" * (total - passed)

        print(f"{BOLD}{'─' * 70}")
        print(f"  {self.category} Summary")
        print(f"{'─' * 70}{RESET}")
        print(f"  Score:         {color}{bar}  {passed}/{total}{RESET}")
        print(f"  Total tokens:  {total_tokens}")
        print(f"  Total time:    {total_time:.1f}s")
        print(f"  Avg tok/s:     {avg_tps:.1f}")
        if tps_vals:
            print(f"  Min tok/s:     {min(tps_vals):.1f}")
            print(f"  Max tok/s:     {max(tps_vals):.1f}")
        print()

    def _save_csv(self, results: list[TestResult], timestamp: str):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename = f"{self.slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)

        rows = []
        for r in results:
            d = asdict(r)
            d["tool_calls"] = json.dumps(r.tool_calls)[:500]
            d["response"] = r.response[:500]
            rows.append(d)

        if rows:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {GREEN}Saved: {filepath}{RESET}\n")
