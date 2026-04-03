#!/usr/bin/env python3
"""Tool-calling assessment for Eden fine-tuned models.

Compares base model vs fine-tuned (with LoRA adapter) on:
1. Single tool selection (20 cases)
2. Argument formatting (10 cases)
3. Multi-tool (5 cases)
4. No tool needed (10 cases)
5. Wrong tool avoidance (5 cases)

Usage:
    python3 eval/tool_calling_eval.py \
        --model mlx-community/Qwen3-1.7B-4bit \
        --adapter-path adapters/qwen3-1.7b-tools-v1 \
        --output eval/results/v1_results.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load


# ─── Tool Definitions ─────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files and directories at a given path",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
            "show_hidden": {"type": "boolean", "description": "Include hidden files", "default": False},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to read"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file, creating it if it doesn't exist",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Execute a shell command and return output",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results", "default": 5},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        }, "required": ["city"]},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Compute a mathematical expression and return the result",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Math expression to compute"},
        }, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body text"},
        }, "required": ["to", "subject", "body"]},
    }},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": "Set a reminder for a specific time",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string", "description": "Reminder message"},
            "time": {"type": "string", "description": "When to remind (ISO 8601 datetime)"},
        }, "required": ["message", "time"]},
    }},
    {"type": "function", "function": {
        "name": "translate_text",
        "description": "Translate text from one language to another",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "source_lang": {"type": "string", "description": "Source language code"},
            "target_lang": {"type": "string", "description": "Target language code"},
        }, "required": ["text", "target_lang"]},
    }},
]

SYSTEM_MSG = "You are a helpful assistant with access to tools. Use them when needed to answer user questions accurately."


# ─── Test Cases ────────────────────────────────────────────

TEST_CASES = []

def test(category, difficulty):
    def decorator(fn):
        TEST_CASES.append({"fn": fn, "category": category, "difficulty": difficulty,
                           "name": fn.__name__.replace("_", " ").strip()})
        return fn
    return decorator

# --- Single Tool Selection (20 cases) ---

@test("single_tool", "easy")
def list_directory():
    return "List the files in /home/user/projects", "list_files", {"path": "/home/user/projects"}

@test("single_tool", "easy")
def read_specific_file():
    return "Show me the contents of /etc/hostname", "read_file", {"path": "/etc/hostname"}

@test("single_tool", "easy")
def weather_query():
    return "What's the weather like in Tokyo right now?", "get_weather", {"city": "Tokyo"}

@test("single_tool", "easy")
def web_search_current():
    return "Who won the Super Bowl in 2024?", "search_web", None

@test("single_tool", "easy")
def run_git_status():
    return "Run git status in the current directory", "run_command", {"command": "git status"}

@test("single_tool", "easy")
def write_hello_file():
    return "Create a file at /tmp/hello.txt containing 'Hello World'", "write_file", {"path": "/tmp/hello.txt"}

@test("single_tool", "easy")
def send_email_task():
    return "Send an email to alice@example.com with subject 'Meeting' and body 'See you at 3pm'", "send_email", None

@test("single_tool", "easy")
def translate_to_french():
    return "Translate 'Good morning' to French", "translate_text", {"target_lang": "fr"}

@test("single_tool", "easy")
def set_reminder():
    return "Remind me to call the dentist tomorrow at 9am", "create_reminder", None

@test("single_tool", "easy")
def calculate_expression():
    return "What is 245 * 18 + 33?", "calculate", None

@test("single_tool", "medium")
def weather_fahrenheit():
    return "What's the temperature in New York in fahrenheit?", "get_weather", {"city": "New York"}

@test("single_tool", "medium")
def list_hidden_files():
    return "Show me all files including hidden ones in /home/user", "list_files", {"show_hidden": True}

@test("single_tool", "medium")
def search_technical():
    return "Search for the latest Python 3.13 release notes", "search_web", None

@test("single_tool", "medium")
def read_config():
    return "Read the configuration file at ~/.config/app/settings.json", "read_file", None

@test("single_tool", "medium")
def run_pip_install():
    return "Install the requests library using pip", "run_command", None

@test("single_tool", "medium")
def translate_spanish():
    return "How do you say 'Where is the train station?' in Spanish?", "translate_text", None

@test("single_tool", "medium")
def weather_specific_city():
    return "Is it raining in London right now?", "get_weather", {"city": "London"}

@test("single_tool", "medium")
def write_python_file():
    return "Write a Python script to /tmp/test.py that prints numbers 1 to 10", "write_file", None

@test("single_tool", "hard")
def search_ambiguous():
    return "Find information about the MLX framework by Apple", "search_web", None

@test("single_tool", "hard")
def command_with_pipe():
    return "Count how many Python files are in /home/user/projects", "run_command", None


# --- Argument Formatting (10 cases) ---

@test("arg_format", "medium")
def email_full_args():
    return "Email bob@corp.com about the quarterly report being ready for review", "send_email", None

@test("arg_format", "medium")
def weather_with_units():
    return "Temperature in Berlin in celsius please", "get_weather", {"city": "Berlin", "units": "celsius"}

@test("arg_format", "medium")
def translate_with_source():
    return "Translate this German text to English: 'Guten Morgen, wie geht es Ihnen?'", "translate_text", None

@test("arg_format", "medium")
def write_multiline():
    return "Create /tmp/notes.txt with the following content:\nLine 1: Hello\nLine 2: World", "write_file", None

@test("arg_format", "medium")
def calculate_complex():
    return "Compute (100 + 50) * 2 / 3", "calculate", None

@test("arg_format", "hard")
def list_with_special_path():
    return "List files in the directory '/home/user/my documents/work'", "list_files", None

@test("arg_format", "hard")
def command_with_quotes():
    return "Run: echo 'Hello World' | wc -c", "run_command", None

@test("arg_format", "hard")
def search_with_special_chars():
    return "Search for 'C++ vs Rust performance comparison 2024'", "search_web", None

@test("arg_format", "hard")
def write_json_content():
    return 'Write a JSON config file to /tmp/config.json with {"debug": true, "port": 8080}', "write_file", None

@test("arg_format", "hard")
def email_with_html():
    return "Email support@example.com with subject 'Bug Report #123' and body explaining a crash on startup with error code 500", "send_email", None


# --- Multi-tool (5 cases) ---

@test("multi_tool", "hard")
def read_then_search():
    return "Read the file /tmp/query.txt and then search the web for its contents", None, None

@test("multi_tool", "hard")
def list_then_read():
    return "List the files in /home/user/docs and then read README.md from that directory", None, None

@test("multi_tool", "hard")
def search_then_email():
    return "Search for the weather in Paris and email the results to alice@example.com", None, None

@test("multi_tool", "hard")
def calculate_then_write():
    return "Compute 15% tip on a $85 bill and write the result to /tmp/tip.txt", None, None

@test("multi_tool", "hard")
def translate_then_email():
    return "Translate 'Meeting confirmed for Monday' to Japanese and email it to tanaka@example.jp", None, None


# --- No Tool Needed (10 cases) ---

@test("no_tool", "easy")
def capital_of_france():
    return "What is the capital of France?", None, None

@test("no_tool", "easy")
def simple_math():
    return "What is 2 + 2?", None, None

@test("no_tool", "easy")
def explain_concept():
    return "What is machine learning?", None, None

@test("no_tool", "medium")
def coding_question():
    return "How do I write a for loop in Python?", None, None

@test("no_tool", "medium")
def history_question():
    return "When did World War II end?", None, None

@test("no_tool", "medium")
def definition():
    return "What does 'ephemeral' mean?", None, None

@test("no_tool", "medium")
def opinion_question():
    return "What are the pros and cons of using TypeScript?", None, None

@test("no_tool", "medium")
def greeting():
    return "Hello, how are you today?", None, None

@test("no_tool", "hard")
def math_reasoning():
    return "If a train travels at 60mph for 2.5 hours, how far does it go?", None, None

@test("no_tool", "hard")
def ambiguous_no_tool():
    return "Tell me about the Eiffel Tower", None, None


# --- Wrong Tool Avoidance (5 cases) ---

@test("wrong_tool", "hard")
def no_database_tool():
    return "Query the PostgreSQL database for all users created this month", None, None

@test("wrong_tool", "hard")
def no_image_tool():
    return "Generate an image of a sunset over the ocean", None, None

@test("wrong_tool", "hard")
def no_video_tool():
    return "Play the video file at /home/user/movie.mp4", None, None

@test("wrong_tool", "hard")
def no_calendar_tool():
    return "Schedule a meeting with John for next Tuesday at 2pm", None, None

@test("wrong_tool", "hard")
def no_payment_tool():
    return "Process a payment of $50 to merchant ID 12345", None, None


# ─── Assessment Logic ─────────────────────────────────────

def extract_tool_call(response: str):
    """Extract tool call from model response. Handles Qwen3 format."""
    # Look for <tool_call> tags (Qwen3 native format)
    tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if tc_match:
        try:
            data = json.loads(tc_match.group(1))
            return data.get("name"), data.get("arguments", {})
        except json.JSONDecodeError:
            pass

    # Look for function call JSON patterns
    fc_match = re.search(r'\{"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\})\s*\}', response, re.DOTALL)
    if fc_match:
        name = fc_match.group(1)
        try:
            args = json.loads(fc_match.group(2))
        except json.JSONDecodeError:
            args = {}
        return name, args

    return None, {}


def has_tool_call(response: str) -> bool:
    """Check if response contains any tool call."""
    return bool(re.search(r'<tool_call>|"name"\s*:\s*"\w+".*?"arguments"', response, re.DOTALL))


def assess_single(test_case, response: str) -> dict:
    """Assess a single test case response."""
    category = test_case["category"]
    prompt, expected_tool, expected_args = test_case["fn"]()

    name, args = extract_tool_call(response)
    made_call = has_tool_call(response)

    result = {
        "name": test_case["name"],
        "category": category,
        "difficulty": test_case["difficulty"],
        "prompt": prompt,
        "response_preview": response[:500],
        "extracted_tool": name,
        "extracted_args": args,
        "made_tool_call": made_call,
    }

    if category == "single_tool":
        # Must call the right tool
        tool_correct = name == expected_tool
        # Check key args if specified
        args_correct = True
        if expected_args:
            for k, v in expected_args.items():
                if isinstance(v, str):
                    args_correct = args_correct and v.lower() in str(args.get(k, "")).lower()
                elif isinstance(v, bool):
                    args_correct = args_correct and args.get(k) == v
        result["passed"] = tool_correct
        result["args_correct"] = args_correct if tool_correct else False
        result["expected_tool"] = expected_tool

    elif category == "arg_format":
        tool_correct = name == expected_tool
        # Check that arguments is valid JSON with required fields
        args_valid = isinstance(args, dict) and len(args) > 0
        result["passed"] = tool_correct and args_valid
        result["args_valid"] = args_valid
        result["expected_tool"] = expected_tool

    elif category == "multi_tool":
        # Should attempt at least one tool call
        result["passed"] = made_call
        result["note"] = "Multi-tool: checked if at least one tool call made"

    elif category == "no_tool":
        # Should NOT make a tool call
        result["passed"] = not made_call
        result["note"] = "Should respond directly without tools"

    elif category == "wrong_tool":
        # Should either not call a tool, or explain it can't do this
        cant_phrases = ["can't", "cannot", "don't have", "no tool", "not available",
                        "unable to", "not able", "sorry", "don't support"]
        has_refusal = any(p in response.lower() for p in cant_phrases)
        result["passed"] = not made_call or has_refusal
        result["note"] = "Should refuse or explain limitation"

    return result


def run_assessment(model_path: str, adapter_path: str = None, max_tokens: int = 512):
    """Run full assessment suite."""
    print(f"Loading model: {model_path}")
    if adapter_path:
        print(f"With adapter: {adapter_path}")

    model, tokenizer = load(model_path, adapter_path=adapter_path,
                            tokenizer_config={"trust_remote_code": True})

    results = []
    categories = {}

    for i, tc in enumerate(TEST_CASES):
        prompt, _, _ = tc["fn"]()

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]

        formatted = tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True, tokenize=False
        )

        t0 = time.time()
        response = generate(
            model, tokenizer, prompt=formatted,
            max_tokens=max_tokens, verbose=False,
        )
        elapsed = time.time() - t0

        result = assess_single(tc, response)
        result["generation_time"] = round(elapsed, 2)
        result["tokens_approx"] = len(response.split())
        results.append(result)

        cat = tc["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]["total"] += 1
        if result["passed"]:
            categories[cat]["passed"] += 1

        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {tc['name']} ({cat}/{tc['difficulty']})")

    # Summary
    total_passed = sum(c["passed"] for c in categories.values())
    total_tests = sum(c["total"] for c in categories.values())

    print(f"\n{'='*60}")
    print(f"Overall: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)")
    print(f"{'='*60}")
    for cat, stats in categories.items():
        pct = 100 * stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat:20s}: {stats['passed']:2d}/{stats['total']:2d} ({pct:.0f}%)")

    return {
        "model": model_path,
        "adapter": adapter_path,
        "total_passed": total_passed,
        "total_tests": total_tests,
        "accuracy": round(total_passed / total_tests, 4),
        "categories": categories,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Tool-calling assessment")
    parser.add_argument("--model", required=True, help="Model path or HF repo")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    results = run_assessment(args.model, args.adapter_path, args.max_tokens)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
