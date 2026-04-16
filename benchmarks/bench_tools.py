#!/usr/bin/env python3
"""Tool Calling Benchmark — tool selection, argument formatting, multi-tool, multi-step."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "Tool Calling",
    "bench_tools",
    "Tests tool selection, argument generation, multi-step planning, and tool avoidance.",
)

# ─── Tool definitions ─────────────────────────────────────

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
        "description": "Write content to a file",
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
]


def get_tool_call(result):
    """Extract first tool call name and parsed args."""
    calls = result.get("tool_calls", [])
    if not calls:
        return None, {}
    tc = calls[0]
    name = tc.get("function", {}).get("name", "")
    try:
        args = json.loads(tc["function"]["arguments"])
    except (json.JSONDecodeError, KeyError):
        args = {}
    return name, args


@suite.test("Single tool: list directory", "Easy")
def _():
    r = query([{"role": "user", "content": "List the files in /home/user/projects"}], tools=TOOLS)
    name, args = get_tool_call(r)
    passed = name == "list_files" and "/home/user/projects" in args.get("path", "")
    r["tool_calls_summary"] = json.dumps(r.get("tool_calls", []))[:500]
    return r, passed, "Should call list_files with correct path"


@suite.test("Single tool: read specific file", "Easy")
def _():
    r = query([{"role": "user", "content": "Show me the contents of /etc/hostname"}], tools=TOOLS)
    name, args = get_tool_call(r)
    passed = name == "read_file" and args.get("path") == "/etc/hostname"
    r["tool_calls_summary"] = json.dumps(r.get("tool_calls", []))[:500]
    return r, passed, "Should call read_file with /etc/hostname"


@suite.test("Tool selection: weather (not file tools)", "Medium")
def _():
    r = query([{"role": "user", "content": "What's the weather like in Tokyo right now?"}], tools=TOOLS)
    name, args = get_tool_call(r)
    passed = name == "get_weather" and "tokyo" in args.get("city", "").lower()
    r["tool_calls_summary"] = json.dumps(r.get("tool_calls", []))[:500]
    return r, passed, "Should call get_weather, not search_web or file tools"


@suite.test("Tool selection: web search for current info", "Medium")
def _():
    r = query([{"role": "user", "content": "Who won the NBA finals last year?"}], tools=TOOLS)
    name, args = get_tool_call(r)
    passed = name == "search_web" and "query" in args
    r["tool_calls_summary"] = json.dumps(r.get("tool_calls", []))[:500]
    return r, passed, "Should call search_web for current events"


@suite.test("Tool avoidance: simple question", "Medium")
def _():
    r = query([{"role": "user", "content": "What is the capital of France?"}], tools=TOOLS)
    calls = r.get("tool_calls", [])
    content = r.get("content", "").lower()
    passed = len(calls) == 0 and "paris" in content
    return r, passed, "Should answer directly without tools"


@suite.test("Tool avoidance: math question", "Medium")
def _():
    r = query([{"role": "user", "content": "What is 15 * 23?"}], tools=TOOLS)
    calls = r.get("tool_calls", [])
    content = r.get("content", "")
    passed = len(calls) == 0 and "345" in content
    return r, passed, "Should compute 345 without calling any tools"


@suite.test("Write file with specific content", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Create a file at /home/user/hello.txt containing 'Hello, World!'"}], tools=TOOLS)
    name, args = get_tool_call(r)
    passed = (name == "write_file" and 
              "/home/user/hello.txt" in args.get("path", "") and
              "Hello, World!" in args.get("content", ""))
    r["tool_calls_summary"] = json.dumps(r.get("tool_calls", []))[:500]
    return r, passed, "Should call write_file with correct path and content"


@suite.test("Multi-step: find and read", "Hard")
def _():
    r = query([{"role": "user", "content":
        "Find all Python files in /home/user/projects and then read the README.md file."}], tools=TOOLS)
    calls = r.get("tool_calls", [])
    # Should call at least list_files or run_command first
    first_name = calls[0].get("function", {}).get("name", "") if calls else ""
    passed = len(calls) >= 1 and first_name in ("list_files", "run_command")
    r["tool_calls_summary"] = json.dumps(calls)[:500]
    return r, passed, "Should start with list_files or run_command"


if __name__ == "__main__":
    suite.run()
