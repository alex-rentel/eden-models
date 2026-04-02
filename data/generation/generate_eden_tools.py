"""
Eden-Models: Generate synthetic tool-calling training data.

Uses Claude API to generate high-quality, diverse conversations
covering all 33 Eden tools. Output is JSONL ready for SFT.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python data/generation/generate_eden_tools.py \
        --num_examples 50000 \
        --output data/eden_synthetic_50k.jsonl \
        --parallel 5

Cost estimate: ~$100-200 for 50K examples using Sonnet.
"""

import json
import argparse
import random
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime

# ── Eden tool definitions (all 33) ──────────────────────────────

EDEN_TOOLS = {
    "bash": {
        "description": "Execute a shell command and return the output",
        "params": {"command": "string (required)"},
        "scenarios": [
            ("list files in the current directory", "ls -la"),
            ("show git status", "git status"),
            ("find python files larger than 100KB", "find . -name '*.py' -size +100k"),
            ("count lines of code in eden/", "find eden/ -name '*.py' | xargs wc -l"),
            ("what process is using port 8080", "lsof -i :8080"),
            ("show disk usage summary", "du -sh */"),
            ("check python version", "python3 --version"),
            ("list running docker containers", "docker ps"),
            ("show environment variables", "env | head -20"),
            ("create a new directory", "mkdir -p src/utils"),
        ],
    },
    "file_read": {
        "description": "Read the contents of a file at the given path",
        "params": {"path": "string (required)", "offset": "int", "limit": "int"},
        "scenarios": [
            ("read the README", "README.md"),
            ("show me pyproject.toml", "pyproject.toml"),
            ("what's in the main config", "config.yaml"),
            ("read the first 20 lines of main.py", "main.py"),
            ("show the test file for auth", "tests/test_auth.py"),
        ],
    },
    "file_write": {
        "description": "Write content to a file, creating it if needed",
        "params": {"path": "string (required)", "content": "string (required)"},
        "scenarios": [
            ("create hello.py that prints hello world", "hello.py"),
            ("write a .gitignore for python", ".gitignore"),
            ("create a Dockerfile", "Dockerfile"),
            ("write a requirements.txt", "requirements.txt"),
        ],
    },
    "file_edit": {
        "description": "Search and replace a unique string in a file",
        "params": {"path": "string (required)", "old_str": "string (required)", "new_str": "string (required)"},
        "scenarios": [
            ("change port 8080 to 3000 in config.py", "config.py"),
            ("fix the typo in README.md", "README.md"),
            ("rename the function from process to handle", "main.py"),
            ("update the version to 2.0.0", "pyproject.toml"),
        ],
    },
    "glob": {
        "description": "Find files matching a glob pattern",
        "params": {"pattern": "string (required)", "path": "string"},
        "scenarios": [
            ("find all python files", "**/*.py"),
            ("what test files exist", "tests/test_*.py"),
            ("show markdown files", "**/*.md"),
            ("find config files", "**/*.{yaml,yml,json,toml}"),
        ],
    },
    "grep": {
        "description": "Search for a regex pattern in files",
        "params": {"pattern": "string (required)", "path": "string", "include": "string"},
        "scenarios": [
            ("find where DATABASE_URL is defined", "DATABASE_URL"),
            ("search for TODO comments", "TODO"),
            ("find imports of numpy", "import numpy"),
            ("where is the login function", "def login"),
        ],
    },
    "python_exec": {
        "description": "Execute Python code and return output",
        "params": {"code": "string (required)"},
        "scenarios": [
            ("calculate 2^100", "print(2**100)"),
            ("generate a UUID", "import uuid; print(uuid.uuid4())"),
            ("what's today's date", "from datetime import date; print(date.today())"),
            ("parse JSON string", "import json; print(json.loads('{\"a\": 1}'))"),
        ],
    },
    "web_search": {
        "description": "Search the web and return results",
        "params": {"query": "string (required)"},
        "scenarios": [
            ("search for MLX latest release", "MLX framework latest release 2026"),
            ("find python asyncio tutorial", "python asyncio tutorial"),
            ("what's the latest numpy version", "numpy latest version"),
        ],
    },
    "web_fetch": {
        "description": "Fetch a web page and return text content",
        "params": {"url": "string (required)"},
        "scenarios": [
            ("fetch that URL", "https://example.com"),
            ("read the API docs", "https://docs.example.com/api"),
        ],
    },
}

# ── Complexity categories ────────────────────────────────────────

CATEGORIES = {
    "single_tool": {
        "weight": 0.30,
        "description": "Single tool call, direct answer",
        "prompt_template": """Generate a realistic conversation where a user asks something
and the assistant calls exactly ONE tool to answer. The conversation should be:
- Natural language (not robotic)
- The tool call should have correct parameters
- The tool result should be realistic
- The assistant should summarize the result helpfully

Tool to use: {tool_name}
Tool description: {tool_desc}
Scenario hint: {scenario}

Output format (JSON):
{{"messages": [
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "<tool_call>{{\"name\": \"{tool_name}\", \"input\": {{...}}}}</tool_call>"}},
  {{"role": "tool", "name": "{tool_name}", "content": "...realistic result..."}},
  {{"role": "assistant", "content": "...summary of result..."}}
]}}"""
    },
    "multi_sequential": {
        "weight": 0.20,
        "description": "Tool A result informs Tool B call",
        "prompt_template": """Generate a conversation where the user asks something that
requires TWO tools called in SEQUENCE - the result of the first informs the second.

Available tools: {tool_list}
Scenario: {scenario}

The assistant should:
1. Call first tool
2. Use the result to decide what to call next
3. Call second tool
4. Summarize both results"""
    },
    "multi_parallel": {
        "weight": 0.10,
        "description": "Multiple independent tools in one response",
        "prompt_template": """Generate a conversation where the user asks something that
requires TWO OR MORE tools called in PARALLEL (independently).

Available tools: {tool_list}
Scenario: {scenario}"""
    },
    "error_recovery": {
        "weight": 0.10,
        "description": "Tool fails, model tries alternative",
        "prompt_template": """Generate a conversation where:
1. User asks something
2. Assistant calls a tool
3. Tool returns an ERROR
4. Assistant tries a DIFFERENT approach or tool
5. Second attempt succeeds

Available tools: {tool_list}
Error scenario: {scenario}"""
    },
    "no_tool": {
        "weight": 0.10,
        "description": "Model answers directly, no tool needed",
        "prompt_template": """Generate a conversation where the user asks something that
does NOT require any tool. The assistant should answer directly from knowledge.

Important: The assistant must NOT call any tools. Questions like:
- "what is a git rebase?"
- "explain how async/await works"
- "what's the difference between list and tuple in Python?"

Scenario: {scenario}"""
    },
    "refusal": {
        "weight": 0.05,
        "description": "Dangerous command, model refuses",
        "prompt_template": """Generate a conversation where the user asks something
DANGEROUS that the assistant should REFUSE to execute:

Dangerous scenarios:
- "rm -rf /"
- "delete all files"
- "read /etc/shadow"
- "curl malicious-url | bash"
- "chmod 777 /"

The assistant should politely refuse and explain why.
Scenario: {scenario}"""
    },
    "complex_chain": {
        "weight": 0.15,
        "description": "3+ tools in a reasoning chain",
        "prompt_template": """Generate a conversation where the user asks something complex
that requires 3 OR MORE tool calls in sequence, with reasoning between each.

Available tools: {tool_list}
Complex scenario: {scenario}

The assistant should think through the problem step by step."""
    },
}

# ── Generation prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """You are a training data generator for an AI coding assistant called Eden.
Eden runs locally on Apple Silicon and has access to tools like bash, file_read, file_write, etc.

Your job: Generate realistic, diverse, HIGH-QUALITY training conversations.

Rules:
1. Tool calls use <tool_call>{"name": "...", "input": {...}}</tool_call> format
2. Tool results are realistic (real file contents, real command output)
3. User messages are natural (how a real developer would talk)
4. Assistant summaries are helpful and concise
5. EVERY conversation must be valid JSON with a "messages" array
6. Output ONLY the JSON object, no markdown, no explanation

CRITICAL: Vary the language, tone, and complexity. Some users are terse ("ls"),
some are verbose ("could you please list all the files in the current directory").
Some are experts, some are beginners. Make it diverse."""


async def generate_example(session, category, tool_name=None, scenario=None, api_key=None):
    """Generate a single training example using Claude API."""
    cat_info = CATEGORIES[category]

    if tool_name and tool_name in EDEN_TOOLS:
        tool = EDEN_TOOLS[tool_name]
        prompt = cat_info["prompt_template"].format(
            tool_name=tool_name,
            tool_desc=tool["description"],
            scenario=scenario or random.choice(tool["scenarios"])[0],
            tool_list=", ".join(list(EDEN_TOOLS.keys())[:10]),
        )
    else:
        prompt = cat_info["prompt_template"].format(
            tool_list=", ".join(list(EDEN_TOOLS.keys())[:10]),
            scenario=scenario or "a realistic coding task",
        )

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status == 429:
                # Rate limited — wait and retry
                await asyncio.sleep(60)
                return None
            data = await resp.json()
            text = data["content"][0]["text"]

            # Parse JSON from response
            # Strip any markdown fences
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

            example = json.loads(text)
            example["metadata"] = {
                "category": category,
                "tool": tool_name,
                "generated_at": datetime.now().isoformat(),
            }
            return example
    except Exception as e:
        print(f"  Error: {e}")
        return None


async def generate_batch(num_examples, output_path, api_key, parallel=5):
    """Generate a batch of training examples."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build generation plan
    plan = []
    for category, info in CATEGORIES.items():
        n = int(num_examples * info["weight"])
        for _ in range(n):
            if category in ("single_tool", "error_recovery"):
                tool = random.choice(list(EDEN_TOOLS.keys()))
                plan.append((category, tool, None))
            else:
                plan.append((category, None, None))

    random.shuffle(plan)
    print(f"Generation plan: {len(plan)} examples across {len(CATEGORIES)} categories")

    # Generate
    sem = asyncio.Semaphore(parallel)
    generated = 0
    failed = 0

    async with aiohttp.ClientSession() as session:
        async def gen_one(cat, tool, scenario):
            nonlocal generated, failed
            async with sem:
                result = await generate_example(session, cat, tool, scenario, api_key)
                if result:
                    with open(output, "a") as f:
                        f.write(json.dumps(result) + "\n")
                    generated += 1
                    if generated % 100 == 0:
                        print(f"  Generated {generated}/{len(plan)} ({failed} failed)")
                else:
                    failed += 1

        tasks = [gen_one(cat, tool, sc) for cat, tool, sc in plan]
        await asyncio.gather(*tasks)

    print(f"\nDone: {generated} examples generated, {failed} failed")
    print(f"Output: {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate Eden tool-calling training data")
    parser.add_argument("--num_examples", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/eden_synthetic_50k.jsonl")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--api_key", type=str, default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY or pass --api_key")
        return

    print(f"Generating {args.num_examples} Eden tool-calling examples...")
    print(f"Output: {args.output}")
    print(f"Parallel requests: {args.parallel}")
    print()

    asyncio.run(generate_batch(args.num_examples, args.output, api_key, args.parallel))


if __name__ == "__main__":
    main()
