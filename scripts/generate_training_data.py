#!/usr/bin/env python3
"""Generate synthetic tool-calling training data using Qwen 3.6 Plus on OpenRouter.

Generates realistic user<->assistant conversations with tool calls in ChatML format,
for when training-flywheel hasn't captured enough real sessions yet.

Usage:
    python scripts/generate_training_data.py \
        --count 1000 \
        --output data/synthetic_sft.jsonl \
        --tools configs/tool_schemas.json \
        --difficulty mixed

Requires: OPENROUTER_API_KEY env var.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.6-plus-preview:free"

DIFFICULTY_PROMPTS = {
    "easy": "Generate a SIMPLE conversation where the user asks one clear question that requires exactly ONE tool call.",
    "medium": "Generate a MEDIUM complexity conversation where the user's request requires 2-3 tool calls in sequence.",
    "hard": "Generate a COMPLEX conversation with 3+ tool calls, including error recovery or parallel tool use.",
}

SYSTEM_PROMPT = """\
You are a training data generator for a tool-calling AI assistant called Eden.

Generate a realistic multi-turn conversation between a user and Eden. Eden can call tools using <tool_call> and receives results in <tool_result> tags.

RULES:
1. The user message should sound natural — terse, verbose, or anything in between.
2. Eden MUST use <tool_call>{{"name": "TOOL_NAME", "input": {{...}}}}</tool_call> format for tool calls.
3. After each tool call, include a realistic <tool_result>...</tool_result> block.
4. Eden should summarize tool results naturally, not dump raw output.
5. Tool names and parameters MUST match the provided schemas exactly.
6. Include at least one tool call in the conversation.
7. Output ONLY valid JSON: {{"messages": [...]}} where each message has "role" and "content".
8. Roles: "system", "user", "assistant". Tool calls and results go in assistant/system content.

AVAILABLE TOOLS:
{tools_json}

DIFFICULTY: {difficulty}
"""


@dataclass
class Stats:
    total_generated: int = 0
    valid: int = 0
    rejected: int = 0
    quality_scores: list = field(default_factory=list)

    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0


def load_tools(tools_path: str) -> dict:
    with open(tools_path) as f:
        return json.load(f)


def build_prompt(tools: dict, difficulty: str) -> str:
    tools_json = json.dumps(tools["tools"], indent=2)
    diff_instruction = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["medium"])
    return SYSTEM_PROMPT.format(tools_json=tools_json, difficulty=diff_instruction)


def call_openrouter(system_prompt: str, api_key: str) -> str | None:
    """Call OpenRouter API and return the response content."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alex-rentel/eden-models",
        "X-Title": "Eden Training Data Generator",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate one training conversation now. Output ONLY the JSON."},
        ],
        "temperature": 0.9,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None


def extract_json(text: str) -> dict | None:
    """Extract JSON from response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_conversation(conv: dict, tool_names: set) -> bool:
    """Validate a generated conversation meets format requirements."""
    if not isinstance(conv, dict) or "messages" not in conv:
        return False

    messages = conv["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    # Check roles are valid
    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in valid_roles:
            return False
        if "content" not in msg:
            return False

    # Check at least one tool_call tag exists
    has_tool_call = any("<tool_call>" in msg.get("content", "") for msg in messages)
    if not has_tool_call:
        return False

    # Validate tool_call JSON is parseable and function names match schemas
    for msg in messages:
        content = msg.get("content", "")
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        idx = 0
        while True:
            start = content.find(start_tag, idx)
            if start == -1:
                break
            end = content.find(end_tag, start)
            if end == -1:
                return False
            call_json_str = content[start + len(start_tag):end].strip()
            try:
                call_json = json.loads(call_json_str)
            except json.JSONDecodeError:
                return False
            if call_json.get("name") not in tool_names:
                return False
            idx = end + len(end_tag)

    return True


def score_quality(conv: dict) -> float:
    """Score conversation quality 0-1 using heuristics."""
    messages = conv["messages"]
    score = 0.0

    # Multi-turn is better (more messages = more complex)
    num_turns = len([m for m in messages if m["role"] in ("user", "assistant")])
    if num_turns >= 4:
        score += 0.3
    elif num_turns >= 2:
        score += 0.15

    # Count tool calls
    tool_call_count = sum(
        msg.get("content", "").count("<tool_call>")
        for msg in messages
    )
    if tool_call_count >= 3:
        score += 0.3
    elif tool_call_count >= 1:
        score += 0.15

    # Has tool results
    has_results = any("<tool_result>" in msg.get("content", "") for msg in messages)
    if has_results:
        score += 0.2

    # Assistant summarizes (doesn't just dump tool output)
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if assistant_msgs:
        last_assistant = assistant_msgs[-1]["content"]
        # If last assistant message has text outside of tags, it's summarizing
        stripped = last_assistant
        for tag in ["<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>"]:
            stripped = stripped.replace(tag, "")
        if len(stripped.strip()) > 20:
            score += 0.2

    return min(score, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tool-calling training data")
    parser.add_argument("--count", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_sft.jsonl", help="Output JSONL path")
    parser.add_argument("--tools", type=str, default="configs/tool_schemas.json", help="Tool schemas JSON")
    parser.add_argument("--difficulty", type=str, default="mixed", choices=["easy", "medium", "hard", "mixed"],
                        help="Difficulty level")
    parser.add_argument("--min-quality", type=float, default=0.3, help="Minimum quality score to keep")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    tools = load_tools(args.tools)
    tool_names = {t["name"] for t in tools["tools"]}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    difficulties = ["easy", "medium", "hard"]

    print(f"Generating {args.count} conversations → {args.output}")
    print(f"Model: {MODEL}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Min quality: {args.min_quality}")
    print()

    with open(output_path, "w") as f:
        while stats.valid < args.count:
            # Pick difficulty
            if args.difficulty == "mixed":
                diff = random.choice(difficulties)
            else:
                diff = args.difficulty

            prompt = build_prompt(tools, diff)
            raw = call_openrouter(prompt, api_key)
            stats.total_generated += 1

            if raw is None:
                stats.rejected += 1
                continue

            conv = extract_json(raw)
            if conv is None:
                stats.rejected += 1
                continue

            if not validate_conversation(conv, tool_names):
                stats.rejected += 1
                continue

            quality = score_quality(conv)
            if quality < args.min_quality:
                stats.rejected += 1
                continue

            stats.valid += 1
            stats.quality_scores.append(quality)
            f.write(json.dumps(conv) + "\n")

            if stats.valid % 10 == 0:
                print(f"  [{stats.valid}/{args.count}] generated={stats.total_generated} "
                      f"rejected={stats.rejected} avg_quality={stats.avg_quality:.2f}")

            # Rate limiting for free tier
            time.sleep(0.5)

    print()
    print("=== Generation Complete ===")
    print(f"Total API calls:    {stats.total_generated}")
    print(f"Valid conversations: {stats.valid}")
    print(f"Rejected:           {stats.rejected}")
    print(f"Avg quality score:  {stats.avg_quality:.2f}")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    main()
