#!/usr/bin/env python3
"""Convert glaive-function-calling-v2 to mlx_lm.lora ChatDataset format.

Output format (per JSONL line):
  {"messages": [...], "tools": [...]}

The tools key is passed to tokenizer.apply_chat_template(messages, tools=tools),
so we use OpenAI-style tool definitions that Qwen3's tokenizer understands.
"""

import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset


def fix_single_quoted_json(text: str) -> str:
    """Fix JSON where string values use single quotes: '{...}' -> "..." """
    return re.sub(r"'(\{.*?\})'", lambda m: json.dumps(m.group(1)), text)


def extract_json_objects(text: str) -> list[dict]:
    """Extract all top-level JSON objects from text using brace counting."""
    # Pre-fix single-quoted JSON strings (common in glaive data)
    text = fix_single_quoted_json(text)

    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape = False
            j = i
            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                elif c == '\\' and in_string:
                    escape = True
                elif c == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(text[start:j+1])
                                objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
            else:
                pass  # Unclosed brace, skip
        i += 1
    return objects


def parse_system_tools(system_text: str) -> list[dict]:
    """Extract tool definitions from the SYSTEM field."""
    text = system_text.strip()
    if text.startswith("SYSTEM:"):
        text = text[len("SYSTEM:"):].strip()

    tools = []
    objects = extract_json_objects(text)

    for obj in objects:
        if "name" in obj and ("parameters" in obj or "description" in obj):
            tools.append({
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "description": obj.get("description", ""),
                    "parameters": obj.get("parameters", {"type": "object", "properties": {}}),
                }
            })

    return tools


def parse_chat(chat_text: str) -> list[dict]:
    """Parse the chat field into messages list."""
    messages = []

    # Split on role prefixes
    parts = re.split(r'\n{2,}(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat_text.strip())

    for part in parts:
        part = part.strip().replace("<|endoftext|>", "").strip()
        if not part:
            continue

        if part.startswith("USER:"):
            content = part[len("USER:"):].strip()
            if content:
                messages.append({"role": "user", "content": content})

        elif part.startswith("ASSISTANT:"):
            content = part[len("ASSISTANT:"):].strip()
            if not content:
                continue

            # Check for function call
            fc_match = re.search(r'<functioncall>\s*', content)
            if fc_match:
                pre_text = content[:fc_match.start()].strip()
                fc_json_text = content[fc_match.end():].strip()

                # Parse the function call JSON
                fc_objects = extract_json_objects(fc_json_text)
                if fc_objects:
                    fc_data = fc_objects[0]
                    name = fc_data.get("name", "")
                    args_raw = fc_data.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {"raw": args_raw}
                    else:
                        args = args_raw

                    msg = {
                        "role": "assistant",
                        "content": pre_text if pre_text else "",
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            }
                        }]
                    }
                    messages.append(msg)
                else:
                    messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})

        elif part.startswith("FUNCTION RESPONSE:"):
            content = part[len("FUNCTION RESPONSE:"):].strip()
            # Find the tool name from previous assistant tool_call
            tool_name = ""
            for msg in reversed(messages):
                if msg.get("tool_calls"):
                    tool_name = msg["tool_calls"][0]["function"]["name"]
                    break
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": content,
            })

    return messages


def convert_example(example: dict) -> dict | None:
    """Convert a single glaive example to mlx_lm.lora format."""
    system_text = example.get("system", "")
    chat_text = example.get("chat", "")

    if not chat_text.strip():
        return None

    tools = parse_system_tools(system_text)
    messages = parse_chat(chat_text)

    if not messages:
        return None

    # Add system message
    messages.insert(0, {
        "role": "system",
        "content": "You are a helpful assistant with access to tools. Use them when needed to answer user questions accurately."
    })

    # Validate: must have at least system + user + assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    result = {"messages": messages}
    if tools:
        result["tools"] = tools

    return result


def main():
    print("Loading glaive-function-calling-v2...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    print(f"Loaded {len(ds)} examples")

    converted = []
    skipped = 0
    has_tools = 0
    has_tool_calls = 0

    for ex in ds:
        result = convert_example(ex)
        if result is None:
            skipped += 1
            continue

        if "tools" in result:
            has_tools += 1

        if any(m.get("tool_calls") for m in result["messages"]):
            has_tool_calls += 1

        converted.append(result)

    print(f"\nConversion stats:")
    print(f"  Total input:        {len(ds)}")
    print(f"  Converted:          {len(converted)}")
    print(f"  Skipped:            {skipped}")
    print(f"  With tools defined: {has_tools}")
    print(f"  With tool_calls:    {has_tool_calls}")
    print(f"  Direct response:    {len(converted) - has_tool_calls}")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(converted)

    # Split 90/10
    split_idx = int(len(converted) * 0.9)
    train_data = converted[:split_idx]
    valid_data = converted[split_idx:]

    # Write output
    out_dir = Path(__file__).parent.parent

    for name, data in [("train", train_data), ("valid", valid_data)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    # Token length estimation
    total_chars = sum(
        sum(len(m.get("content", "")) for m in item["messages"])
        for item in converted
    )
    avg_chars = total_chars / len(converted)
    print(f"\nAvg message chars per example: {avg_chars:.0f} (~{avg_chars/4:.0f} tokens)")

    # Print a sample with tool_calls
    for item in train_data:
        if any(m.get("tool_calls") for m in item["messages"]):
            print("\n--- Sample with tool_calls ---")
            print(json.dumps(item, indent=2, ensure_ascii=False)[:1500])
            break


if __name__ == "__main__":
    main()
