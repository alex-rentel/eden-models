#!/usr/bin/env python3
"""Rebalance training data: 80% tool-calling examples, 20% no-tool examples.

The v1 model learned to refuse tool calls because the original glaive dataset
had ~50% refusal/no-tool examples. This script filters for tool-calling examples
and adds a small fraction of no-tool examples for balance.
"""

import json
import random
from pathlib import Path


def main():
    data_dir = Path(__file__).parent.parent

    # Load all training data
    train_data = []
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            train_data.append(json.loads(line))

    valid_data = []
    with open(data_dir / "valid.jsonl") as f:
        for line in f:
            valid_data.append(json.loads(line))

    # Split by whether they have tool_calls
    def has_tool_calls(item):
        return any(m.get("tool_calls") for m in item["messages"])

    def has_tools_defined(item):
        return bool(item.get("tools"))

    train_with_calls = [x for x in train_data if has_tool_calls(x)]
    train_with_tools_no_call = [x for x in train_data if has_tools_defined(x) and not has_tool_calls(x)]
    train_no_tools = [x for x in train_data if not has_tools_defined(x)]

    print(f"Original train split:")
    print(f"  With tool_calls: {len(train_with_calls)}")
    print(f"  With tools but no calls (refusals): {len(train_with_tools_no_call)}")
    print(f"  No tools defined: {len(train_no_tools)}")

    # New composition: 80% tool calling, 10% refusal, 10% no-tool
    random.seed(42)
    n_tool_calls = len(train_with_calls)  # Use all tool-calling examples
    n_refusals = n_tool_calls // 8  # ~12.5%
    n_no_tools = n_tool_calls // 8  # ~12.5%

    random.shuffle(train_with_tools_no_call)
    random.shuffle(train_no_tools)

    balanced_train = (
        train_with_calls +
        train_with_tools_no_call[:n_refusals] +
        train_no_tools[:n_no_tools]
    )
    random.shuffle(balanced_train)

    # Same for validation
    valid_with_calls = [x for x in valid_data if has_tool_calls(x)]
    valid_other = [x for x in valid_data if not has_tool_calls(x)]
    random.shuffle(valid_other)
    n_valid_other = len(valid_with_calls) // 4
    balanced_valid = valid_with_calls + valid_other[:n_valid_other]
    random.shuffle(balanced_valid)

    print(f"\nRebalanced train: {len(balanced_train)}")
    print(f"  Tool-calling: {n_tool_calls} ({100*n_tool_calls/len(balanced_train):.0f}%)")
    print(f"  Refusals: {min(n_refusals, len(train_with_tools_no_call))}")
    print(f"  No-tool: {min(n_no_tools, len(train_no_tools))}")
    print(f"Rebalanced valid: {len(balanced_valid)}")

    # Write to v2 files
    for name, data in [("train", balanced_train), ("valid", balanced_valid)]:
        path = data_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} to {path}")


if __name__ == "__main__":
    main()
