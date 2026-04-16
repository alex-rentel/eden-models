# Bonsai 4B Tool Calling Investigation

**Date:** 2026-04-02  
**Issue:** Bonsai 4B scored 0/8 on tool calling benchmarks while 8B scored 8/8 and 1.7B scored 6/8.

## Root Cause

**The Bonsai 4B model does not emit the `<tool_call>` opening tag (token 151657) before tool call JSON.** It generates the JSON body and the closing `</tool_call>` tag correctly, but skips the opening tag. This is a training/fine-tuning defect specific to the 4B model.

The MLX server's tool call parser uses a **per-token state machine** (server.py line 1426) that triggers on `gen.text == "<tool_call>"`. Without the opening tag token, the state machine never enters tool-call mode, and the entire output is treated as plain text content.

## Evidence

### Token-level comparison

All three models share identical tokenizers with `<tool_call>` as special token ID 151657.

**Bonsai 8B output tokens** (correct):
```
151657: '<tool_call>'       ← opening tag present
   198: '\n'
  4913: '{"'
   606: 'name'
   ...
 95642: '"}}\n'
151658: '</tool_call>'
```

**Bonsai 1.7B output tokens** (correct):
```
151657: '<tool_call>'       ← opening tag present
   198: '\n'
  4913: '{"'
   606: 'name'
   ...
```

**Bonsai 4B output tokens** (broken):
```
  4913: '{"'                ← jumps straight to JSON, no opening tag
   606: 'name'
   788: '":'
   ...
 95642: '"}}\n'
151658: '</tool_call>'      ← closing tag IS present
```

### Raw API responses

**4B (broken):** `finish_reason: "stop"`, tool_calls: `[]`, content contains raw JSON:
```json
{
  "message": {
    "role": "assistant",
    "content": "{\"name\": \"list_files\", \"arguments\": {\"path\": \"/home/user/projects\"}}\n</tool_call>",
    "tool_calls": []
  }
}
```

**8B (correct):** `finish_reason: "tool_calls"`, content is empty, tool_calls populated:
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "list_files",
          "arguments": "{\"path\": \"/home/user/projects\"}"
        },
        "type": "function",
        "id": "b1af029f-..."
      }
    ]
  }
}
```

### Ruled out causes

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Different chat templates | ❌ Ruled out | Both models use identical Jinja2 chat templates |
| Different tokenizers | ❌ Ruled out | Both have `<tool_call>` as special token 151657, same vocab |
| Server parsing bug | ❌ Ruled out | Server works correctly for 8B and 1.7B |
| **Missing opening tag in 4B generation** | ✅ **Confirmed** | Token-level analysis shows 4B never generates token 151657 |

## Diagnosis

This is a **model training defect** in `prism-ml/Bonsai-4B-mlx-1bit`. During instruction tuning or RLHF, the 4B model learned to generate tool call JSON and the closing `</tool_call>` tag, but the opening `<tool_call>` tag was either:
- Underrepresented in the 4B training data
- Lost during quantization to 1-bit (less likely, since the closing tag works fine)
- A bug in the fine-tuning pipeline for the 4B variant specifically

The model is **functionally capable** of tool calling — it selects the correct tool and generates valid arguments in all 8 tests. It just doesn't wrap them in the expected tags.

## Impact on benchmark scores

The 4B model actually got the "right answer" on 6/8 tool calling tests (it failed the 2 tool-avoidance tests where it should NOT have called a tool). But because the server can't parse the tool calls, all 8 tests show as failures in the benchmark.

## Possible workarounds

1. **Server-side fix:** Modify the MLX server to also detect tool call JSON in the content field (e.g., regex for `{"name": "...", "arguments": ...}\n</tool_call>`)
2. **Benchmark-side fix:** Add fallback parsing in `bench_tools.py` to check the content field for tool call JSON when tool_calls is empty
3. **Report upstream:** File an issue on PrismML's repo for the 4B model's missing `<tool_call>` tag

None of these were applied — the benchmark results reflect the model as-shipped.
