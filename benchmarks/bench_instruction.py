#!/usr/bin/env python3
"""Instruction Following Benchmark — format constraints, rule adherence, precision."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "Instruction Following",
    "bench_instruction",
    "Tests ability to follow precise formatting rules, length constraints, and complex instructions.",
)


@suite.test("Exact sentence count", "Easy")
def _():
    r = query([{"role": "user", "content":
        "Describe machine learning in EXACTLY 3 sentences. Not 2, not 4 — exactly 3."}])
    content = r.get("content", "").strip()
    # Count sentence-ending punctuation
    sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    passed = len(sentences) == 3
    return r, passed, f"Expected 3 sentences, got {len(sentences)}"


@suite.test("Bullet point count", "Easy")
def _():
    r = query([{"role": "user", "content":
        "Give me exactly 5 tips for better sleep. Use a numbered list (1-5). No intro or outro text."}])
    content = r.get("content", "").strip()
    numbered = [l for l in content.split("\n") if l.strip() and l.strip()[0].isdigit()]
    passed = len(numbered) == 5
    return r, passed, f"Expected 5 numbered items, got {len(numbered)}"


@suite.test("Word avoidance", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Explain what a computer does without using the words 'computer', 'machine', 'device', or 'technology'."}])
    content = r.get("content", "").lower()
    forbidden = ["computer", "machine", "device", "technology"]
    violations = [w for w in forbidden if w in content]
    passed = len(violations) == 0
    return r, passed, f"Forbidden words found: {violations}" if violations else "No forbidden words"


@suite.test("Response in specific format", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Give me 3 book recommendations. Format EACH as:\n"
        "TITLE: [title]\nAUTHOR: [author]\nWHY: [one sentence]\n\n"
        "Follow this format exactly. No other text."}])
    content = r.get("content", "")
    title_count = content.count("TITLE:")
    author_count = content.count("AUTHOR:")
    why_count = content.count("WHY:")
    passed = title_count == 3 and author_count == 3 and why_count == 3
    return r, passed, f"Expected 3 each of TITLE/AUTHOR/WHY, got {title_count}/{author_count}/{why_count}"


@suite.test("First letter constraint (acrostic)", "Hard")
def _():
    r = query([{"role": "user", "content":
        "Write 5 sentences where the first letter of each sentence spells 'HELLO'. "
        "The sentences should be about space exploration."}])
    content = r.get("content", "").strip()
    sentences = [s.strip() for s in content.split("\n") if s.strip() and len(s.strip()) > 5]
    if len(sentences) < 5:
        sentences = [s.strip() for s in content.replace(". ", ".\n").split("\n") if s.strip() and len(s.strip()) > 5]
    first_letters = "".join(s[0].upper() for s in sentences[:5]) if len(sentences) >= 5 else ""
    passed = first_letters == "HELLO"
    return r, passed, f"Expected first letters 'HELLO', got '{first_letters}'"


@suite.test("JSON-only response", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Return a JSON object with keys 'animal', 'color', 'count' for: 'Three blue whales'. "
        "Return ONLY valid JSON. No markdown, no explanation, no code fences."}])
    content = r.get("content", "").strip()
    # Strip fences if present despite instructions
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content[:-3].strip()
    try:
        parsed = json.loads(content)
        passed = (parsed.get("animal", "").lower() in ["whale", "whales", "blue whale", "blue whales"] and
                  parsed.get("color", "").lower() == "blue" and
                  parsed.get("count") == 3)
    except json.JSONDecodeError:
        passed = False
    return r, passed, "Must return valid JSON with correct values, no wrapping"


@suite.test("Multi-constraint response", "Hard")
def _():
    r = query([{"role": "user", "content":
        "Write a paragraph about coffee that:\n"
        "1. Is exactly 4 sentences long\n"
        "2. Mentions at least 2 countries\n"
        "3. Includes a number/statistic\n"
        "4. Starts with the word 'Every'\n"
        "5. Ends with a question"}])
    content = r.get("content", "").strip()
    sentences = [s.strip() for s in content.replace("!", ".").replace("?", "?|").split("|") if s.strip()]
    # Flatten further
    all_sents = []
    for s in sentences:
        all_sents.extend([x.strip() for x in s.split(".") if x.strip()])
    
    starts_every = content.startswith("Every")
    ends_question = content.rstrip().endswith("?")
    has_number = any(c.isdigit() for c in content)
    
    passed = starts_every and ends_question and has_number
    return r, passed, f"Starts with 'Every': {starts_every}, ends with '?': {ends_question}, has number: {has_number}"


if __name__ == "__main__":
    suite.run()
