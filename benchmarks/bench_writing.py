#!/usr/bin/env python3
"""Writing Benchmark — creative writing, summarization, tone control, compression."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "Writing",
    "bench_writing",
    "Tests creative writing, summarization, tone adaptation, and compression.",
)


@suite.test("Product description", "Easy")
def _():
    r = query([{"role": "user", "content":
        "Write a 2-3 sentence product description for a wireless mechanical keyboard "
        "targeted at programmers. Highlight: hot-swappable switches, split design, "
        "programmable layers. Keep it punchy and professional."}])
    content = r.get("content", "").lower()
    passed = all(w in content for w in ["switch", "split", "layer"]) and len(content) < 600
    return r, passed, "Must mention all 3 features, under 600 chars"


@suite.test("Summarize to exact length", "Medium")
def _():
    passage = (
        "The James Webb Space Telescope (JWST) has transformed our understanding of the early universe "
        "since its deployment at the L2 Lagrange point in 2022. Its 6.5-meter gold-coated primary mirror "
        "and suite of infrared instruments have revealed galaxies forming just 300 million years after "
        "the Big Bang, far earlier than previously thought possible. The telescope has also provided "
        "unprecedented views of exoplanet atmospheres, detecting water vapor, carbon dioxide, and "
        "even sulfur dioxide in the atmospheres of gas giants orbiting distant stars. Closer to home, "
        "JWST has captured detailed images of planets in our own solar system, including seasonal "
        "changes on Neptune and new ring structures around Jupiter. The $10 billion mission, a joint "
        "project between NASA, ESA, and CSA, is expected to operate for at least 20 years, far "
        "exceeding its original 5-10 year design life."
    )
    r = query([{"role": "user", "content":
        f"Summarize this passage in EXACTLY 3 sentences. Not 2, not 4 — exactly 3.\n\n{passage}"}])
    content = r.get("content", "").strip()
    # Count sentences (rough: split on . ! ?)
    sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    passed = len(sentences) == 3
    return r, passed, f"Expected exactly 3 sentences, got {len(sentences)}"


@suite.test("Compress to word limit", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Explain how a CPU works in under 50 words. Count carefully."}])
    content = r.get("content", "").strip()
    words = len(content.split())
    passed = words <= 55  # small buffer for counting differences
    return r, passed, f"Expected ≤50 words, got {words}"


@suite.test("Tone: formal to casual rewrite", "Medium")
def _():
    formal = (
        "We wish to inform you that the quarterly financial review has been rescheduled "
        "to the subsequent Monday due to unforeseen circumstances. Your attendance at the "
        "revised meeting time would be greatly appreciated. Please confirm your availability "
        "at your earliest convenience."
    )
    r = query([{"role": "user", "content":
        f"Rewrite this in a casual Slack message tone. Keep it short.\n\n{formal}"}])
    content = r.get("content", "").lower()
    # Casual indicators
    casual_signals = ["hey", "heads up", "pushed", "moved", "lmk", "let me know", "!", "fyi", "btw"]
    passed = any(s in content for s in casual_signals) and len(content) < len(formal)
    return r, passed, "Should be shorter, more casual, with informal language"


@suite.test("Tone: technical to 5-year-old", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Explain DNS (Domain Name System) to a 5-year-old. Use an analogy. Keep it under 100 words."}])
    content = r.get("content", "").lower()
    words = len(content.split())
    # Should have simple language and an analogy
    has_analogy = any(w in content for w in ["like", "imagine", "pretend", "think of", "phone book", 
                                              "address book", "ask", "remember"])
    passed = words <= 110 and has_analogy
    return r, passed, f"Must use analogy, ≤100 words (got {words})"


@suite.test("Creative: story with constraints", "Hard")
def _():
    r = query([{"role": "user", "content":
        "Write a complete flash fiction story (under 200 words) that:\n"
        "1. Is set on Mars\n"
        "2. Features exactly 2 characters\n"
        "3. Contains a plot twist in the last sentence\n"
        "4. Uses present tense throughout"}])
    content = r.get("content", "")
    words = len(content.split())
    has_mars = "mars" in content.lower()
    passed = words <= 220 and has_mars
    return r, passed, f"Must be set on Mars, under 200 words (got {words})"


@suite.test("Email: professional with constraints", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Write a professional email declining a job offer. You're choosing a different company. "
        "Be grateful, leave the door open, keep it under 150 words. Include subject line."}])
    content = r.get("content", "")
    words = len(content.split())
    has_subject = "subject" in content.lower() or "re:" in content.lower()
    has_gratitude = any(w in content.lower() for w in ["thank", "grateful", "appreciate"])
    passed = words <= 170 and has_subject and has_gratitude
    return r, passed, f"Must have subject line, gratitude, ≤150 words (got {words})"


if __name__ == "__main__":
    suite.run()
