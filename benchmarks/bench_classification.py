#!/usr/bin/env python3
"""Classification Benchmark — sentiment, entity extraction, intent detection, topic tagging."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "Classification",
    "bench_classification",
    "Tests sentiment analysis, NER, intent detection, and topic classification.",
)

CLASSIFY_PROMPT = "Return ONLY the classification label, nothing else."


@suite.test("Sentiment: clearly positive", "Easy")
def _():
    r = query([{"role": "user", "content":
        f"Classify the sentiment as 'positive', 'negative', or 'neutral':\n"
        f"'This is hands down the best laptop I've ever owned. Battery life is incredible!'\n{CLASSIFY_PROMPT}"}])
    passed = "positive" in r.get("content", "").lower()
    return r, passed, "Expected: positive"


@suite.test("Sentiment: clearly negative", "Easy")
def _():
    r = query([{"role": "user", "content":
        f"Classify the sentiment as 'positive', 'negative', or 'neutral':\n"
        f"'Terrible service. Waited 2 hours and the food was cold. Never coming back.'\n{CLASSIFY_PROMPT}"}])
    passed = "negative" in r.get("content", "").lower()
    return r, passed, "Expected: negative"


@suite.test("Sentiment: subtle/mixed", "Medium")
def _():
    r = query([{"role": "user", "content":
        f"Classify the sentiment as 'positive', 'negative', or 'neutral':\n"
        f"'The camera is amazing but the battery drains way too fast for the price.'\n{CLASSIFY_PROMPT}"}])
    content = r.get("content", "").lower()
    # Mixed could be classified as negative or neutral (complaint outweighs praise in context of price)
    passed = any(w in content for w in ["mixed", "neutral", "negative"])
    return r, passed, "Expected: mixed, neutral, or negative"


@suite.test("Named Entity Recognition", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Extract all named entities from this text. Return as JSON with keys 'people', 'organizations', "
        "'locations', each containing a list.\n\n"
        "'Tim Cook announced that Apple will open a new headquarters in Austin, Texas. "
        "The project, backed by Goldman Sachs, will create 5,000 jobs. "
        "Senator Maria Garcia praised the initiative.'"}])
    content = r.get("content", "").strip()
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content[:-3].strip()
    try:
        parsed = json.loads(content)
        people = [p.lower() for p in parsed.get("people", [])]
        orgs = [o.lower() for o in parsed.get("organizations", [])]
        locs = [l.lower() for l in parsed.get("locations", [])]
        has_people = any("tim cook" in p for p in people) and any("garcia" in p for p in people)
        has_orgs = any("apple" in o for o in orgs) and any("goldman" in o for o in orgs)
        has_locs = any("austin" in l or "texas" in l for l in locs)
        passed = has_people and has_orgs and has_locs
    except json.JSONDecodeError:
        passed = False
    return r, passed, "Must extract Tim Cook, Garcia, Apple, Goldman Sachs, Austin/Texas"


@suite.test("Intent detection: single", "Easy")
def _():
    r = query([{"role": "user", "content":
        "Classify the user intent. Options: 'book_flight', 'cancel_booking', 'check_status', 'complaint'.\n\n"
        f"User: 'I need to fly from NYC to LA next Friday, preferably morning.'\n{CLASSIFY_PROMPT}"}])
    passed = "book_flight" in r.get("content", "").lower()
    return r, passed, "Expected: book_flight"


@suite.test("Intent detection: ambiguous", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Classify the user intent. Options: 'refund_request', 'exchange_request', 'complaint', 'inquiry'.\n\n"
        f"User: 'I got the wrong size. Can I get my money back or swap it for a medium?'\n{CLASSIFY_PROMPT}"}])
    content = r.get("content", "").lower()
    # Could be refund or exchange — both are reasonable
    passed = any(w in content for w in ["refund", "exchange"])
    return r, passed, "Expected: refund_request or exchange_request"


@suite.test("Topic classification: multi-label", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Assign 1-3 topic labels from: ['technology', 'business', 'health', 'politics', 'sports', "
        "'science', 'entertainment']. Return as a JSON array.\n\n"
        "'The FDA approved a new AI-powered diagnostic tool developed by Google Health, "
        "marking a milestone in the $50B digital health market.'\n"
        "Return ONLY the JSON array."}])
    content = r.get("content", "").strip()
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content[:-3].strip()
    try:
        labels = json.loads(content)
        label_set = set(l.lower() for l in labels)
        # Should include technology and health, maybe business
        passed = "technology" in label_set and "health" in label_set
    except json.JSONDecodeError:
        passed = False
    return r, passed, "Expected at least 'technology' and 'health'"


@suite.test("Spam detection", "Easy")
def _():
    r = query([{"role": "user", "content":
        f"Is this email spam or legitimate? Reply with only 'spam' or 'legitimate'.\n\n"
        f"'CONGRATULATIONS! You've been selected as our LUCKY WINNER! "
        f"Click here to claim your $1,000,000 prize! Act NOW before it expires!!!'\n{CLASSIFY_PROMPT}"}])
    passed = "spam" in r.get("content", "").lower()
    return r, passed, "Expected: spam"


if __name__ == "__main__":
    suite.run()
