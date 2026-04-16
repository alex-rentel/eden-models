#!/usr/bin/env python3
"""Multilingual Benchmark — translation, cross-lingual understanding, language detection."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "Multilingual",
    "bench_multilingual",
    "Tests translation accuracy, cross-lingual understanding, and language detection.",
)


@suite.test("English to Spanish", "Easy")
def _():
    r = query([{"role": "user", "content":
        "Translate to Spanish: 'The weather is beautiful today and I want to go to the park with my friends.'"}])
    content = r.get("content", "").lower()
    # Key words: tiempo/clima, hermoso/bonito, parque, amigos
    passed = ("parque" in content and "amigos" in content and 
              any(w in content for w in ["hermoso", "bonito", "bello", "lindo"]))
    return r, passed, "Should contain parque, amigos, and a word for beautiful"


@suite.test("Spanish to English", "Easy")
def _():
    r = query([{"role": "user", "content":
        "Translate to English: 'La inteligencia artificial está cambiando el mundo de maneras sorprendentes.'"}])
    content = r.get("content", "").lower()
    passed = "artificial intelligence" in content and "world" in content
    return r, passed, "Should mention 'artificial intelligence' and 'world'"


@suite.test("English to French", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Translate to French: 'I need to buy groceries including bread, cheese, and wine for tonight's dinner.'"}])
    content = r.get("content", "").lower()
    passed = ("pain" in content and "fromage" in content and "vin" in content)
    return r, passed, "Should contain pain (bread), fromage (cheese), vin (wine)"


@suite.test("Japanese to English", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Translate to English: 'プログラミングを学ぶことは、将来のキャリアにとって非常に重要です。'"}])
    content = r.get("content", "").lower()
    passed = ("programming" in content and "career" in content and 
              any(w in content for w in ["important", "crucial", "essential", "valuable"]))
    return r, passed, "Should mention programming, career, and importance"


@suite.test("Language detection", "Medium")
def _():
    r = query([{"role": "user", "content":
        "What language is this written in? Just name the language.\n"
        "'Ik hou van programmeren en het maken van nieuwe dingen.'"}])
    content = r.get("content", "").lower()
    passed = "dutch" in content or "nederlands" in content
    return r, passed, "Should identify Dutch"


@suite.test("Respond in requested language", "Medium")
def _():
    r = query([{"role": "user", "content":
        "Respond in German: What are three benefits of exercise?"}])
    content = r.get("content", "").lower()
    # German indicators
    german_words = ["und", "der", "die", "das", "ist", "gesundheit", "körper", "hilft", 
                    "verbessert", "bewegung", "training", "sport", "vorteile"]
    german_count = sum(1 for w in german_words if w in content)
    passed = german_count >= 3
    return r, passed, f"Should respond in German (found {german_count} German words)"


@suite.test("Idiomatic translation", "Hard")
def _():
    r = query([{"role": "user", "content":
        "Translate this English idiom to its Spanish equivalent (not literal translation): "
        "'It's raining cats and dogs.'"}])
    content = r.get("content", "").lower()
    # Spanish equivalent: "Llueve a cántaros" or similar
    passed = any(w in content for w in ["cántaros", "cantaros", "mares", "diluvio", "chuzos"]) or \
             ("not literal" in content or "equivalent" in content or "idiom" in content)
    return r, passed, "Should give idiomatic Spanish equivalent, not literal translation"


if __name__ == "__main__":
    suite.run()
