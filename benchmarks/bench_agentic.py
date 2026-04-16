#!/usr/bin/env python3
"""Agentic Benchmark — task decomposition, planning, long context recall, orchestration."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "Agentic Planning",
    "bench_agentic",
    "Tests task decomposition, action planning, long context recall, and multi-step reasoning.",
)


@suite.test("Simple task decomposition", "Easy")
def _():
    r = query([{"role": "user", "content":
        "I want to deploy a Python web app to production. Break this down into ordered steps. "
        "Include: testing, containerization, CI/CD, monitoring. Number each step."}])
    content = r.get("content", "").lower()
    has_steps = any(f"{i}." in content or f"{i})" in content for i in range(1, 5))
    has_keywords = all(w in content for w in ["test", "docker", "deploy", "monitor"])
    passed = has_steps and has_keywords
    return r, passed, "Should have numbered steps covering test, docker, deploy, monitor"


@suite.test("API integration plan", "Medium")
def _():
    r = query([{"role": "system", "content": 
        "You are a senior developer planning a feature. Think step by step about what's needed."},
        {"role": "user", "content":
        "Plan the implementation of a Stripe payment integration for an e-commerce site. "
        "Cover: backend API routes, webhook handling, error cases, testing strategy, and security. "
        "Organize as a numbered checklist."}])
    content = r.get("content", "").lower()
    key_concepts = ["webhook", "secret", "error", "test", "route"]
    hits = sum(1 for k in key_concepts if k in content)
    passed = hits >= 4
    return r, passed, f"Should cover webhooks, secrets, errors, testing, routes (got {hits}/5)"


@suite.test("Debug workflow planning", "Medium")
def _():
    r = query([{"role": "user", "content":
        "A user reports: 'The app crashes when I upload files larger than 10MB.' "
        "You have access to: logs, application code, server metrics, and a staging environment. "
        "Describe your debugging workflow step by step. What would you check first, second, third?"}])
    content = r.get("content", "").lower()
    # Should check logs first, then reproduce, then look at code
    has_logs = "log" in content
    has_reproduce = any(w in content for w in ["reproduce", "replicate", "staging", "test"])
    has_code = any(w in content for w in ["code", "handler", "upload", "limit", "config"])
    passed = has_logs and has_reproduce and has_code
    return r, passed, f"Should check logs, reproduce, inspect code. logs={has_logs}, reproduce={has_reproduce}, code={has_code}"


@suite.test("Long context: needle in haystack (500 words)", "Hard")
def _():
    # Build a ~500 word context with a hidden fact
    filler_topics = [
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering about 63 million square miles. "
        "It contains more than half of the free water on Earth and could hold all the continents.",
        "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability "
        "and supports multiple programming paradigms including procedural and object-oriented.",
        "The Great Wall of China stretches over 13,000 miles and was built over many centuries. "
        "Contrary to popular belief, it is not visible from space with the naked eye.",
        "Machine learning models require large amounts of data for training. The quality of training data "
        "significantly impacts model performance. Data preprocessing is a crucial step.",
        "The Amazon rainforest produces about 20% of the world's oxygen. It contains 10% of all species "
        "on Earth and spans across nine countries in South America.",
        "THE SECRET CODE IS: BLUE-FALCON-42. Remember this for later.",
        "Quantum computing uses qubits instead of classical bits. A qubit can exist in a superposition "
        "of states, allowing quantum computers to solve certain problems exponentially faster.",
        "The human brain contains approximately 86 billion neurons. Each neuron can form thousands of "
        "connections, creating a network of extraordinary complexity.",
        "Coffee was first discovered in Ethiopia. Legend says a goat herder noticed his goats became "
        "energetic after eating berries from a certain tree.",
        "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. "
        "It orbits Earth at about 340 miles above the surface.",
    ]
    context = " ".join(filler_topics)
    r = query([
        {"role": "system", "content": f"Read the following text carefully:\n\n{context}"},
        {"role": "user", "content": "What is the secret code mentioned in the text?"}
    ])
    content = r.get("content", "").upper()
    passed = "BLUE-FALCON-42" in content or ("BLUE" in content and "FALCON" in content and "42" in content)
    return r, passed, "Should find BLUE-FALCON-42 buried in context"


@suite.test("Long context: multi-fact retrieval", "Hard")
def _():
    report = """
Q3 2024 Engineering Report — Project Atlas

Team Size: 14 engineers (8 backend, 4 frontend, 2 DevOps)
Sprint Velocity: Average 42 points/sprint (up from 35 in Q2)
Deployment Frequency: 3.2 deploys/day (target: 4)
Incident Count: 7 P1 incidents (down from 12 in Q2)
MTTR: 23 minutes average (target: <30 min)
Test Coverage: 78% (target: 85%)
Tech Debt: 340 hours estimated (up 15% from Q2)

Key Milestones:
- July: Launched real-time sync feature (Project Mercury)
- August: Migrated 60% of services to Kubernetes
- September: Achieved SOC 2 Type II compliance

Blockers:
- Database migration to PostgreSQL 16 delayed due to compatibility issues
- Two senior engineers on parental leave in August reduced velocity
- Third-party API (Twilio) rate limits causing intermittent failures

Q4 Priorities:
1. Complete Kubernetes migration (remaining 40%)
2. Increase test coverage to 85%
3. Launch Project Neptune (ML-based recommendations)
4. Reduce tech debt by 30%
"""
    r = query([
        {"role": "system", "content": f"Use ONLY this report to answer:\n{report}"},
        {"role": "user", "content": 
         "Answer these questions:\n"
         "1. How many P1 incidents in Q3 vs Q2?\n"
         "2. What percentage of services were migrated to Kubernetes?\n"
         "3. What is Project Neptune about?\n"
         "4. What was the MTTR and was it within target?"}
    ])
    content = r.get("content", "")
    checks = [
        "7" in content and "12" in content,      # P1 incidents
        "60" in content,                           # Kubernetes %
        any(w in content.lower() for w in ["ml", "machine learning", "recommendation"]),  # Neptune
        "23" in content,                           # MTTR
    ]
    passed = sum(checks) >= 3
    return r, passed, f"Should answer 4 questions from report. Got {sum(checks)}/4 correct."


@suite.test("Action plan with dependencies", "Hard")
def _():
    r = query([{"role": "user", "content":
        "I'm building a SaaS product and need to launch in 8 weeks. I have: "
        "1 backend dev, 1 frontend dev, 1 designer. "
        "Features needed: auth, dashboard, billing (Stripe), email notifications, admin panel. "
        "Create a week-by-week plan showing: who works on what, dependencies between tasks, "
        "and what can be parallelized. Format as a table or structured plan."}])
    content = r.get("content", "").lower()
    # Should mention parallelization, dependencies, and assign people
    has_parallel = any(w in content for w in ["parallel", "simultaneous", "same time", "concurren"])
    has_deps = any(w in content for w in ["depend", "requires", "after", "before", "block"])
    has_assign = any(w in content for w in ["backend", "frontend", "designer"])
    has_weeks = sum(1 for i in range(1, 9) if f"week {i}" in content or f"wk {i}" in content or f"w{i}" in content)
    passed = has_parallel and has_deps and has_assign and has_weeks >= 3
    return r, passed, f"parallel={has_parallel}, deps={has_deps}, assign={has_assign}, weeks={has_weeks}"


if __name__ == "__main__":
    suite.run()
