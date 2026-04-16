#!/usr/bin/env python3
"""RAG Benchmark — grounded QA, context adherence, hallucination resistance."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query

suite = BenchmarkSuite(
    "RAG / Grounded QA",
    "bench_rag",
    "Tests ability to answer from provided context, resist hallucination, and admit uncertainty.",
)

DOC_STARTUP = """
Acme Robotics Inc. was founded in 2019 by Dr. Sarah Chen and Marcus Rivera in Austin, Texas.
The company develops autonomous warehouse robots. Their flagship product, the ArBot-3, can
carry up to 200kg and operates for 12 hours on a single charge. In 2023, they raised a $45M
Series B led by Spark Ventures. The company has 127 employees as of Q4 2024. Their main
competitor is WareBot Systems, based in Boston. Acme's revenue was $18.2M in 2023 and 
projected to reach $32M in 2024. The ArBot-3 uses LiDAR and computer vision for navigation.
Dr. Chen previously worked at Boston Dynamics for 8 years. Marcus Rivera is the CTO and holds
12 patents in robotics navigation. The company operates from a 50,000 sq ft facility in
East Austin and has a secondary office in San Francisco.
"""

DOC_POLICY = """
Remote Work Policy — TechCorp Inc. (Effective January 2025)

Eligibility: All full-time employees who have completed their 90-day probation period.
Part-time employees and contractors are NOT eligible for remote work.

Schedule: Employees may work remotely up to 3 days per week. Tuesday and Thursday are
mandatory in-office days. Remote work on Monday, Wednesday, or Friday requires manager
approval 48 hours in advance.

Equipment: The company provides a laptop and one monitor. Employees must supply their own
desk and chair. A $500 annual stipend is available for home office setup (receipts required).

Security: All remote work must use the company VPN. Personal devices may NOT access company
systems. Screen sharing during meetings is required when presenting.

Exceptions: Engineering leads and above may work fully remote with VP approval.
Customer-facing roles (Sales, Support) must be in-office minimum 4 days per week.
"""


@suite.test("Simple fact extraction", "Easy")
def _():
    r = query([
        {"role": "system", "content": f"Answer questions using ONLY the following document:\n{DOC_STARTUP}"},
        {"role": "user", "content": "How much can the ArBot-3 carry?"}
    ])
    passed = "200" in r.get("content", "")
    return r, passed, "Expected 200kg"


@suite.test("Multi-fact question", "Medium")
def _():
    r = query([
        {"role": "system", "content": f"Answer questions using ONLY the following document:\n{DOC_STARTUP}"},
        {"role": "user", "content": "Who founded the company, when, and where?"}
    ])
    content = r.get("content", "")
    passed = all(w in content for w in ["Sarah Chen", "Marcus Rivera", "2019", "Austin"])
    return r, passed, "Must mention both founders, 2019, and Austin"


@suite.test("Numerical reasoning from context", "Medium")
def _():
    r = query([
        {"role": "system", "content": f"Answer questions using ONLY the following document:\n{DOC_STARTUP}"},
        {"role": "user", "content": "What was the projected revenue growth percentage from 2023 to 2024?"}
    ])
    content = r.get("content", "")
    # 18.2 to 32: (32-18.2)/18.2 ≈ 75.8%
    passed = any(x in content for x in ["75", "76", "75.8", "~76"])
    return r, passed, "Expected ~75-76% growth"


@suite.test("Answer NOT in context (should say so)", "Hard")
def _():
    r = query([
        {"role": "system", "content": f"Answer questions using ONLY the following document. "
         f"If the answer is not in the document, say 'Not mentioned in the document'.\n{DOC_STARTUP}"},
        {"role": "user", "content": "What programming language does the ArBot-3 use?"}
    ])
    content = r.get("content", "").lower()
    refusal_signals = ["not mentioned", "not specified", "doesn't mention", "not provided",
                       "document doesn't", "no information", "not stated", "not included"]
    passed = any(s in content for s in refusal_signals)
    return r, passed, "Should indicate info is not in the document"


@suite.test("Resist hallucination with plausible wrong answer", "Hard")
def _():
    r = query([
        {"role": "system", "content": f"Answer questions using ONLY the following document. "
         f"Do NOT make up information.\n{DOC_STARTUP}"},
        {"role": "user", "content": "How many Series A funding did Acme raise?"}
    ])
    content = r.get("content", "").lower()
    # Doc only mentions Series B ($45M). Should NOT invent a Series A number.
    refusal = any(s in content for s in ["not mentioned", "only", "series b", "doesn't mention",
                                          "no information", "not specified"])
    hallucinated = any(s in content for s in ["$10m", "$15m", "$20m", "$25m", "$5m", "series a was"])
    passed = refusal and not hallucinated
    return r, passed, "Should note only Series B is mentioned, NOT invent Series A amount"


@suite.test("Policy: eligible or not?", "Medium")
def _():
    r = query([
        {"role": "system", "content": f"Use ONLY this policy to answer:\n{DOC_POLICY}"},
        {"role": "user", "content": "I'm a part-time contractor who's been here 6 months. Can I work remotely?"}
    ])
    content = r.get("content", "").lower()
    passed = any(s in content for s in ["not eligible", "no", "cannot", "aren't eligible", "ineligible"])
    return r, passed, "Should say not eligible (part-time AND contractor both excluded)"


@suite.test("Policy: complex scenario", "Hard")
def _():
    r = query([
        {"role": "system", "content": f"Use ONLY this policy to answer:\n{DOC_POLICY}"},
        {"role": "user", "content": 
         "I'm a full-time engineering lead who completed probation. Can I work from home "
         "every day? What approvals do I need?"}
    ])
    content = r.get("content", "").lower()
    # Eng leads can be fully remote with VP approval
    passed = ("vp" in content or "vice president" in content) and "approv" in content
    return r, passed, "Should mention VP approval needed for fully remote engineering leads"


if __name__ == "__main__":
    suite.run()
