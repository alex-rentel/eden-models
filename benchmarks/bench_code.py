#!/usr/bin/env python3
"""Code Benchmark — generation, debugging, refactoring. All code is executed and validated."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import BenchmarkSuite, query, run_python

suite = BenchmarkSuite(
    "Code",
    "bench_code",
    "Tests code generation, debugging, and refactoring. All code is executed.",
)

CODE_PROMPT = "Return ONLY the Python code. No explanation, no markdown prose outside the code block."


@suite.test("FizzBuzz with twist", "Easy")
def _():
    r = query([{"role": "user", "content":
        f"Write a Python script that prints numbers 1-30. For multiples of 3 print 'Fizz', "
        f"multiples of 5 print 'Buzz', multiples of both print 'FizzBuzz', "
        f"and multiples of 7 print 'Boom'. If a number is a multiple of both 3 and 7, print 'FizzBoom'. "
        f"At the end, print 'DONE'. {CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "FizzBuzz" in ex["stdout"] and "FizzBoom" in ex["stdout"] and "DONE" in ex["stdout"]
    return r, passed, f"Must run and print FizzBuzz, FizzBoom, DONE. {r['exec_output'][:150]}"


@suite.test("Binary search (first occurrence)", "Medium")
def _():
    r = query([{"role": "user", "content":
        f"Write a complete Python script implementing binary search that returns the index of the "
        f"FIRST occurrence of a target in a sorted list, or -1 if not found. Test with:\n"
        f"  assert binary_search([1,2,2,2,3,4], 2) == 1\n"
        f"  assert binary_search([1,2,3,4,5], 6) == -1\n"
        f"  assert binary_search([], 1) == -1\n"
        f"  assert binary_search([5], 5) == 0\n"
        f"  assert binary_search([1,1,1,1,1], 1) == 0\n"
        f"  print('ALL PASSED')\n{CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "ALL PASSED" in ex["stdout"]
    return r, passed, f"Must pass all assertions. {r['exec_output'][:150]}"


@suite.test("LRU Cache from scratch", "Hard")
def _():
    r = query([{"role": "user", "content":
        f"Implement an LRU Cache class in Python with get(key) and put(key, value), both O(1). "
        f"Do NOT use functools. Use an OrderedDict or doubly-linked list + dict.\n"
        f"Test with:\n"
        f"  c = LRUCache(2)\n"
        f"  c.put(1, 1); c.put(2, 2)\n"
        f"  assert c.get(1) == 1\n"
        f"  c.put(3, 3)  # evicts 2\n"
        f"  assert c.get(2) == -1\n"
        f"  c.put(4, 4)  # evicts 1\n"
        f"  assert c.get(1) == -1\n"
        f"  assert c.get(3) == 3\n"
        f"  assert c.get(4) == 4\n"
        f"  print('ALL PASSED')\n{CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "ALL PASSED" in ex["stdout"]
    return r, passed, f"Must implement O(1) LRU. {r['exec_output'][:150]}"


@suite.test("Debug: off-by-one error", "Medium")
def _():
    buggy_code = '''
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i <= len(a) and j <= len(b):  # BUG: should be <, not <=
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result

print(merge_sorted([1, 3, 5], [2, 4, 6]))
'''
    r = query([{"role": "user", "content":
        f"This Python code has a bug. Find it, explain it, and provide the corrected code that "
        f"runs successfully. Add this test at the end:\n"
        f"  assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]\n"
        f"  assert merge_sorted([], [1,2]) == [1,2]\n"
        f"  assert merge_sorted([1], []) == [1]\n"
        f"  print('ALL PASSED')\n\nBuggy code:\n```python\n{buggy_code}\n```\n{CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "ALL PASSED" in ex["stdout"]
    return r, passed, f"Must fix the off-by-one and pass tests. {r['exec_output'][:150]}"


@suite.test("Debug: logic error in recursion", "Hard")
def _():
    buggy_code = '''
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            flatten(item)  # BUG: doesn't use return value
        else:
            result.append(item)
    return result

print(flatten([1, [2, [3, 4], 5], 6]))  # Should print [1, 2, 3, 4, 5, 6]
'''
    r = query([{"role": "user", "content":
        f"This recursive flatten function has a bug. Find it, explain it, and provide corrected code.\n"
        f"Add at the end:\n"
        f"  assert flatten([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]\n"
        f"  assert flatten([]) == []\n"
        f"  assert flatten([[[[1]]]]) == [1]\n"
        f"  print('ALL PASSED')\n\nBuggy code:\n```python\n{buggy_code}\n```\n{CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "ALL PASSED" in ex["stdout"]
    return r, passed, f"Must fix the recursion bug. {r['exec_output'][:150]}"


@suite.test("Refactor: extract and improve", "Hard")
def _():
    messy_code = '''
data = [{"name": "alice", "score": 85}, {"name": "bob", "score": 92}, 
        {"name": "charlie", "score": 78}, {"name": "diana", "score": 95},
        {"name": "eve", "score": 88}]
result = []
for d in data:
    if d["score"] >= 80:
        result.append({"name": d["name"].upper(), "score": d["score"], "grade": "A" if d["score"] >= 90 else "B"})
result.sort(key=lambda x: x["score"], reverse=True)
for r in result:
    print(f"{r['name']}: {r['score']} ({r['grade']})")
'''
    r = query([{"role": "user", "content":
        f"Refactor this code to use list comprehensions, proper functions, and type hints. "
        f"Keep the same output. Add at the end: print('REFACTORED')\n\n"
        f"```python\n{messy_code}\n```\n{CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "REFACTORED" in ex["stdout"] and "DIANA" in ex["stdout"]
    return r, passed, f"Must produce same output + REFACTORED. {r['exec_output'][:150]}"


@suite.test("Tree traversals", "Medium")
def _():
    r = query([{"role": "user", "content":
        f"Write a complete Python script that:\n"
        f"1. Defines a binary tree node class\n"
        f"2. Builds this tree:       4\n"
        f"                          / \\\n"
        f"                         2   6\n"
        f"                        / \\ / \\\n"
        f"                       1  3 5  7\n"
        f"3. Implements inorder, preorder, postorder traversal\n"
        f"4. Asserts inorder == [1,2,3,4,5,6,7]\n"
        f"5. Asserts preorder == [4,2,1,3,6,5,7]\n"
        f"6. Asserts postorder == [1,3,2,5,7,6,4]\n"
        f"7. Prints 'ALL PASSED'\n{CODE_PROMPT}"}])
    ex = run_python(r.get("content", ""))
    r["exec_output"] = ex["stdout"][:500] if ex["success"] else ex["stderr"][:500]
    passed = ex["success"] and "ALL PASSED" in ex["stdout"]
    return r, passed, f"Must pass all traversal assertions. {r['exec_output'][:150]}"


if __name__ == "__main__":
    suite.run()
