#!/usr/bin/env python3
"""
Generate high-quality tool-calling training data for Eden models.
Template-based generation — no API calls needed.

Output: mlx_lm.lora compatible JSONL with OpenAI-style messages + tools.

Usage:
    python3 data/generation/generate_eden_data.py --num 15000 --output data/eden_15k.jsonl
"""

import json
import random
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

# ─── Eden Tool Definitions (OpenAI format for mlx_lm.lora) ─────

EDEN_TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command and return the output",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "file_read",
        "description": "Read the contents of a file at the given path",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "offset": {"type": "integer", "description": "Line number to start from"},
            "limit": {"type": "integer", "description": "Number of lines to read"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "file_write",
        "description": "Write content to a file, creating it if it doesn't exist",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to write to"},
            "content": {"type": "string", "description": "Content to write"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "file_edit",
        "description": "Search and replace a unique string in a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to edit"},
            "old_str": {"type": "string", "description": "String to find"},
            "new_str": {"type": "string", "description": "Replacement string"},
        }, "required": ["path", "old_str", "new_str"]},
    }},
    {"type": "function", "function": {
        "name": "glob",
        "description": "Find files matching a glob pattern",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match"},
            "path": {"type": "string", "description": "Directory to search in"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search for a regex pattern in files",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "File or directory to search"},
            "include": {"type": "string", "description": "File glob filter"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "python_run",
        "description": "Run Python code and return output",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to run"},
        }, "required": ["code"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web and return results",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch a web page URL and return text content",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
        }, "required": ["url"]},
    }},
]

TOOL_BY_NAME = {t["function"]["name"]: t for t in EDEN_TOOLS}
SYSTEM_MSG = ("You are Eden, a local AI coding assistant. You have access to "
              "tools that you can call to help answer questions. Use them when needed.")


def sample_tools(must_include: Optional[str] = None, k: int = 5) -> List[Dict]:
    pool = list(EDEN_TOOLS)
    if must_include:
        chosen = [TOOL_BY_NAME[must_include]]
        pool = [t for t in pool if t["function"]["name"] != must_include]
        chosen += random.sample(pool, min(k - 1, len(pool)))
    else:
        chosen = random.sample(pool, min(k, len(pool)))
    random.shuffle(chosen)
    return chosen


def tc(name: str, args: Dict) -> Dict:
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

def msg_sys():
    return {"role": "system", "content": SYSTEM_MSG}

def msg_user(content: str):
    return {"role": "user", "content": content}

def msg_asst(content: str, tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m

def msg_tool(name: str, content: str):
    return {"role": "tool", "name": name, "content": content}

def make_example(messages: List[Dict], tools: List[Dict]) -> Dict:
    return {"messages": messages, "tools": tools}


# ─── Data Pools ────────────────────────────────────────────────

FILE_PATHS = [
    "src/main.py", "src/app.py", "src/server.py", "src/utils.py", "src/config.py",
    "src/auth/login.py", "src/auth/middleware.py", "src/api/routes.py",
    "src/models/user.py", "src/models/database.py", "src/services/email.py",
    "tests/test_main.py", "tests/test_auth.py", "tests/test_api.py",
    "README.md", "pyproject.toml", "Dockerfile", "docker-compose.yml",
    "config.yaml", "config/settings.json", ".gitignore", "Makefile",
    "requirements.txt", "migrations/001_init.py",
]

DIR_PATHS = [".", "src", "src/api", "src/auth", "tests", "config", "scripts"]

SHELL_CMDS = {
    "list": (["ls -la", "ls -la src/", "ls *.py"],
             ["List files here", "Show me what's in this directory", "What files are in src/"]),
    "git": (["git status", "git log --oneline -10", "git diff", "git branch -a"],
            ["Show git status", "What are the recent commits?", "Any uncommitted changes?", "Show branches"]),
    "find": (["find . -name '*.py' -type f", "find . -name '*.py' | wc -l"],
             ["Find all Python files", "How many Python files are there?"]),
    "proc": (["ps aux | grep python", "lsof -i :8080", "docker ps"],
             ["What's running on port 8080?", "Show running processes", "Are containers running?"]),
    "pkg": (["pip install requests", "pip install -r requirements.txt", "pip install --upgrade numpy"],
            ["Install requests", "Install dependencies from requirements.txt", "Upgrade numpy"]),
    "test": (["python3 -m pytest tests/", "python3 -m pytest tests/test_auth.py -v"],
             ["Run the tests", "Run auth tests with verbose output"]),
    "run": (["python3 src/main.py", "python3 -m uvicorn src.app:app --reload"],
            ["Start the server", "Run the app with hot reload"]),
    "sys": (["du -sh */", "df -h", "python3 --version", "uname -a"],
            ["Show disk usage", "How much disk space?", "What Python version?", "System info"]),
    "docker": (["docker build -t myapp .", "docker-compose up -d", "docker logs myapp --tail 50"],
               ["Build the Docker image", "Start containers", "Show Docker logs"]),
}

PYTHON_SNIPPETS = [
    ("calculate 2^100", "print(2**100)", "1267650600228229401496703205376"),
    ("generate a UUID", "import uuid; print(uuid.uuid4())", "f47ac10b-58cc-4372-a567-0e02b2c3d479"),
    ("get today's date", "from datetime import date; print(date.today())", "2026-04-03"),
    ("parse JSON", 'import json; print(json.loads(\'{"users": 42}\'))', "{'users': 42}"),
    ("list squares", "print([x**2 for x in range(10)])", "[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]"),
    ("base64 encode", "import base64; print(base64.b64encode(b'hello').decode())", "aGVsbG8="),
    ("random password", "import secrets,string; print(''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(16)))", "kX9mQ2vRpL4wJn7B"),
    ("check platform", "import platform; print(platform.system(), platform.machine())", "Darwin arm64"),
    ("count words", "text='hello world foo bar'; print(len(text.split()))", "4"),
    ("fibonacci", "fib=lambda n:n if n<2 else fib(n-1)+fib(n-2); print([fib(i) for i in range(10)])", "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"),
]

GREP_PATTERNS = [
    ("TODO comments", "TODO"), ("FIXME markers", "FIXME"),
    ("import requests", "import requests"), ("test functions", "def test_"),
    ("error classes", "class.*Error"), ("DATABASE_URL", "DATABASE_URL"),
    ("api_key references", "api_key"), ("main function", "def main"),
    ("async functions", "async def"), ("logging calls", "logging\\.error"),
]

GLOB_PATTERNS = [
    ("**/*.py", "Python files"), ("tests/test_*.py", "test files"),
    ("**/*.md", "markdown files"), ("src/**/*.py", "source files"),
    ("**/*.yaml", "YAML configs"), ("**/*.json", "JSON files"),
    ("docker*", "Docker files"),
]

GENERAL_QA = [
    ("What is a Python decorator?",
     "A decorator is a function that wraps another function to extend its behavior. Use `@decorator` syntax. Common uses: logging, auth, caching.\n\n```python\ndef timer(func):\n    def wrapper(*args):\n        t0 = time.time()\n        result = func(*args)\n        print(f'{time.time()-t0:.3f}s')\n        return result\n    return wrapper\n```"),
    ("Explain list vs tuple in Python",
     "Lists are mutable (`[]`), tuples are immutable (`()`). Tuples are faster, hashable (usable as dict keys), and good for fixed collections. Lists are for collections you need to modify."),
    ("What does git rebase do?",
     "Git rebase replays your commits on top of another branch, creating a linear history. `git rebase main` moves your feature commits after main's latest. Unlike merge, it doesn't create merge commits. Interactive rebase (`-i`) lets you squash/reorder commits."),
    ("How does async/await work?",
     "`async def` creates a coroutine. `await` suspends it until the awaited operation completes, letting other coroutines run. You need `asyncio.run()` to start the event loop. Best for I/O-bound work: network calls, file I/O, database queries."),
    ("What is Docker?",
     "Docker packages apps in containers with all dependencies. Dockerfile defines the image, `docker build` creates it, `docker run` starts a container. Docker Compose manages multi-container apps. Containers are lighter than VMs."),
    ("How do virtual environments work?",
     "`python3 -m venv .venv` creates an isolated Python. Activate with `source .venv/bin/activate`. Packages install into the venv, not system Python. Track deps with `requirements.txt` or `pyproject.toml`. Deactivate with `deactivate`."),
    ("What is type hinting in Python?",
     "Type hints annotate expected types: `def greet(name: str) -> str`. Not enforced at runtime but enable mypy checks and better IDE support. Use `List[int]`, `Dict[str, Any]`, `Optional[str]` from typing module."),
    ("Explain REST API",
     "REST uses HTTP methods on resources: GET (read), POST (create), PUT (update), DELETE (remove). URLs identify resources (`/api/users/123`). Responses use JSON. Key: stateless — each request has all needed info."),
    ("What is the GIL?",
     "CPython's Global Interpreter Lock allows only one thread to run Python bytecode at a time. CPU-bound threads don't get true parallelism. Use `multiprocessing` for CPU work, `asyncio`/threading for I/O. It's a CPython detail, not a language feature."),
    ("How do I handle errors in Python?",
     "Use try/except:\n```python\ntry:\n    value = int(input)\nexcept ValueError:\n    print('Not a number')\nexcept Exception as e:\n    logger.error(f'Unexpected: {e}')\nfinally:\n    cleanup()\n```\nCatch specific exceptions, not bare `except`."),
    ("What are f-strings?",
     "f-strings (Python 3.6+) embed expressions in strings: `f'Hello {name}'`. Support formatting: `f'{pi:.2f}'`, expressions: `f'{len(items)} items'`, and even multiline with `f'''...'''`. Faster than `.format()` or `%`."),
    ("Explain context managers",
     "`with` statement manages resources automatically. Files, locks, DB connections close properly even if exceptions occur:\n```python\nwith open('file.txt') as f:\n    data = f.read()\n# f is automatically closed\n```\nCreate custom ones with `__enter__`/`__exit__` or `@contextmanager`."),
]

SEARCH_QUERIES = [
    ("Search for the latest MLX release", "MLX framework latest release"),
    ("Find Python asyncio best practices", "Python asyncio best practices 2026"),
    ("What's the latest version of FastAPI?", "FastAPI latest version"),
    ("Look up SQLAlchemy 2.0 migration guide", "SQLAlchemy 2.0 migration guide"),
    ("Find Docker best practices for Python", "Docker best practices Python production"),
    ("Search for pytest fixture docs", "pytest fixtures documentation"),
    ("Find the MLX fine-tuning guide", "MLX LoRA fine-tuning guide Apple Silicon"),
    ("Look up pydantic v2 changes", "pydantic v2 migration changes"),
]

FETCH_URLS = [
    ("Fetch the API docs", "https://api.example.com/docs",
     "# API Documentation\n\n## Endpoints\n\nGET /users - List users\nPOST /users - Create user\nGET /users/:id - Get user\nDELETE /users/:id - Delete user"),
    ("Read that page", "https://docs.example.com/getting-started",
     "# Getting Started\n\nInstall with pip:\n```\npip install mylib\n```\n\nBasic usage:\n```python\nfrom mylib import Client\nclient = Client(api_key='...')\n```"),
]


# ─── Result generators ─────────────────────────────────────────

def gen_ls():
    files = random.sample([
        "drwxr-xr-x  5 user staff  160 Mar 15 10:23 src",
        "drwxr-xr-x  3 user staff   96 Mar 14 09:11 tests",
        "-rw-r--r--  1 user staff 2847 Mar 15 14:30 README.md",
        "-rw-r--r--  1 user staff  542 Mar 12 11:00 pyproject.toml",
        "-rw-r--r--  1 user staff 1205 Mar 15 10:23 Dockerfile",
        "-rw-r--r--  1 user staff  187 Mar 10 08:45 .gitignore",
        "-rw-r--r--  1 user staff  923 Mar 15 14:12 requirements.txt",
        "drwxr-xr-x  4 user staff  128 Mar 13 16:00 config",
        "-rw-r--r--  1 user staff 4521 Mar 15 13:45 docker-compose.yml",
    ], k=random.randint(4, 7))
    return f"total {random.randint(20, 80)}\n" + "\n".join(files)

def gen_git_status():
    mod = random.sample(FILE_PATHS[:10], k=random.randint(1, 3))
    branch = random.choice(["main", "feature/auth", "fix/login-bug", "dev"])
    return f"On branch {branch}\nChanges not staged for commit:\n" + "\n".join(f"\tmodified:   {f}" for f in mod)

def gen_git_log():
    msgs = ["fix: resolve login timeout", "feat: add user profile endpoint",
            "refactor: extract auth middleware", "test: add integration tests",
            "docs: update README", "chore: bump deps", "fix: null email handling"]
    return "\n".join(f"{random.randint(0,0xfffffff):07x} {m}" for m in random.sample(msgs, 5))

def gen_pytest(ok=True):
    n = random.randint(8, 20)
    if ok:
        return f"{'.'*n} [100%]\n\n{n} passed in {random.uniform(0.5,4):.2f}s"
    f = random.randint(1, 2)
    return f"{'.'*(n-f)}{'F'*f} [100%]\n\nFAILED tests/test_auth.py::test_login - AssertionError\n{n-f} passed, {f} failed in {random.uniform(1,6):.2f}s"

def gen_code(kind="func"):
    snippets = {
        "func": 'def process_user(data: dict) -> dict:\n    email = data.get("email", "").strip().lower()\n    if not email or "@" not in email:\n        raise ValueError(f"Invalid email: {email}")\n    return {"email": email, "username": email.split("@")[0]}',
        "cls": 'class DatabasePool:\n    def __init__(self, url: str, size: int = 5):\n        self.url = url\n        self.size = size\n        self._pool = None\n\n    async def connect(self):\n        self._pool = await asyncpg.create_pool(self.url, min_size=self.size)',
        "cfg": 'import os\n\nDATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/myapp")\nSECRET_KEY = os.getenv("SECRET_KEY", "change-me")\nDEBUG = os.getenv("DEBUG", "false").lower() == "true"\nPORT = int(os.getenv("PORT", "8080"))',
        "test": 'import pytest\nfrom src.auth import authenticate\n\ndef test_valid_login():\n    result = authenticate("user@test.com", "pass123")\n    assert result["status"] == "success"\n    assert "token" in result\n\ndef test_invalid_login():\n    with pytest.raises(ValueError):\n        authenticate("", "pass123")',
    }
    return snippets.get(kind, snippets["func"])

def gen_grep_out(desc):
    results = []
    for _ in range(random.randint(2, 5)):
        f = random.choice(FILE_PATHS[:12])
        ln = random.randint(5, 150)
        results.append(f"{f}:{ln}:    {desc}")
    return "\n".join(results)

def gen_glob_out(pat):
    ext = pat.split("*.")[-1] if "*." in pat else "py"
    return "\n".join(random.sample([
        f"src/main.{ext}", f"src/app.{ext}", f"src/utils.{ext}",
        f"src/api/routes.{ext}", f"tests/test_main.{ext}", f"tests/conftest.{ext}",
    ], k=random.randint(3, 5)))


# ─── Category Generators ──────────────────────────────────────

def gen_single_bash():
    cat = random.choice(list(SHELL_CMDS.keys()))
    cmds, queries = SHELL_CMDS[cat]
    cmd, query = random.choice(cmds), random.choice(queries)
    result_map = {"list": gen_ls(), "git": gen_git_status() if "status" in cmd else gen_git_log(),
                  "find": "\n".join(random.sample(FILE_PATHS[:12], 5)), "proc": "user 12345 0.5 1.2 python3 src/main.py",
                  "pkg": "Successfully installed requests-2.31.0", "test": gen_pytest(),
                  "run": "INFO: Uvicorn running on http://0.0.0.0:8080", "sys": "Python 3.12.2",
                  "docker": "CONTAINER ID  IMAGE  STATUS\na1b2c3  myapp  Up 2 min"}
    summary_map = {"list": "Here are the files.", "git": "Here's the repository status.",
                   "test": "All tests passed.", "pkg": "Package installed successfully.",
                   "run": "The server is now running on port 8080."}
    return make_example([
        msg_sys(), msg_user(query), msg_asst("", [tc("bash", {"command": cmd})]),
        msg_tool("bash", result_map.get(cat, "OK")),
        msg_asst(summary_map.get(cat, "Done. Here's the output.")),
    ], sample_tools("bash"))

def gen_single_file_read():
    path = random.choice(FILE_PATHS)
    queries = [f"Read {path}", f"Show me {path}", f"What's in {path}?", f"Open {path}"]
    args = {"path": path}
    if random.random() < 0.3:
        args["limit"] = random.choice([20, 50])
    kind = "test" if "test" in path else ("cfg" if "config" in path or "toml" in path else "func")
    return make_example([
        msg_sys(), msg_user(random.choice(queries)),
        msg_asst("", [tc("file_read", args)]),
        msg_tool("file_read", gen_code(kind)),
        msg_asst(f"Here's the contents of `{path}`."),
    ], sample_tools("file_read"))

def gen_single_file_write():
    templates = [
        ("Create a .gitignore for Python", ".gitignore",
         "__pycache__/\n*.pyc\n.venv/\n.env\ndist/\n*.egg-info/\n.pytest_cache/"),
        ("Write a Dockerfile for a Python app", "Dockerfile",
         "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"src/main.py\"]"),
        ("Create a requirements.txt", "requirements.txt",
         "fastapi==0.109.0\nuvicorn==0.27.0\npydantic==2.6.0\nhttpx==0.26.0"),
        ("Write a hello world script", "hello.py",
         'def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()'),
        ("Create a basic config.yaml", "config.yaml",
         "server:\n  host: 0.0.0.0\n  port: 8080\n  debug: false\n\ndatabase:\n  url: postgresql://localhost/myapp"),
        ("Write a Makefile for the project", "Makefile",
         ".PHONY: run test lint\n\nrun:\n\tpython3 src/main.py\n\ntest:\n\tpython3 -m pytest tests/ -v\n\nlint:\n\tblack src/ tests/"),
    ]
    query, path, content = random.choice(templates)
    return make_example([
        msg_sys(), msg_user(query),
        msg_asst("", [tc("file_write", {"path": path, "content": content})]),
        msg_tool("file_write", f"Wrote {len(content)} bytes to {path}"),
        msg_asst(f"Created `{path}` with the requested content."),
    ], sample_tools("file_write"))

def gen_single_file_edit():
    templates = [
        ("Change the port from 8080 to 3000", "config.yaml", "  port: 8080", "  port: 3000"),
        ("Update version to 2.0.0", "pyproject.toml", 'version = "1.0.0"', 'version = "2.0.0"'),
        ("Fix the typo — 'configuraton' to 'configuration'", "README.md", "configuraton", "configuration"),
        ("Rename process_data to handle_data", "src/main.py", "def process_data(", "def handle_data("),
        ("Set DEBUG to False", "src/config.py", "DEBUG = True", "DEBUG = False"),
        ("Change database port to 5433", "config.yaml", "localhost/myapp", "localhost:5433/myapp"),
    ]
    query, path, old, new = random.choice(templates)
    return make_example([
        msg_sys(), msg_user(query),
        msg_asst("", [tc("file_edit", {"path": path, "old_str": old, "new_str": new})]),
        msg_tool("file_edit", f"Replaced in {path}"),
        msg_asst(f"Updated `{path}` — changed `{old.strip()}` to `{new.strip()}`."),
    ], sample_tools("file_edit"))

def gen_single_grep():
    desc, pat = random.choice(GREP_PATTERNS)
    queries = [f"Search for {desc}", f"Find {desc} in the code", f"Grep for {desc}", f"Where is {desc}?"]
    args = {"pattern": pat}
    if random.random() < 0.4: args["path"] = random.choice(DIR_PATHS[:4])
    if random.random() < 0.3: args["include"] = "*.py"
    return make_example([
        msg_sys(), msg_user(random.choice(queries)),
        msg_asst("", [tc("grep", args)]),
        msg_tool("grep", gen_grep_out(desc)),
        msg_asst(f"Found several matches for `{desc}` across the codebase."),
    ], sample_tools("grep"))

def gen_single_glob():
    pat, desc = random.choice(GLOB_PATTERNS)
    queries = [f"Find all {desc}", f"What {desc} exist?", f"List {desc} in the project"]
    args = {"pattern": pat}
    if random.random() < 0.3: args["path"] = random.choice(DIR_PATHS[:3])
    return make_example([
        msg_sys(), msg_user(random.choice(queries)),
        msg_asst("", [tc("glob", args)]),
        msg_tool("glob", gen_glob_out(pat)),
        msg_asst(f"Found several {desc} in the project."),
    ], sample_tools("glob"))

def gen_single_python():
    desc, code, result = random.choice(PYTHON_SNIPPETS)
    queries = [desc.capitalize(), f"Can you {desc}?", f"Use Python to {desc}"]
    return make_example([
        msg_sys(), msg_user(random.choice(queries)),
        msg_asst("", [tc("python_run", {"code": code})]),
        msg_tool("python_run", result),
        msg_asst(f"The result is: {result}"),
    ], sample_tools("python_run"))

def gen_single_web_search():
    user_q, search_q = random.choice(SEARCH_QUERIES)
    result = json.dumps({"results": [
        {"title": f"Guide: {search_q}", "url": "https://docs.example.com", "snippet": f"Comprehensive resource on {search_q}..."},
        {"title": f"{search_q.split()[0]} Documentation", "url": "https://example.com/docs", "snippet": f"Official docs for {search_q.split()[0]}..."},
    ]})
    return make_example([
        msg_sys(), msg_user(user_q),
        msg_asst("", [tc("web_search", {"query": search_q})]),
        msg_tool("web_search", result),
        msg_asst(f"I found relevant results about {search_q.split()[0]}."),
    ], sample_tools("web_search"))

def gen_single_web_fetch():
    query, url, content = random.choice(FETCH_URLS)
    return make_example([
        msg_sys(), msg_user(query),
        msg_asst("", [tc("web_fetch", {"url": url})]),
        msg_tool("web_fetch", content),
        msg_asst("Here's the content from that page."),
    ], sample_tools("web_fetch"))


# ─── Multi-turn generators ────────────────────────────────────

def gen_multi_glob_read():
    pat = random.choice(["**/*.py", "tests/test_*.py", "src/**/*.py"])
    files = gen_glob_out(pat)
    first = files.split("\n")[0]
    tools = sample_tools("glob")
    if not any(t["function"]["name"] == "file_read" for t in tools):
        tools.append(TOOL_BY_NAME["file_read"])
    return make_example([
        msg_sys(), msg_user("Find all Python files and show me the first one"),
        msg_asst("", [tc("glob", {"pattern": pat})]),
        msg_tool("glob", files),
        msg_asst(f"Found several files. Let me read `{first}`.", [tc("file_read", {"path": first})]),
        msg_tool("file_read", gen_code("func")),
        msg_asst(f"Here's `{first}`."),
    ], tools)

def gen_multi_read_edit():
    path = random.choice(["src/config.py", "config.yaml", "pyproject.toml"])
    tools = sample_tools("file_read")
    if not any(t["function"]["name"] == "file_edit" for t in tools):
        tools.append(TOOL_BY_NAME["file_edit"])
    return make_example([
        msg_sys(), msg_user(f"Read {path} and change the port to 3000"),
        msg_asst("Let me read it first.", [tc("file_read", {"path": path})]),
        msg_tool("file_read", gen_code("cfg")),
        msg_asst("Port is 8080. Changing it.", [tc("file_edit", {"path": path, "old_str": "8080", "new_str": "3000"})]),
        msg_tool("file_edit", f"Replaced in {path}"),
        msg_asst(f"Changed port from 8080 to 3000 in `{path}`."),
    ], tools)

def gen_multi_grep_read():
    desc, pat = random.choice(GREP_PATTERNS[:5])
    grep_out = gen_grep_out(desc)
    first = grep_out.split(":")[0]
    tools = sample_tools("grep")
    if not any(t["function"]["name"] == "file_read" for t in tools):
        tools.append(TOOL_BY_NAME["file_read"])
    return make_example([
        msg_sys(), msg_user(f"Find where {desc} is used and show me that file"),
        msg_asst("", [tc("grep", {"pattern": pat})]),
        msg_tool("grep", grep_out),
        msg_asst(f"Found in several files. Showing `{first}`.", [tc("file_read", {"path": first})]),
        msg_tool("file_read", gen_code("func")),
        msg_asst(f"Here's `{first}` where `{desc}` appears."),
    ], tools)

def gen_multi_test_fix():
    tools = sample_tools("bash")
    for n in ["file_read", "file_edit"]:
        if not any(t["function"]["name"] == n for t in tools):
            tools.append(TOOL_BY_NAME[n])
    return make_example([
        msg_sys(), msg_user("Run tests and fix any failures"),
        msg_asst("Running tests.", [tc("bash", {"command": "python3 -m pytest tests/ -v"})]),
        msg_tool("bash", gen_pytest(ok=False)),
        msg_asst("Test failing. Let me check the file.", [tc("file_read", {"path": "tests/test_auth.py"})]),
        msg_tool("file_read", gen_code("test")),
        msg_asst("Found the issue. Fixing it.", [tc("file_edit", {"path": "tests/test_auth.py", "old_str": 'authenticate("", "pass123")', "new_str": 'authenticate(None, "pass123")'})]),
        msg_tool("file_edit", "Replaced in tests/test_auth.py"),
        msg_asst("Fixed. Re-run tests to verify."),
    ], tools)


# ─── No-tool, Error recovery, Clarification ───────────────────

def gen_no_tool():
    q, a = random.choice(GENERAL_QA)
    return make_example([msg_sys(), msg_user(q), msg_asst(a)], sample_tools(k=5))

def gen_error_recovery():
    templates = [
        ("Read the config file", "file_read", {"path": "config.yaml"},
         "Error: File not found: config.yaml",
         "file_read", {"path": "config/settings.yaml"}, gen_code("cfg"),
         "Found it at `config/settings.yaml` instead."),
        ("Check port 8080", "bash", {"command": "lsof -i :8080"},
         "bash: lsof: command not found",
         "bash", {"command": "ss -tlnp | grep 8080"},
         "LISTEN  0  128  *:8080  *:*  users:((\"python3\",pid=1234,fd=5))",
         "Found it using ss — Python is listening on 8080."),
        ("Fix typo in README", "file_edit", {"path": "README.md", "old_str": "configuraton", "new_str": "configuration"},
         "Error: old_str not found in file.",
         "file_read", {"path": "README.md"},
         "# Project\n\nThis is the confguration guide.\n\n## Setup",
         "The actual typo is 'confguration'. Let me fix that."),
        ("Find API key definition", "grep", {"pattern": "API_KEY", "path": "src/"},
         "No matches found.",
         "grep", {"pattern": "api_key", "path": "."},
         ".env:3:api_key=sk-test-123\nconfig.yaml:8:api_key: ${API_KEY}",
         "Found it with lowercase search — defined in `.env` and referenced in `config.yaml`."),
    ]
    q, t1, a1, err, t2, a2, res, summary = random.choice(templates)
    tools = sample_tools(t1)
    if not any(t["function"]["name"] == t2 for t in tools):
        tools.append(TOOL_BY_NAME[t2])
    return make_example([
        msg_sys(), msg_user(q),
        msg_asst("", [tc(t1, a1)]), msg_tool(t1, err),
        msg_asst("Let me try a different approach.", [tc(t2, a2)]), msg_tool(t2, res),
        msg_asst(summary),
    ], tools)

def gen_clarification():
    templates = [
        ("Delete the file", "Which file should I delete?",
         "Delete tests/test_old.py", "bash", {"command": "rm tests/test_old.py"}, "Removed."),
        ("Edit the config", "Which config — config.yaml or src/config.py?",
         "The yaml one, set debug to true", "file_edit",
         {"path": "config.yaml", "old_str": "debug: false", "new_str": "debug: true"}, "Done."),
        ("Run it", "Run what — the app, tests, or a script?",
         "Run the tests", "bash", {"command": "python3 -m pytest tests/ -v"}, gen_pytest()),
        ("Search for that bug", "What should I search for? A function name or error message?",
         "The KeyError we keep seeing", "grep", {"pattern": "KeyError"}, gen_grep_out("KeyError")),
    ]
    q1, clarify, q2, tool, args, result = random.choice(templates)
    return make_example([
        msg_sys(), msg_user(q1), msg_asst(clarify), msg_user(q2),
        msg_asst("", [tc(tool, args)]), msg_tool(tool, result),
        msg_asst("Done."),
    ], sample_tools(tool))


# ─── Main ─────────────────────────────────────────────────────

SINGLE_GENS = [
    (gen_single_bash, 6), (gen_single_file_read, 4), (gen_single_file_write, 3),
    (gen_single_file_edit, 3), (gen_single_grep, 3), (gen_single_glob, 2),
    (gen_single_python, 3), (gen_single_web_search, 3), (gen_single_web_fetch, 1),
]

MULTI_GENS = [
    (gen_multi_glob_read, 3), (gen_multi_read_edit, 3),
    (gen_multi_grep_read, 3), (gen_multi_test_fix, 2),
]


def weighted_pick(gens):
    fns, wts = zip(*gens)
    return random.choices(fns, weights=wts, k=1)[0]


def generate_dataset(num: int, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    data = []
    counts = {"single": int(num*0.30), "single_result": int(num*0.20),
              "multi": int(num*0.15), "no_tool": int(num*0.15),
              "error": int(num*0.10), "clarify": int(num*0.10)}

    for _ in range(counts["single"]):
        data.append(weighted_pick(SINGLE_GENS)())
    for _ in range(counts["single_result"]):
        data.append(weighted_pick(SINGLE_GENS)())
    for _ in range(counts["multi"]):
        data.append(weighted_pick(MULTI_GENS)())
    for _ in range(counts["no_tool"]):
        data.append(gen_no_tool())
    for _ in range(counts["error"]):
        data.append(gen_error_recovery())
    for _ in range(counts["clarify"]):
        data.append(gen_clarification())

    random.shuffle(data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=15000)
    parser.add_argument("--output", type=str, default="data/eden_15k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.num} examples...")
    data = generate_dataset(args.num, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from collections import Counter
    cats, tools_used = Counter(), Counter()
    for item in data:
        msgs = item["messages"]
        has_tc = any(m.get("tool_calls") for m in msgs)
        has_err = any("Error" in m.get("content", "") or "not found" in m.get("content", "").lower()
                      for m in msgs if m["role"] == "tool")
        n_tc = sum(1 for m in msgs if m.get("tool_calls"))
        if not has_tc: cats["no_tool"] += 1
        elif has_err: cats["error_recovery"] += 1
        elif n_tc > 1: cats["multi_tool"] += 1
        else: cats["single_tool"] += 1
        for m in msgs:
            for t in m.get("tool_calls", []):
                tools_used[t["function"]["name"]] += 1

    print(f"\nWrote {len(data)} examples to {args.output}")
    print(f"\nCategories:")
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n} ({100*n/len(data):.1f}%)")
    print(f"\nTool usage:")
    for t, n in sorted(tools_used.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
