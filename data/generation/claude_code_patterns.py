"""
Extract Claude Code's tool-calling patterns for Eden training data.

Claude Code's internal system prompt and tool format are the gold standard
for how a coding agent should behave. We reverse-engineer these patterns
from eden/claude-code-ref/ and public documentation to ensure Eden-1B
learns the EXACT same behavioral patterns.

This does NOT copy any Anthropic code — it captures observable behavior
patterns that are fair game for training data.
"""

# ── Claude Code behavioral patterns to encode in training data ──

CLAUDE_CODE_PATTERNS = {
    # 1. Tool selection logic
    "tool_selection": {
        "rules": [
            "Use glob/grep to explore before modifying",
            "Read a file before editing it (verify current state)",
            "Use bash for system operations, not python_exec",
            "Prefer file_edit over file_write for modifications",
            "Use grep to find before sed/replace",
            "Never write a file you haven't read first",
        ],
        "training_implications": """
            Generate examples where the model reads before writing,
            explores before modifying, and uses the simplest tool
            for the job.
        """,
    },

    # 2. Multi-step patterns
    "multi_step": {
        "patterns": [
            "Find → Read → Edit → Verify (4-step file modification)",
            "Search → Read → Understand → Respond (codebase exploration)",
            "Read → Plan → Execute → Test (code changes)",
            "Error → Diagnose → Fix → Verify (debugging)",
        ],
        "training_implications": """
            Generate multi-turn examples that follow these patterns.
            The model should learn the SEQUENCE of operations,
            not just individual tool calls.
        """,
    },

    # 3. Error handling
    "error_handling": {
        "patterns": [
            "File not found → try alternative path or ask user",
            "Command failed → read error, try different approach",
            "Permission denied → explain, suggest fix",
            "Timeout → break into smaller operations",
            "Empty result → broaden search, try different tool",
        ],
        "training_implications": """
            Generate error scenarios where the model receives a
            tool error and gracefully recovers.
        """,
    },

    # 4. Permission model
    "permissions": {
        "safe_auto": [
            "file_read", "glob", "grep", "web_search",
            "todo", "bash (read-only like ls, cat, git status)",
        ],
        "ask_first": [
            "file_write (new files)",
            "file_edit (modifying existing)",
            "bash (write commands like rm, mv, git commit)",
            "python_exec (running code)",
        ],
        "always_block": [
            "rm -rf /", "sudo rm", "chmod 777 /",
            "curl | bash", "> /dev/sda",
        ],
        "training_implications": """
            Training data should include examples where the model:
            - Auto-executes safe tools without asking
            - Asks permission for write operations
            - Refuses dangerous commands outright
        """,
    },

    # 5. Response format
    "response_format": {
        "rules": [
            "After tool results: summarize concisely, don't repeat raw output",
            "For file reads: highlight relevant sections, not dump everything",
            "For errors: explain what went wrong AND suggest next step",
            "For multi-tool: explain reasoning between each tool call",
            "For code changes: show diff or describe what changed",
            "Keep summaries under 200 words unless user asks for detail",
        ],
    },

    # 6. Context awareness
    "context_awareness": {
        "patterns": [
            "Detect project type from files (package.json = Node, pyproject.toml = Python)",
            "Read CLAUDE.md/AGENTS.md for project-specific instructions",
            "Remember previous tool results in conversation",
            "Reference file contents by name after reading them",
            "Track which files have been modified in the session",
        ],
    },

    # 7. The tool call format (what the model outputs)
    "tool_call_format": {
        "single": '<tool_call>{"name": "bash", "input": {"command": "ls -la"}}</tool_call>',
        "with_text": 'Let me check what files are here.\n<tool_call>{"name": "bash", "input": {"command": "ls -la"}}</tool_call>',
        "parallel": '<tool_call>{"name": "glob", "input": {"pattern": "*.py"}}</tool_call>\n<tool_call>{"name": "glob", "input": {"pattern": "*.js"}}</tool_call>',
        "after_result": "Based on the file listing, I can see there are 12 Python files...",
    },
}


def generate_claude_code_aligned_prompt(tool_name, scenario_type):
    """Generate a system prompt that teaches Claude Code behavioral patterns."""
    return f"""You are Eden, a local AI coding assistant running on Apple Silicon.

You have access to tools that you call using <tool_call> tags.

BEHAVIORAL RULES (match Claude Code patterns):
1. EXPLORE before MODIFY — always read/glob/grep before writing
2. ONE TOOL AT A TIME unless tools are truly independent  
3. SUMMARIZE tool results — don't dump raw output
4. ASK before destructive operations (delete, overwrite)
5. REFUSE dangerous commands (rm -rf /, sudo rm, etc.)
6. RECOVER from errors — try alternative approaches
7. REMEMBER context — reference earlier results by name

Tool call format:
<tool_call>{{"name": "{tool_name}", "input": {{...}}}}</tool_call>

After receiving a <tool_result>, summarize it helpfully and decide if more tools are needed."""


# ── Generate training examples aligned with Claude Code patterns ──

def build_explore_before_modify_examples():
    """Examples where the model reads before writing (Claude Code pattern #1)."""
    return [
        {
            "category": "explore_before_modify",
            "messages": [
                {"role": "user", "content": "Fix the bug in the login function"},
                {"role": "assistant", "content": 'Let me find the login function first.\n<tool_call>{"name": "grep", "input": {"pattern": "def login", "include": "*.py"}}</tool_call>'},
                {"role": "tool", "name": "grep", "content": "src/auth.py:42: def login(username, password):"},
                {"role": "assistant", "content": '<tool_call>{"name": "file_read", "input": {"path": "src/auth.py", "offset": 35, "limit": 30}}</tool_call>'},
                {"role": "tool", "name": "file_read", "content": "def login(username, password):\n    user = db.query(User).filter(name=username).first()\n    if user.check_password(password):\n        return create_token(user)\n    return None"},
                {"role": "assistant", "content": "I can see the issue — there's no null check on the user query. If the username doesn't exist, `user.check_password()` will throw an AttributeError.\n\n<tool_call>{\"name\": \"file_edit\", \"input\": {\"path\": \"src/auth.py\", \"old_str\": \"    user = db.query(User).filter(name=username).first()\\n    if user.check_password(password):\", \"new_str\": \"    user = db.query(User).filter(name=username).first()\\n    if user and user.check_password(password):\"}}</tool_call>"},
                {"role": "tool", "name": "file_edit", "content": "OK — replaced 1 occurrence in src/auth.py"},
                {"role": "assistant", "content": "Fixed. Added a null check so `login()` returns `None` instead of crashing when the username doesn't exist."},
            ],
        },
    ]


def build_error_recovery_examples():
    """Examples where tools fail and the model recovers (Claude Code pattern)."""
    return [
        {
            "category": "error_recovery",
            "messages": [
                {"role": "user", "content": "Read the config file"},
                {"role": "assistant", "content": '<tool_call>{"name": "file_read", "input": {"path": "config.yaml"}}</tool_call>'},
                {"role": "tool", "name": "file_read", "content": "Error: file not found: config.yaml"},
                {"role": "assistant", "content": 'Config file not at the expected path. Let me search for it.\n<tool_call>{"name": "glob", "input": {"pattern": "**/*.{yaml,yml,json,toml}"}}</tool_call>'},
                {"role": "tool", "name": "glob", "content": "settings/config.toml\n.env"},
                {"role": "assistant", "content": '<tool_call>{"name": "file_read", "input": {"path": "settings/config.toml"}}</tool_call>'},
                {"role": "tool", "name": "file_read", "content": "[database]\nhost = \"localhost\"\nport = 5432\nname = \"myapp\""},
                {"role": "assistant", "content": "Found the config at `settings/config.toml` (not `config.yaml`). It's a TOML file with database settings: PostgreSQL on localhost:5432, database name \"myapp\"."},
            ],
        },
    ]


if __name__ == "__main__":
    # Print patterns for review
    for name, pattern in CLAUDE_CODE_PATTERNS.items():
        print(f"\n=== {name} ===")
        if isinstance(pattern, dict):
            for k, v in pattern.items():
                if isinstance(v, list):
                    for item in v:
                        print(f"  - {item}")
                else:
                    print(f"  {k}: {v}")
