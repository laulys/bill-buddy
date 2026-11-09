"""Command-line chat with Bill Buddy.

    python -m bill_buddy                                         # chat
    python -m bill_buddy "How much should I tip on $60 at 20%?"  # one question
    python -m bill_buddy --quiet "..."                           # hide the tool calls
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    import readline  # noqa: F401  enables arrow keys, history and Ctrl+L in the chat prompt
except ImportError:  # not available on Windows
    pass

from dotenv import load_dotenv

from .agent import BillBuddyAgent
from .llm import ConfigError, get_llm


def for_terminal(text: str) -> str:
    """Drop Markdown markers terminals show literally: **bold** and ### headings."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.replace("**", "")


def shorten(text: str, limit: int = 150) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_step(name: str, args: dict, result) -> None:
    arguments = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    print(f"  🔧 {name}({arguments})")
    print(f"     → {shorten(json.dumps(result, ensure_ascii=False))}")


def model_error_hint(exc: Exception) -> str:
    if os.getenv("LLM_PROVIDER", "ollama").lower() != "ollama":
        return "Check your API key and model name in .env."
    if "not found" in str(exc):
        return f"Download the model first: ollama pull {os.getenv('OLLAMA_MODEL', 'qwen3:8b')}"
    return "Is Ollama running? Start it with: ollama serve"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bill_buddy",
        description="Split bills, calculate tips and convert currencies by chatting.",
    )
    parser.add_argument("question", nargs="*", help="ask one question and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="don't show tool calls")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        agent = BillBuddyAgent(get_llm(), on_step=None if args.quiet else print_step)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    def safe_ask(question: str):
        try:
            return agent.ask(question)
        except Exception as exc:  # e.g. Ollama not running, invalid API key
            print(f"Error talking to the model: {exc}\n{model_error_hint(exc)}", file=sys.stderr)
            return None

    if args.question:
        answer = safe_ask(" ".join(args.question))
        if answer is None:
            return 1
        print(for_terminal(answer))
        return 0

    print("💸 Bill Buddy · split bills, tips and currencies (type 'exit' to quit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "salir"}:
            return 0
        if question:
            answer = safe_ask(question)
            if answer is not None:
                print(f"\nBill Buddy: {for_terminal(answer)}\n")


if __name__ == "__main__":
    sys.exit(main())
