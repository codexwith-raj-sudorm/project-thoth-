"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .adapters import GeminiAdapter, OllamaAdapter, OpenAICompatibleAdapter
from .exceptions import MaxTurnsExceeded, StagnantLoopException, ThothError
from .locking import SingleInstanceLock
from .orchestrator import Orchestrator

NOTICE = """[Notice] Project Thoth uses an AST-based heuristic sandbox.
It is intended for trusted personal tasks, not for running
untrusted third-party prompts. For stronger isolation, use
containerized execution (Docker/Podman)."""


def _adapter(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaAdapter(args.model, args.base_url or "http://localhost:11434")
    if args.provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY is required for the groq provider")
        return OpenAICompatibleAdapter(
            args.model, args.base_url or "https://api.groq.com/openai/v1", key
        )
    if args.provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for the gemini provider")
        return GeminiAdapter(key, args.model)
    return OpenAICompatibleAdapter(args.model, args.base_url or "http://localhost:8000/v1", requests_per_minute=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thoth", description="Dual-agent Python synthesis and verification")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="synthesize and verify a Python script")
    run.add_argument("requirement")
    run.add_argument("--provider", choices=("ollama", "groq", "gemini", "vllm"), default="ollama")
    run.add_argument("--model", default="qwen2.5-coder:7b")
    run.add_argument("--verifier-model", help="optional separate verifier model")
    run.add_argument("--base-url")
    run.add_argument("--max-turns", type=int, default=3, choices=range(1, 6))
    run.add_argument("--allow-filesystem", action="store_true")
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("-o", "--output", type=Path, default=Path("thoth_output.py"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(NOTICE, file=sys.stderr)
    try:
        coder = _adapter(args)
        if args.verifier_model:
            original = args.model
            args.model = args.verifier_model
            verifier = _adapter(args)
            args.model = original
        else:
            verifier = _adapter(args)
        orchestrator = Orchestrator(
            coder,
            verifier,
            max_turns=args.max_turns,
            allow_filesystem=args.allow_filesystem,
            allow_network=args.allow_network,
        )
        with SingleInstanceLock():
            result = orchestrator.run(args.requirement)
        args.output.write_text(result.code, encoding="utf-8")
        print(f"APPROVED in {result.turns} turn(s): {args.output}")
        print(f"Session log: {result.log_path}")
        return 0
    except (MaxTurnsExceeded, StagnantLoopException) as exc:
        partial = args.output.with_suffix(".partial.py")
        if exc.partial_code:
            partial.write_text(exc.partial_code, encoding="utf-8")
            print(f"Best partial artifact: {partial}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ThothError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
