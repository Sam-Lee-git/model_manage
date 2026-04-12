"""Argument parser for the `mm` command (single entry point, no subcommands)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from model_manager import __version__

_PROVIDERS = ["claude", "openai", "gemini", "qwen", "minimax", "deepseek"]


@dataclass
class CLIArgs:
    resume: Optional[str]
    api_key: Optional[str]
    provider: Optional[str]
    llm_model: Optional[str]    # LLM model name (not the AI model to deploy)
    deploy_model: Optional[str] # AI model to deploy (was --model before)
    backend: Optional[str]
    path: Optional[str]
    list_sessions: bool
    version: bool


def parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="mm",
        description=(
            "mm — AI Model Manager\n"
            "Deploy AI models interactively on any platform.\n\n"
            "Run without arguments to start an interactive session."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--resume", metavar="SESSION_ID",
        help="Resume an interrupted installation session",
    )
    parser.add_argument(
        "--api-key", metavar="KEY", dest="api_key",
        help="API key for the selected LLM provider",
    )
    parser.add_argument(
        "--provider", choices=_PROVIDERS, metavar="PROVIDER",
        help=(
            f"LLM provider for the agent brain. One of: {', '.join(_PROVIDERS)}. "
            "Default: claude. Keys read from env: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "GEMINI_API_KEY, QWEN_API_KEY, MINIMAX_API_KEY, DEEPSEEK_API_KEY."
        ),
    )
    parser.add_argument(
        "--llm-model", metavar="MODEL_ID", dest="llm_model",
        help=(
            "Override the LLM model ID used for the agent "
            "(e.g. gpt-4o-mini, qwen-plus, deepseek-reasoner)"
        ),
    )
    parser.add_argument(
        "--deploy-model", metavar="MODEL_ID", dest="deploy_model",
        help="Skip recommendation and deploy this AI model directly",
    )
    parser.add_argument(
        "--backend", choices=["pip_venv", "conda", "docker"],
        help="Force a specific installation backend",
    )
    parser.add_argument(
        "--path", metavar="DIR",
        help="Installation directory (default: auto-selected)",
    )
    parser.add_argument(
        "--list-sessions", action="store_true", dest="list_sessions",
        help="List past sessions and exit",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print version and exit",
    )

    ns = parser.parse_args(argv)
    return CLIArgs(
        resume=ns.resume,
        api_key=ns.api_key,
        provider=ns.provider,
        llm_model=ns.llm_model,
        deploy_model=ns.deploy_model,
        backend=ns.backend,
        path=ns.path,
        list_sessions=ns.list_sessions,
        version=ns.version,
    )
