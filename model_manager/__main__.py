"""Entry point: `python -m model_manager` or the `mm` script."""

from __future__ import annotations

import asyncio
import os
import sys

from model_manager import __version__
from model_manager.cli import parse_args
from model_manager.config.paths import ensure_dirs
from model_manager.ui.console import console

_PROVIDER_KEY_ENV: dict[str, str] = {
    "claude":   "ANTHROPIC_API_KEY",
    "openai":   "OPENAI_API_KEY",
    "gemini":   "GEMINI_API_KEY",
    "qwen":     "QWEN_API_KEY",
    "minimax":  "MINIMAX_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.version:
        console.print(f"mm version {__version__}")
        return

    if args.list_sessions:
        ensure_dirs()
        from model_manager.state.store import StateStore
        store = StateStore()
        sessions = store.list_sessions()
        if not sessions:
            console.print("[muted]No sessions found.[/muted]")
            return
        from rich.table import Table
        table = Table(title="Past Sessions")
        table.add_column("Session ID", style="info")
        table.add_column("State")
        table.add_column("Model", style="model")
        table.add_column("Updated")
        for s in sessions:
            table.add_row(
                s["session_id"][:12],
                s["current_state"],
                s["model"] or "—",
                s["updated_at"][:19] if s["updated_at"] else "—",
            )
        console.print(table)
        return

    # ── Inject provider/key into env before Settings is first read ────────────
    if args.provider:
        os.environ["MM_LLM_PROVIDER"] = args.provider

    if args.api_key and args.provider:
        env_key = _PROVIDER_KEY_ENV.get(args.provider, "ANTHROPIC_API_KEY")
        os.environ[env_key] = args.api_key
    elif args.api_key:
        # No provider specified — guess from current setting or default to claude
        provider = os.environ.get("MM_LLM_PROVIDER", "claude")
        env_key  = _PROVIDER_KEY_ENV.get(provider, "ANTHROPIC_API_KEY")
        os.environ[env_key] = args.api_key

    if args.llm_model and args.provider:
        os.environ[f"MM_{args.provider.upper()}_MODEL"] = args.llm_model

    # Force settings re-read after env injection
    from model_manager.config.settings import reset_settings
    reset_settings()

    # ── Detect provider and print banner ─────────────────────────────────────
    from model_manager.config.settings import get_settings
    from model_manager.agent.factory import detect_available_providers, PROVIDER_ENV_VARS, LLMProvider

    settings = get_settings()
    available = detect_available_providers()

    if not available:
        console.print(f"[header]mm — AI Model Manager[/header]  [muted]v{__version__}[/muted]")
        console.print("[warning]No LLM API key found in environment variables.[/warning]")
        console.print("[muted]Set one of:[/muted]")
        for env_var in PROVIDER_ENV_VARS.values():
            console.print(f"  [muted]{env_var}[/muted]")
        console.print("[muted](Continuing with simple mode — no AI assistance)[/muted]\n")
    else:
        active_provider = available[0][0]
        all_names = ", ".join(p.value for p, _ in available)
        console.print(
            f"[header]mm — AI Model Manager[/header]  [muted]v{__version__}[/muted]  "
            f"[info]{active_provider.value}[/info]"
        )
        if len(available) > 1:
            console.print(f"[muted]Detected providers: {all_names}. Using {active_provider.value}.[/muted]")
            console.print(f"[muted]Switch with: mm --provider <name>  or  MM_LLM_PROVIDER=<name>[/muted]")
    console.print("[muted]Type /help for commands, /exit to quit.[/muted]\n")

    from model_manager.app import App
    app = App(
        resume_session_id=args.resume,
        force_backend=args.backend,
        force_path=args.path,
        force_model=args.deploy_model,
    )

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        console.print("\n[muted]Goodbye.[/muted]")
        sys.exit(0)


if __name__ == "__main__":
    main()
