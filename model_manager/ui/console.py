"""Shared Rich Console singleton."""

from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "info":    "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "muted":   "dim white",
    "header":  "bold blue",
    "model":   "bold magenta",
})

console = Console(theme=THEME, highlight=True)
err_console = Console(stderr=True, theme=THEME)
