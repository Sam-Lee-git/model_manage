#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/venv/bin/python" -m model_manager "$@"
