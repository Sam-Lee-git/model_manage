#!/usr/bin/env bash
# install.sh — install model-manager and make `mm` available in PATH permanently
# Usage: bash install.sh
# Run from the repo root directory.

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing model-manager from ${REPO_DIR}"
pip install -e "${REPO_DIR}" --quiet

# Find where pip installed the mm script
SCRIPTS_DIR="$(python -c "import sysconfig; print(sysconfig.get_path('scripts'))")"
MM_SCRIPT="${SCRIPTS_DIR}/mm"

if [ ! -f "${MM_SCRIPT}" ]; then
    echo "ERROR: mm not found at ${MM_SCRIPT}" >&2
    exit 1
fi

# If mm is already reachable in PATH, nothing to do
if command -v mm &>/dev/null && [ "$(command -v mm)" = "${MM_SCRIPT}" ]; then
    echo "==> mm is already in PATH: $(command -v mm)"
    mm --version
    exit 0
fi

# Try to symlink into /usr/local/bin (available to all users, no relogin needed)
if [ -w /usr/local/bin ]; then
    ln -sf "${MM_SCRIPT}" /usr/local/bin/mm
    echo "==> Linked: /usr/local/bin/mm -> ${MM_SCRIPT}"
elif sudo -n true 2>/dev/null; then
    sudo ln -sf "${MM_SCRIPT}" /usr/local/bin/mm
    echo "==> Linked (sudo): /usr/local/bin/mm -> ${MM_SCRIPT}"
else
    # No write access to /usr/local/bin — use ~/.local/bin
    mkdir -p "${HOME}/.local/bin"
    ln -sf "${MM_SCRIPT}" "${HOME}/.local/bin/mm"
    echo "==> Linked: ~/.local/bin/mm -> ${MM_SCRIPT}"

    LOCAL_BIN="${HOME}/.local/bin"

    # Add to current session immediately
    export PATH="${LOCAL_BIN}:${PATH}"

    # Persist across new shells
    for RC in "${HOME}/.bashrc" "${HOME}/.zshrc" "${HOME}/.profile"; do
        if [ -f "${RC}" ] && ! grep -q "${LOCAL_BIN}" "${RC}" 2>/dev/null; then
            printf '\n# Added by model-manager install.sh\nexport PATH="%s:$PATH"\n' "${LOCAL_BIN}" >> "${RC}"
            echo "==> Persisted PATH in ${RC}"
        fi
    done
fi

echo ""
echo "==> Done. Verifying..."
mm --version
echo ""
echo "You can now run: mm"
echo "(New terminals will also have mm available — no need to rerun this script)"
