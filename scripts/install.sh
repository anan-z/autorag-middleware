#!/usr/bin/env bash
# Unix installer wrapper for AutoRAG Middleware
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "AutoRAG Middleware – Unix installer"
echo "  Root: $ROOT"

# Prefer python3
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python 3.10+ is required." >&2
  exit 1
fi

# Optional: create venv if none active
if [[ -z "${VIRTUAL_ENV:-}" && ! -d "$ROOT/venv" ]]; then
  echo "Creating virtual environment at $ROOT/venv …"
  "$PYTHON" -m venv venv
fi

if [[ -d "$ROOT/venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
  PYTHON=python
fi

EXTRA_ARGS=()
for arg in "$@"; do
  EXTRA_ARGS+=("$arg")
done

"$PYTHON" "$ROOT/scripts/install.py" "${EXTRA_ARGS[@]}"

echo
echo "To activate the venv later:"
echo "  source $ROOT/venv/bin/activate"
echo "  python -m autorag"
