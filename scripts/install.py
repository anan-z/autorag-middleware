#!/usr/bin/env python3
"""Cross-platform installer for AutoRAG Middleware.

Usage:
    python scripts/install.py
    python scripts/install.py --ner   # also install spaCy + model
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import platformdirs
except ImportError:
    platformdirs = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> None:
    print(f"  → {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def data_dir() -> Path:
    if platformdirs is not None:
        d = Path(platformdirs.user_data_dir("autorag-middleware"))
    else:
        d = Path.home() / ".local" / "share" / "autorag-middleware"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_dir() -> Path:
    if platformdirs is not None:
        d = Path(platformdirs.user_config_dir("autorag-middleware"))
    else:
        d = Path.home() / ".config" / "autorag-middleware"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_default_config(path: Path) -> None:
    if path.exists():
        print(f"  Config already exists: {path}")
        return
    src = ROOT / "configs" / "default.yaml"
    if src.is_file():
        shutil.copy(src, path)
    else:
        path.write_text(
            """host: "0.0.0.0"
port: 8000
extraction:
  enabled: true
  use_spacy: false
validation:
  enabled: false
backends:
  default:
    api_base: "http://127.0.0.1:1234/v1"
    api_key: "not-needed"
    model: "local-model"
  lmstudio:
    api_base: "http://127.0.0.1:1234/v1"
    api_key: "not-needed"
    model: "local-model"
  ollama:
    api_base: "http://127.0.0.1:11434/v1"
    api_key: "not-needed"
    model: "llama3.2"
""",
            encoding="utf-8",
        )
    print(f"  Wrote default config: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install AutoRAG Middleware")
    parser.add_argument("--ner", action="store_true", help="Install spaCy + en_core_web_sm")
    parser.add_argument("--user", action="store_true", help="pip install --user")
    args = parser.parse_args()

    print("Installing AutoRAG Middleware…")
    print(f"  Project root: {ROOT}")

    pip = [sys.executable, "-m", "pip", "install"]
    if args.user:
        pip.append("--user")

    # Editable install of this package
    extras = "[ner]" if args.ner else ""
    run(pip + ["-e", f"{ROOT}{extras}"])

    if args.ner:
        print("  Downloading spaCy model en_core_web_sm…")
        try:
            run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        except subprocess.CalledProcessError:
            print("  WARNING: spaCy model download failed. Run manually:")
            print("    python -m spacy download en_core_web_sm")

    ddir = data_dir()
    cdir = config_dir()
    write_default_config(cdir / "config.yaml")

    # Convenience launcher on Unix
    if sys.platform != "win32":
        launcher = ddir / "run.sh"
        launcher.write_text(
            f"""#!/bin/bash
exec "{sys.executable}" -m autorag "$@"
""",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        print(f"  Launcher: {launcher}")

    print()
    print("Installation complete.")
    print(f"  Config  : {cdir / 'config.yaml'}")
    print(f"  Data dir: {ddir}")
    print()
    print("Start the proxy:")
    print("  python -m autorag")
    print("  # or: autorag")
    print()
    print("Then point your client at http://127.0.0.1:8000/v1")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"Install failed with exit code {e.returncode}", file=sys.stderr)
        raise SystemExit(e.returncode)
