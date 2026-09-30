"""CLI entry: python -m autorag  |  autorag"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from . import __version__
from .config import AutoRAGConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="autorag",
        description=f"AutoRAG Middleware {__version__} – conversation-scoped memory proxy",
    )
    parser.add_argument("--host", default=None, help="Bind host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default from config)")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config:
        os.environ["AUTORAG_CONFIG"] = str(Path(args.config).resolve())

    config = AutoRAGConfig.load(os.environ.get("AUTORAG_CONFIG") or args.config)
    host = args.host or config.host
    port = args.port or config.port

    print(f"AutoRAG Middleware v{__version__}")
    print(f"  Listening : http://{host}:{port}")
    print(f"  Database  : {config.resolved_db_path()}")
    print(f"  Backends  : {', '.join(config.backends.keys()) or '(none)'}")
    print(f"  Extract   : {config.extraction.mode}")
    print(f"  Conflict  : {config.validation.on_conflict}")
    if args.config:
        print(f"  Config    : {args.config}")
    print("Point your client at the URL above + /v1")
    print("Optional header: X-Conversation-Id: <id>  (isolates memory per chat)")
    print()

    uvicorn.run(
        "autorag.proxy:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    main()
