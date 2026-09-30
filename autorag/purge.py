#!/usr/bin/env python3
"""Purge exactly one AutoRAG conversation from the local SQLite database.

Examples:
  python scripts/purge_conversation.py fp_bb11ae3ba671c312
  python scripts/purge_conversation.py fp_bb11ae3ba671c312 --db C:/path/state.db
  python scripts/purge_conversation.py fp_bb11ae3ba671c312 --yes
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import AutoRAGConfig
from .database import StateDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="Purge one AutoRAG conversation; other conversations are untouched.")
    parser.add_argument("conversation_id")
    parser.add_argument("--db", type=Path, help="SQLite path; defaults to AutoRAG's configured database")
    parser.add_argument("--keep-conversation", action="store_true", help="Delete its state rows but retain the conversation metadata row")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = parser.parse_args()

    db_path = args.db or AutoRAGConfig.load().resolved_db_path()
    print(f"Database: {db_path}")
    print(f"Conversation: {args.conversation_id}")
    if not args.yes:
        answer = input("This permanently deletes this conversation's AutoRAG memory. Type PURGE to continue: ")
        if answer != "PURGE":
            print("Cancelled.")
            return 1

    db = StateDatabase(db_path)
    deleted = db.purge_conversation(args.conversation_id, delete_conversation=not args.keep_conversation)
    for table, count in deleted.items():
        print(f"Deleted {count} rows from {table}")
    print("Database purged for this conversation.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
