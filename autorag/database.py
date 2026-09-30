"""SQLite state store: conversation-scoped entities + structured facts."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sqlite_vec
except ImportError:  # pragma: no cover
    sqlite_vec = None  # type: ignore


EMBEDDING_DIM = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_token_hash(tok: str) -> int:
    """Deterministic across process restarts (unlike Python's randomized hash())."""
    import hashlib
    digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _simple_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Stable hashed bag-of-words vector. Not semantic; safe across restarts."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vec.tolist()
    for i, tok in enumerate(tokens):
        h = _stable_token_hash(tok) % dim
        vec[h] += 1.0 + 0.1 * (i % 5)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


def _serialize_f32(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class StateDatabase:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vec_available = False
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if sqlite_vec is not None:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._vec_available = True
            except Exception:
                self._vec_available = False
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    name TEXT NOT NULL COLLATE NOCASE,
                    type TEXT,
                    attributes TEXT DEFAULT '{}',
                    first_seen TEXT,
                    last_seen TEXT,
                    mention_count INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    UNIQUE(conversation_id, name),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    subject TEXT NOT NULL COLLATE NOCASE,
                    predicate TEXT NOT NULL COLLATE NOCASE,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.7,
                    turn_number INTEGER,
                    source TEXT DEFAULT 'heuristic',
                    active INTEGER DEFAULT 1,
                    superseded_by INTEGER,
                    created_at TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY (superseded_by) REFERENCES facts(id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    turn_number INTEGER,
                    description TEXT,
                    payload TEXT DEFAULT '{}',
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS conversation_state (
                    conversation_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (conversation_id, key)
                );

                CREATE INDEX IF NOT EXISTS idx_entities_conv ON entities(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(conversation_id, name);
                CREATE INDEX IF NOT EXISTS idx_facts_conv ON facts(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_facts_lookup ON facts(conversation_id, subject, predicate, active);
                CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(conversation_id, subject);
                """
            )
            if self._vec_available:
                try:
                    conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS fact_vectors
                        USING vec0(fact_id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIM}])
                        """
                    )
                except sqlite3.OperationalError:
                    self._vec_available = False

    # ------------------------------------------------------------------ conversations

    def ensure_conversation(self, conversation_id: str, title: str | None = None) -> str:
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, title or conversation_id, now, now),
                )
        return conversation_id

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ entities

    def upsert_entity(
        self,
        conversation_id: str,
        name: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        confidence: float = 0.5,
    ) -> int:
        self.ensure_conversation(conversation_id)
        attributes = attributes or {}
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, attributes, mention_count, confidence FROM entities
                WHERE conversation_id = ? AND name = ? COLLATE NOCASE
                """,
                (conversation_id, name),
            ).fetchone()

            if row:
                entity_id = row["id"]
                existing = json.loads(row["attributes"] or "{}")
                existing.update({k: v for k, v in attributes.items() if v is not None})
                new_conf = min(1.0, (row["confidence"] + confidence) / 2 + 0.05)
                conn.execute(
                    """
                    UPDATE entities
                    SET attributes = ?, last_seen = ?, mention_count = ?, confidence = ?,
                        type = COALESCE(?, type)
                    WHERE id = ?
                    """,
                    (json.dumps(existing), now, row["mention_count"] + 1, new_conf, entity_type, entity_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO entities
                    (conversation_id, name, type, attributes, first_seen, last_seen, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (conversation_id, name, entity_type, json.dumps(attributes), now, now, confidence),
                )
                entity_id = int(cur.lastrowid)
            return entity_id

    def get_entity(self, conversation_id: str, name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, type, attributes, confidence, mention_count FROM entities
                WHERE conversation_id = ? AND name = ? COLLATE NOCASE
                """,
                (conversation_id, name),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "type": row["type"],
                "attributes": json.loads(row["attributes"] or "{}"),
                "confidence": row["confidence"],
                "mention_count": row["mention_count"],
            }

    def search_entities(
        self, conversation_id: str, query: str = "", limit: int = 8
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if not query.strip():
                rows = conn.execute(
                    """
                    SELECT name, type, attributes, confidence, mention_count
                    FROM entities WHERE conversation_id = ?
                    ORDER BY last_seen DESC LIMIT ?
                    """,
                    (conversation_id, limit),
                ).fetchall()
            else:
                q = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT name, type, attributes, confidence, mention_count
                    FROM entities
                    WHERE conversation_id = ? AND (name LIKE ? OR attributes LIKE ?)
                    ORDER BY mention_count DESC, last_seen DESC LIMIT ?
                    """,
                    (conversation_id, q, q, limit),
                ).fetchall()
            return [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "attributes": json.loads(r["attributes"] or "{}"),
                    "confidence": r["confidence"],
                    "mention_count": r["mention_count"],
                }
                for r in rows
            ]

    def get_recent_entities(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.search_entities(conversation_id, "", limit=limit)

    # ------------------------------------------------------------------ facts (structured memory)

    def get_active_fact(
        self, conversation_id: str, subject: str, predicate: str
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, subject, predicate, object, confidence, turn_number, source, created_at
                FROM facts
                WHERE conversation_id = ? AND subject = ? COLLATE NOCASE
                  AND predicate = ? COLLATE NOCASE AND active = 1
                ORDER BY id DESC LIMIT 1
                """,
                (conversation_id, subject, predicate),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def commit_fact(
        self,
        conversation_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.7,
        turn_number: int = 0,
        source: str = "heuristic",
        supersede_existing: bool = True,
    ) -> dict[str, Any]:
        """Insert a fact. If an active fact with same subject+predicate exists and
        object differs, deactivate it (supersede) when supersede_existing is True.
        Returns {id, status: 'created'|'unchanged'|'superseded', previous?}
        """
        self.ensure_conversation(conversation_id)
        subject = subject.strip()
        predicate = predicate.strip().lower().replace(" ", "_")
        obj = str(obj).strip()
        existing = self.get_active_fact(conversation_id, subject, predicate)

        if existing and existing["object"].strip().lower() == obj.lower():
            # Same fact — bump confidence lightly
            with self._connect() as conn:
                new_conf = min(1.0, max(existing["confidence"], confidence))
                conn.execute(
                    "UPDATE facts SET confidence = ? WHERE id = ?",
                    (new_conf, existing["id"]),
                )
            return {"id": existing["id"], "status": "unchanged", "fact": existing}

        previous = None
        with self._connect() as conn:
            if existing and supersede_existing:
                previous = existing
                # will set superseded_by after insert
            cur = conn.execute(
                """
                INSERT INTO facts
                (conversation_id, subject, predicate, object, confidence, turn_number, source, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (conversation_id, subject, predicate, obj, confidence, turn_number, source, _now()),
            )
            new_id = int(cur.lastrowid)

            if existing and supersede_existing:
                conn.execute(
                    "UPDATE facts SET active = 0, superseded_by = ? WHERE id = ?",
                    (new_id, existing["id"]),
                )

            # Vector index
            if self._vec_available:
                text = f"{subject} {predicate} {obj}"
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO fact_vectors(fact_id, embedding) VALUES (?, ?)",
                        (new_id, _serialize_f32(_simple_embed(text))),
                    )
                except Exception:
                    pass

        status = "superseded" if previous else "created"
        fact = {
            "id": new_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "turn_number": turn_number,
            "source": source,
        }
        return {"id": new_id, "status": status, "fact": fact, "previous": previous}


    def max_turn(self, conversation_id: str) -> int:
        """Highest turn_number seen for facts or events in this conversation."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MAX(t) AS m FROM (
                    SELECT MAX(turn_number) AS t FROM facts WHERE conversation_id = ?
                    UNION ALL
                    SELECT MAX(turn_number) AS t FROM events WHERE conversation_id = ?
                )
                """,
                (conversation_id, conversation_id),
            ).fetchone()
            return int(row["m"] or 0) if row and row["m"] is not None else 0

    def deactivate_facts_from_turn(self, conversation_id: str, from_turn: int) -> int:
        """Roll back facts at or after from_turn (regenerate / swipe).

        Deactivates those facts and reactivates any fact they had superseded,
        so prior canon is restored when the user discards a reply.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, superseded_by FROM facts
                WHERE conversation_id = ? AND turn_number >= ?
                """,
                (conversation_id, from_turn),
            ).fetchall()
            # Map: new_id that superseded old → we need old ids whose superseded_by is in this set
            new_ids = [r["id"] for r in rows]
            if not new_ids:
                return 0
            placeholders = ",".join("?" * len(new_ids))
            # Reactivate predecessors
            conn.execute(
                f"""
                UPDATE facts SET active = 1, superseded_by = NULL
                WHERE conversation_id = ? AND superseded_by IN ({placeholders})
                """,
                [conversation_id, *new_ids],
            )
            cur = conn.execute(
                f"""
                UPDATE facts SET active = 0
                WHERE conversation_id = ? AND id IN ({placeholders})
                """,
                [conversation_id, *new_ids],
            )
            return int(cur.rowcount or 0)

    def list_active_facts(
        self, conversation_id: str, subject: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if subject:
                rows = conn.execute(
                    """
                    SELECT id, subject, predicate, object, confidence, turn_number, source, created_at
                    FROM facts
                    WHERE conversation_id = ? AND active = 1 AND subject = ? COLLATE NOCASE
                    ORDER BY id DESC LIMIT ?
                    """,
                    (conversation_id, subject, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, subject, predicate, object, confidence, turn_number, source, created_at
                    FROM facts
                    WHERE conversation_id = ? AND active = 1
                    ORDER BY id DESC LIMIT ?
                    """,
                    (conversation_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def search_facts(
        self, conversation_id: str, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Score active facts by token overlap with the query (conversation-scoped)."""
        q = (query or "").strip().lower()
        if not q:
            return self.list_active_facts(conversation_id, limit=limit)

        tokens = [t for t in re.findall(r"[a-z0-9_]{2,}", q) if t not in {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "has", "have", "been", "were",
            "what", "when", "where", "who", "why", "how", "did", "does", "this",
            "that", "with", "from", "your", "about", "into", "just", "like",
        }]
        facts = self.list_active_facts(conversation_id, limit=200)
        if not facts:
            return []

        scored: list[tuple[float, dict]] = []
        for f in facts:
            blob = f"{f['subject']} {f['predicate']} {f['object']}".lower()
            score = 0.0
            for t in tokens:
                if t in blob:
                    score += 1.0
                # partial subject match
                if t in f["subject"].lower():
                    score += 0.5
            if score <= 0:
                continue
            conf = float(f.get("confidence") or 0.5)
            scored.append((score + 0.1 * conf, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Drop weak matches if we have stronger ones
        if scored and scored[0][0] >= 1.0:
            scored = [s for s in scored if s[0] >= 1.0]
        return [f for _, f in scored[:limit]]

    def set_fact_kv(self, conversation_id: str, key: str, value: str) -> None:
        self.ensure_conversation(conversation_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_state (conversation_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id, key) DO UPDATE SET
                    value = excluded.value, updated_at = excluded.updated_at
                """,
                (conversation_id, key, value, _now()),
            )

    def get_facts_kv(
        self, conversation_id: str, keys: list[str] | None = None
    ) -> dict[str, str]:
        with self._connect() as conn:
            if keys:
                placeholders = ",".join("?" * len(keys))
                rows = conn.execute(
                    f"""
                    SELECT key, value FROM conversation_state
                    WHERE conversation_id = ? AND key IN ({placeholders})
                    """,
                    [conversation_id, *keys],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, value FROM conversation_state WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchall()
            return {r["key"]: r["value"] for r in rows}

    def log_event(
        self,
        conversation_id: str,
        turn_number: int,
        description: str,
        payload: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (conversation_id, turn_number, description, payload, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, turn_number, description, json.dumps(payload or {}), _now()),
            )

    def purge_conversation(self, conversation_id: str, *, delete_conversation: bool = True) -> dict[str, int]:
        """Delete all persisted state for exactly one conversation.

        Also removes vector rows and conversation metadata. This is intentionally
        explicit rather than a generic "clear DB" operation.
        """
        tables = ["entities", "facts", "events", "relationships", "conversation_state"]
        deleted: dict[str, int] = {}
        with self._connect() as conn:
            # fact_vectors has no conversation_id; remove vectors through fact ids.
            fact_rows = conn.execute("SELECT id FROM facts WHERE conversation_id = ?", (conversation_id,)).fetchall()
            fact_ids = [int(r["id"]) for r in fact_rows]
            if fact_ids:
                placeholders = ",".join("?" * len(fact_ids))
                try:
                    cur = conn.execute(f"DELETE FROM fact_vectors WHERE fact_id IN ({placeholders})", fact_ids)
                    deleted["fact_vectors"] = int(cur.rowcount or 0)
                except sqlite3.OperationalError:
                    deleted["fact_vectors"] = 0
            else:
                deleted["fact_vectors"] = 0
            for table in tables:
                try:
                    cur = conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (conversation_id,))
                    deleted[table] = int(cur.rowcount or 0)
                except sqlite3.OperationalError:
                    # Older/current schemas may not have optional tables.
                    deleted[table] = 0
            if delete_conversation:
                cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
                deleted["conversations"] = int(cur.rowcount or 0)
        return deleted

    def entity_count(self, conversation_id: str | None = None) -> int:
        with self._connect() as conn:
            if conversation_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM entities WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()
            return int(row["c"]) if row else 0

    def fact_count(self, conversation_id: str | None = None) -> int:
        with self._connect() as conn:
            if conversation_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM facts WHERE conversation_id = ? AND active = 1",
                    (conversation_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM facts WHERE active = 1"
                ).fetchone()
            return int(row["c"]) if row else 0
