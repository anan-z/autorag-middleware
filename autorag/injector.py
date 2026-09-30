"""Build context blocks from conversation-scoped facts and entities."""

from __future__ import annotations

from typing import Any

from .config import AutoRAGConfig
from .database import StateDatabase


class ContextInjector:
    def __init__(self, db: StateDatabase, config: AutoRAGConfig):
        self.db = db
        self.config = config

    def build_system_prompt(
        self,
        conversation_id: str,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        sections: list[str] = []

        kv = self.db.get_facts_kv(conversation_id)
        if kv:
            lines = [f"{k}: {v}" for k, v in kv.items()]
            sections.append("[CORE STATE]\n" + "\n".join(lines) + "\n[/CORE STATE]")

        # Facts relevant to the user message
        relevant = self.db.search_facts(conversation_id, user_message or "", limit=8)
        if not relevant:
            relevant = self.db.list_active_facts(conversation_id, limit=8)

        if relevant:
            sections.append(
                "[ESTABLISHED FACTS — keep consistent]\n"
                + self._format_facts(relevant)
                + "\n[/ESTABLISHED FACTS]"
            )

        # Entities mentioned or recent
        mentioned = self._entities_mentioned(conversation_id, user_message or "")
        if mentioned:
            sections.append(
                "[RELEVANT ENTITIES]\n"
                + self._format_entities(mentioned)
                + "\n[/RELEVANT ENTITIES]"
            )

        if not sections:
            return ""

        block = "\n\n".join(sections)
        max_chars = self.config.max_injected_tokens * 4
        if len(block) > max_chars:
            block = block[: max_chars - 20] + "\n...[truncated]"
        return block

    def _entities_mentioned(
        self, conversation_id: str, text: str
    ) -> list[dict[str, Any]]:
        if not text:
            return self.db.get_recent_entities(conversation_id, limit=5)
        candidates = self.db.search_entities(conversation_id, text, limit=10)
        text_l = text.lower()
        direct = [e for e in candidates if e["name"].lower() in text_l]
        if direct:
            return direct[:5]
        all_recent = self.db.get_recent_entities(conversation_id, limit=30)
        present = [e for e in all_recent if e["name"].lower() in text_l]
        return present[:5] or candidates[:3]

    @staticmethod
    def _format_facts(facts: list[dict[str, Any]]) -> str:
        lines = []
        for f in facts:
            conf = f.get("confidence")
            conf_s = f" (conf={conf:.2f})" if isinstance(conf, float) else ""
            lines.append(
                f"- {f['subject']}.{f['predicate']} = {f['object']}{conf_s}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_entities(entities: list[dict[str, Any]]) -> str:
        lines = []
        for e in entities:
            attrs = e.get("attributes") or {}
            show = {k: v for k, v in attrs.items() if k not in ("source", "label", "context")}
            attr_str = ", ".join(f"{k}={v}" for k, v in show.items()) if show else ""
            line = f"- {e['name']} ({e.get('type', '?')})"
            if attr_str:
                line += f": {attr_str}"
            lines.append(line)
        return "\n".join(lines)
