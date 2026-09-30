"""Entity + structured fact extraction (heuristic and optional same-model LLM)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

from .database import StateDatabase

logger = logging.getLogger("autorag.extractor")

_nlp = None

EXTRACT_SYSTEM = """You extract durable facts from a short dialogue exchange.
Return ONLY a JSON object:
{"facts":[{"subject":"string","predicate":"string","object":"string","confidence":0.0,"speaker":"user"|"assistant"}]}

Rules:
- Only durable state (identity, locations of things, inventory, relationships, decisions, physical attributes).
- Resolve pronouns using context (e.g. "them" referring to keys → subject "car keys" or "keys").
- Use subject "user" for the human speaker's personal state when no other name is given.
- Prefer existing subject names from KNOWN FACTS when referring to the same entity.
- Skip questions, speculation, hedges (I think / maybe / probably / not sure).
- Skip pure narration fluff with no persistent attribute.
- predicate: short snake_case (location, hair_color, age, owns, status, decided).
- confidence 0.0-1.0; use <0.6 if uncertain; assistant-only claims max 0.75 unless clearly established.
- If nothing durable: {"facts":[]}
- No markdown fences. No commentary."""


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except ImportError:
        return None


def _parse_facts_json(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("LLM extract returned non-JSON (first 200 chars): %r", text[:200])
        return []
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        logger.warning("LLM extract JSON missing facts list: %r", type(data).__name__)
        return []
    out: list[dict[str, Any]] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        subj = _normalize_subject(str(f.get("subject") or ""))
        pred = str(f.get("predicate") or "").strip().lower().replace(" ", "_")
        obj = str(f.get("object") or "").strip()
        if not subj or not pred or not obj:
            continue
        try:
            conf = float(f.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        speaker = str(f.get("speaker") or "unknown").lower()
        if speaker == "assistant":
            conf = min(conf, 0.75)
        out.append(
            {
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "confidence": max(0.0, min(1.0, conf)),
                "source": "llm",
                "speaker": speaker,
            }
        )
    return out


_PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "their",
    "i", "me", "my", "mine", "we", "us", "our", "you", "your",
}

_HEDGE_RE = re.compile(
    r"\b(i think|i guess|maybe|perhaps|probably|not sure|might have|could have|"
    r"i believe|seems like|sort of|kind of)\b",
    re.I,
)


def _normalize_subject(name: str) -> str:
    name = name.strip()
    name = re.sub(r"^(my|the|a|an)\s+", "", name, flags=re.I).strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        return ""
    # Prefer singular "keys" style consistency
    lower = name.lower()
    if lower in _PRONOUNS:
        return ""
    # Title-case multi-word carefully; keep existing capitals for names
    if name.islower() or name.isupper():
        name = name.title()
    return name


def _is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    return bool(re.match(r"^(where|what|when|who|why|how|did|do|does|is|are|was|were|can|could|would|should)\b", t, re.I))


def strip_reasoning(text: str) -> str:
    """Remove model thinking blocks before extraction."""
    if not text:
        return ""
    text = re.sub(r"<think>[\s\S]*?</think>", " ", text, flags=re.I)
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", " ", text, flags=re.I)
    return text.strip()


class EntityExtractor:
    """Heuristic entity extraction + structured fact candidates."""

    # Name capture groups intentionally CASE-SENSITIVE ([A-Z][a-z]+) — no re.I
    PATTERNS: list[tuple[re.Pattern[str], str, str | None]] = [
        (re.compile(r"\b([A-Z][a-z]+)\s+is\s+(\d{1,3})\s+years?\s+old\b"), "character", "age"),
        (re.compile(r"\b([A-Z][a-z]+),\s*(\d{1,3}),"), "character", "age"),
        (re.compile(r"\b(?:my name is|I am|I'm)\s+([A-Z][a-z]+)\b"), "character", None),
        # left/put keys on X — case-insensitive verbs, normalized subject
        (
            re.compile(
                r"\b(?:left|put|placed|forgot)\s+(?:my\s+|the\s+)?([\w][\w\s]{1,28}?)\s+"
                r"on\s+(?:the\s+)?([\w][\w\s\-]{1,40})",
                re.I,
            ),
            "item",
            "location",
        ),
        (
            re.compile(
                r"\b(?:keys?|wallet|phone|bag|sword|book)\b(?:\s+\w+){0,6}?\s+"
                r"(?:on|in|at)\s+(?:the\s+)?([\w][\w\s\-]{1,40})",
                re.I,
            ),
            "item",
            "location_loose",
        ),
        # Maya's hair is black / Maya's black hair
        (
            re.compile(r"\b([A-Z][a-z]+)'s\s+(hair|eyes)\s+(?:is|are|was|were)\s+(\w+)\b"),
            "character",
            "attr",
        ),
        (
            re.compile(r"\b([A-Z][a-z]+)'s\s+(red|black|blonde|brown|white|blue|green|gray|grey)\s+(hair|eyes)\b"),
            "character",
            "attr_adj",
        ),
        (
            re.compile(r"\b(?:in|at|inside|near)\s+the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
            "location",
            None,
        ),
    ]

    STOP_NAMES = {
        "The", "A", "An", "I", "You", "He", "She", "It", "They", "We",
        "This", "That", "There", "Here", "What", "When", "Where", "Who",
        "How", "Why", "Yes", "No", "Okay", "Ok", "Hello", "Hi", "Hey",
        "Sorry", "Please", "Thanks", "Thank", "Well", "So", "But", "And",
        "Or", "If", "Then", "Old", "New", "Good", "Bad", "Tired", "Happy",
        "Sad", "Maybe", "Perhaps", "Actually", "Suddenly", "Finally",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    }

    def __init__(
        self,
        db: StateDatabase,
        use_spacy: bool = False,
        min_confidence: float = 0.55,
    ):
        self.db = db
        self.use_spacy = use_spacy
        self.min_confidence = min_confidence
        self._nlp = _get_nlp() if use_spacy else None

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        text = strip_reasoning(text)
        entities: list[dict[str, Any]] = []
        if self._nlp is not None:
            entities.extend(self._spacy_extract(text))
        entities.extend(self._pattern_entities(text))
        entities.extend(self._capitalized_names(text))
        return self._merge_entities(entities)

    def extract_fact_candidates(self, text: str, *, from_assistant: bool = False) -> list[dict[str, Any]]:
        text = strip_reasoning(text)
        if not text.strip():
            return []
        if _is_question(text) and not re.search(
            r"\b(left|put|placed|is|are|was)\b.+\b(on|in|at)\b", text, re.I
        ):
            pass

        facts: list[dict[str, Any]] = []
        facts.extend(self._generic_durable_facts(text, from_assistant=from_assistant))
        for pattern, etype, attr_key in self.PATTERNS:
            for m in pattern.finditer(text):
                # Hedge check on surrounding window
                start = max(0, m.start() - 40)
                window = text[start : m.end() + 10]
                if _HEDGE_RE.search(window):
                    continue

                if attr_key == "age":
                    name = _normalize_subject(m.group(1))
                    if not name:
                        continue
                    try:
                        age = int(m.group(2))
                    except ValueError:
                        continue
                    facts.append(self._fact(name, "age", str(age), 0.85, from_assistant))
                elif attr_key == "location":
                    item = _normalize_subject(m.group(1))
                    loc = m.group(2).strip()
                    if item and loc and not _is_question(m.group(0)):
                        facts.append(self._fact(item, "location", loc, 0.8, from_assistant))
                elif attr_key == "location_loose":
                    loc = m.group(1).strip()
                    kind = re.search(r"\b(keys?|wallet|phone|bag|sword|book)\b", m.group(0), re.I)
                    subj = _normalize_subject(kind.group(1) if kind else "item")
                    if subj and loc and not _is_question(m.group(0)):
                        facts.append(self._fact(subj, "location", loc, 0.65, from_assistant))
                elif attr_key == "attr":
                    name = _normalize_subject(m.group(1))
                    attr, val = m.group(2).lower(), m.group(3)
                    pred = "hair_color" if attr == "hair" else "eye_color" if attr == "eyes" else attr
                    if name:
                        facts.append(self._fact(name, pred, val, 0.8, from_assistant))
                elif attr_key == "attr_adj":
                    name = _normalize_subject(m.group(1))
                    color, attr = m.group(2).lower(), m.group(3).lower()
                    pred = "hair_color" if attr == "hair" else "eye_color"
                    if name:
                        facts.append(self._fact(name, pred, color, 0.75, from_assistant))
                elif attr_key is None and etype == "character":
                    name = _normalize_subject(m.group(1))
                    if name:
                        # identity mention only — no fact unless age etc.
                        pass
        return facts

    def _fact(
        self, subject: str, predicate: str, obj: str, conf: float, from_assistant: bool
    ) -> dict[str, Any]:
        if from_assistant:
            conf = min(conf, 0.75)
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj.strip(),
            "confidence": conf,
            "source": "heuristic",
            "speaker": "assistant" if from_assistant else "user",
        }


    def _generic_durable_facts(self, text: str, *, from_assistant: bool) -> list[dict[str, Any]]:
        """Domain-neutral durable facts (finance, product, physics, everyday)."""
        out: list[dict[str, Any]] = []

        for m in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9_]{1,40})\s+value\s+is\s+"
            r"([€$£]?\d[\d.,]*(?:\s*(?:euros?|usd|dollars?|k|m))?)",
            text,
            re.I,
        ):
            subj = _normalize_subject(m.group(1)) or m.group(1)
            out.append(self._fact(subj, "value", m.group(2).strip(), 0.9, from_assistant))

        for m in re.finditer(
            r"\b(?:the\s+)?([A-Za-z][A-Za-z0-9_\-\s]{1,40}?)\s+"
            r"(?:is|are|was|were|must be|should be|set at|remains?)\s+"
            r"([€$£]?\d[\d.,]*\s*(?:k|K|m|M|%|mm|cm|kg|euros?|dollars?|usd|hours?|days?)?|"
            r"v?\d+(?:\.\d+)*)",
            text,
            re.I,
        ):
            subj = (_normalize_subject(m.group(1)) or m.group(1)).strip()
            if subj.lower() in {"it", "this", "that", "there", "what", "which"}:
                continue
            # Drop run-on subjects ("We decided the cash reserve")
            if len(subj.split()) > 4 or re.match(
                r"^(we|i|you|they|he|she)\b", subj, re.I
            ):
                continue
            window = text[max(0, m.start() - 30) : m.end() + 5]
            if _HEDGE_RE.search(window) and not re.search(
                r"\b(decided|agreed|confirmed|settled|record)\b", text, re.I
            ):
                continue
            out.append(self._fact(subj, "value", m.group(2).strip(), 0.85, from_assistant))

        for m in re.finditer(
            r"\b(?:we\s+)?(?:decided|agreed|assumed|concluded|rejected|confirmed)\s+(?:that\s+)?(.{5,100})",
            text,
            re.I,
        ):
            clause = m.group(1).strip().rstrip(".")
            out.append(self._fact("decision", "statement", clause[:120], 0.8, from_assistant))
            inner = re.search(
                r"([A-Za-z][A-Za-z0-9_\-\s]{1,40}?)\s+is\s+([€$£]?\d[\d.,]*\w*)",
                clause,
                re.I,
            )
            if inner:
                subj = (_normalize_subject(inner.group(1)) or inner.group(1)).strip()
                out.append(self._fact(subj, "value", inner.group(2).strip(), 0.88, from_assistant))

        for m in re.finditer(
            r"\b(?:left|put|placed|stored|kept)\s+(?:the\s+|my\s+|our\s+)?(.{2,40}?)\s+"
            r"(?:on|in|at)\s+(?:the\s+)?(.{2,50})",
            text,
            re.I,
        ):
            if _HEDGE_RE.search(text[max(0, m.start() - 40) : m.end()]):
                continue
            subj = (_normalize_subject(m.group(1)) or m.group(1)).strip()
            out.append(self._fact(subj, "location", m.group(2).strip(), 0.8, from_assistant))

        return out

    def process_turn_heuristic(
        self,
        conversation_id: str,
        user_text: str,
        ai_text: str,
        turn_number: int,
    ) -> dict[str, Any]:
        """Upsert entities only. Facts are committed via MemoryPipeline."""
        combined = f"{strip_reasoning(user_text or '')}\n{strip_reasoning(ai_text or '')}"
        entities = self.extract_entities(combined)
        entity_ids: list[int] = []
        for e in entities:
            if e.get("confidence", 0) < self.min_confidence:
                continue
            eid = self.db.upsert_entity(
                conversation_id,
                e["name"],
                e["type"],
                e.get("attributes") or {},
                e.get("confidence", 0.6),
            )
            entity_ids.append(eid)
        if entity_ids:
            self.db.log_event(
                conversation_id,
                turn_number,
                f"heuristic entities: {len(entity_ids)}",
                {"entities": len(entity_ids)},
            )
        return {"entities": len(entity_ids), "facts": []}

    def _spacy_extract(self, text: str) -> list[dict[str, Any]]:
        assert self._nlp is not None
        doc = self._nlp(text)
        out: list[dict[str, Any]] = []
        mapping = {
            "PERSON": "character",
            "ORG": "organization",
            "GPE": "location",
            "LOC": "location",
            "FAC": "location",
            "PRODUCT": "item",
        }
        for ent in doc.ents:
            etype = mapping.get(ent.label_)
            if not etype:
                continue
            name = _normalize_subject(ent.text)
            if not name or name in self.STOP_NAMES or len(name) < 2:
                continue
            out.append(
                {
                    "name": name,
                    "type": etype,
                    "attributes": {"source": "spacy"},
                    "confidence": 0.75,
                }
            )
        return out

    def _pattern_entities(self, text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pattern, etype, attr_key in self.PATTERNS:
            for m in pattern.finditer(text):
                if attr_key in ("location", "location_loose"):
                    continue
                name = _normalize_subject(m.group(1))
                if not name or name in self.STOP_NAMES or len(name) < 2:
                    continue
                attrs: dict[str, Any] = {"source": "pattern"}
                if attr_key == "age" and m.lastindex and m.lastindex >= 2:
                    try:
                        attrs["age"] = int(m.group(2))
                    except ValueError:
                        pass
                out.append(
                    {
                        "name": name,
                        "type": "character" if etype in ("character", "item") else etype,
                        "attributes": attrs,
                        "confidence": 0.7,
                    }
                )
        return out

    def _capitalized_names(self, text: str) -> list[dict[str, Any]]:
        # Only mid-sentence or repeated proper nouns — skip pure sentence-initial once
        candidates = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", text)
        # Also allow start-of-text names that repeat
        starts = re.findall(r"(?:^|[.!?]\s+)([A-Z][a-z]{2,})\b", text)
        counts: dict[str, int] = {}
        for c in candidates + starts:
            if c in self.STOP_NAMES:
                continue
            counts[c] = counts.get(c, 0) + 1
        out = []
        for name, n in counts.items():
            # Require repetition for sentence-initial-only tokens
            conf = min(0.7, 0.4 + 0.12 * n)
            if n == 1 and name in starts and name not in candidates:
                conf = 0.45  # weak single sentence-initial
            out.append(
                {
                    "name": name,
                    "type": "character",
                    "attributes": {"source": "capitalized"},
                    "confidence": conf,
                }
            )
        return out

    def _merge_entities(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for e in entities:
            key = e["name"].lower()
            if key in merged:
                merged[key]["attributes"].update(e.get("attributes") or {})
                merged[key]["confidence"] = max(
                    merged[key]["confidence"], e.get("confidence", 0)
                )
                if e.get("type") and e["type"] != "unknown":
                    merged[key]["type"] = e["type"]
            else:
                merged[key] = {
                    "name": e["name"],
                    "type": e.get("type") or "unknown",
                    "attributes": dict(e.get("attributes") or {}),
                    "confidence": e.get("confidence", 0.5),
                }
        return list(merged.values())


class MemoryPipeline:
    """Extract → Validate → Commit with optional same-model LLM extract."""

    def __init__(
        self,
        db: StateDatabase,
        heuristic: EntityExtractor,
        *,
        mode: str = "hybrid",
        min_confidence: float = 0.55,
        on_conflict: str = "reconcile",
        llm_every_n_turns: int = 1,
        llm_max_tokens: int = 400,
        llm_temperature: float = 0.1,
    ):
        self.db = db
        self.heuristic = heuristic
        self.mode = mode
        self.min_confidence = min_confidence
        self.on_conflict = on_conflict
        self.llm_every_n_turns = max(1, llm_every_n_turns)
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    async def process_turn(
        self,
        conversation_id: str,
        user_text: str,
        ai_text: str,
        turn_number: int,
        llm_complete: Callable[..., Awaitable[str]] | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "entities": 0,
            "facts_committed": [],
            "conflicts": [],
            "llm_used": False,
        }

        user_text = strip_reasoning(user_text or "")
        ai_text = strip_reasoning(ai_text or "")

        h = self.heuristic.process_turn_heuristic(
            conversation_id, user_text, ai_text, turn_number
        )
        summary["entities"] = h["entities"]

        candidates: list[dict[str, Any]] = []
        candidates.extend(self.heuristic.extract_fact_candidates(user_text, from_assistant=False))
        candidates.extend(self.heuristic.extract_fact_candidates(ai_text, from_assistant=True))

        use_llm = False
        if llm_complete is not None and self.mode in ("llm", "hybrid"):
            if self.mode == "llm":
                use_llm = turn_number % self.llm_every_n_turns == 0
            else:
                use_llm = bool(candidates) or (turn_number % self.llm_every_n_turns == 0)

        if use_llm and llm_complete is not None:
            try:
                known = self.db.list_active_facts(conversation_id, limit=15)
                llm_facts = await self._llm_extract(user_text, ai_text, known, llm_complete)
                summary["llm_used"] = True
                candidates.extend(llm_facts)
            except Exception:
                logger.exception("LLM structured extraction failed")

        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for f in candidates:
            key = (f["subject"].lower(), f["predicate"].lower())
            if key not in merged or f.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key] = f
        candidates = list(merged.values())

        for f in candidates:
            if f.get("confidence", 0) < self.min_confidence:
                continue
            result = await self._validate_and_commit(
                conversation_id, f, turn_number, llm_complete
            )
            if result:
                summary["facts_committed"].append(result)
                if result.get("conflict"):
                    summary["conflicts"].append(result["conflict"])

        return summary

    async def _llm_extract(
        self,
        user_text: str,
        ai_text: str,
        known_facts: list[dict[str, Any]],
        llm_complete: Callable[..., Awaitable[str]],
    ) -> list[dict[str, Any]]:
        known_lines = "\n".join(
            f"- {f['subject']}.{f['predicate']} = {f['object']}" for f in known_facts
        ) or "(none)"
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"KNOWN FACTS:\n{known_lines}\n\n"
                    f"User: {user_text or '(empty)'}\n"
                    f"Assistant: {ai_text or '(empty)'}\n\n"
                    "Extract durable facts as JSON."
                ),
            },
        ]
        raw = await llm_complete(
            messages,
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
        )
        return _parse_facts_json(raw or "")

    async def _validate_and_commit(
        self,
        conversation_id: str,
        fact: dict[str, Any],
        turn_number: int,
        llm_complete: Callable[..., Awaitable[str]] | None,
    ) -> dict[str, Any] | None:
        existing = self.db.get_active_fact(
            conversation_id, fact["subject"], fact["predicate"]
        )
        conflict = None

        if existing and existing["object"].strip().lower() != str(fact["object"]).strip().lower():
            conflict = {
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "existing": existing["object"],
                "incoming": fact["object"],
            }
            if self.on_conflict == "flag":
                logger.info(
                    "Fact conflict %s.%s: %r -> %r",
                    fact["subject"],
                    fact["predicate"],
                    existing["object"],
                    fact["object"],
                )
            elif self.on_conflict == "reconcile" and llm_complete is not None:
                resolved = await self._reconcile(existing, fact, llm_complete)
                if resolved is None:
                    self.db.log_event(
                        conversation_id, turn_number, "conflict kept existing", conflict
                    )
                    return {"status": "rejected", "conflict": conflict, "fact": existing}
                fact = resolved

        result = self.db.commit_fact(
            conversation_id,
            fact["subject"],
            fact["predicate"],
            fact["object"],
            confidence=fact.get("confidence", 0.7),
            turn_number=turn_number,
            source=fact.get("source", "heuristic"),
        )
        if conflict:
            result["conflict"] = conflict
        return result

    async def _reconcile(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any],
        llm_complete: Callable[..., Awaitable[str]],
    ) -> dict[str, Any] | None:
        prompt = (
            "Conversation memory conflict. Reply with ONLY JSON:\n"
            '{"action":"keep_existing"|"accept_new"|"update","object":"...","confidence":0.0}\n\n'
            f"Subject: {incoming['subject']}\n"
            f"Predicate: {incoming['predicate']}\n"
            f"Existing value: {existing['object']}\n"
            f"New value from latest turn: {incoming['object']}\n"
            "If the new text intentionally changes state, accept_new or update. "
            "If it looks like a continuity error or speculation, keep_existing."
        )
        messages = [
            {"role": "system", "content": "You resolve memory conflicts. JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await llm_complete(messages, max_tokens=150, temperature=0.0)
        except Exception:
            logger.exception("Reconcile call failed")
            return None

        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        action = str(data.get("action", "")).lower()
        if action == "keep_existing":
            return None
        obj = str(data.get("object") or incoming["object"]).strip()
        try:
            conf = float(data.get("confidence", incoming.get("confidence", 0.7)))
        except (TypeError, ValueError):
            conf = 0.7
        return {
            "subject": incoming["subject"],
            "predicate": incoming["predicate"],
            "object": obj,
            "confidence": conf,
            "source": "reconcile",
        }
