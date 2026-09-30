"""Entity and fact extraction for AutoRAG middleware.

Rewritten with proper NLP filtering to prevent:
- Pronouns/prepositions/adverbs extracted as entities
- System prompt tokens extracted as entities
- Prose captured as location values
- Multi-word proper nouns split incorrectly
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from .database import StateDatabase

logger = logging.getLogger(__name__)


# --- spaCy lazy loader ---
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            try:
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", "en_core_web_sm"],
                    check=True, capture_output=True
                )
                import spacy
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"spaCy unavailable: {e}")
                _nlp = False
    return _nlp if _nlp is not False else None


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks if present."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _normalize_subject(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    return value


# Property labels that should NOT be promoted to their own subject.
# If a param line's label is in this set, it stays a predicate of the
# current subject (from marker or "params" fallback). Otherwise, if the
# label looks like a proper noun and the value looks like data, the label
# is promoted to the subject (definition-line case).
_KNOWN_PROPERTY_LABELS = {
    "width", "height", "length", "depth", "thickness", "radius", "diameter",
    "weight", "mass", "volume", "area", "size",
    "price", "cost", "value", "budget", "reserve", "balance",
    "age", "count", "quantity", "iterations", "steps", "turns",
    "voltage", "current", "resistance", "power", "wattage", "amperage",
    "frequency", "speed", "temperature", "pressure",
    "model", "version", "revision", "status", "state", "type", "kind",
    "name", "title", "label", "id", "identifier", "key",
    "location", "position", "address", "coordinates",
    "color", "colour", "material", "finish",
    "capacity", "limit", "quota", "threshold", "timeout", "interval",
    "description", "summary", "note", "comment",
    "start", "end", "duration", "deadline", "date", "time",
    "author", "owner", "creator", "user",
    "language", "framework", "library", "platform", "engine",
    "protocol", "format", "encoding", "schema",
    "host", "port", "url", "uri", "path", "endpoint",
}


class EntityExtractor:
    """Conservative, domain-neutral durable-memory extractor.

    Design rule: false positives are more damaging than missed memories.
    We therefore extract only high-signal statements (explicit locations,
    durable attributes, decisions/requirements, and technical parameter lines).
    We do *not* treat arbitrary capitalized words as entities.
    """

    STOP_SUBJECTS = {
        "i", "you", "he", "she", "it", "we", "they", "this", "that",
        "the", "a", "an", "someone", "something", "thing", "things",
        "user", "assistant", "system", "model", "response", "answer",
    }
    HEDGE_RE = re.compile(r"\b(?:maybe|perhaps|possibly|probably|might|could|i think|i guess|it seems|apparently)\b", re.I)
    # High-signal durable forms. Keep these deliberately narrow.
    LOCATION_RE = re.compile(
        r"\b(?:I|we|you|he|she|they)\s+(?:left|put|placed|stored|kept)\s+(?:the|my|our|your|his|her|their)\s+(.{2,60}?)\s+(?:on|in|at|under|inside|beside|behind|near)\s+(?:the\s+)?(.{2,60}?)(?=[.!?,;]|$)", re.I
    )
    POSSESSION_LOCATION_RE = re.compile(
        r"\b(?:the|my|our|your|his|her|their)\s+(.{2,60}?)\s+(?:is|are|was|were)\s+(?:on|in|at|under|inside|beside|behind|near)\s+(?:the\s+)?(.{2,60}?)(?=[.!?,;]|$)", re.I
    )
    AGE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})\s+is\s+(\d{1,3})\s+years?\s+old\b")
    HAIR_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})['\u2019]s\s+hair\s+(?:is|was)\s+([A-Za-z-]{2,20})\b", re.I)
    HAIR2_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})\s+has\s+([A-Za-z-]{2,20})\s+hair\b", re.I)
    EYES_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})['\u2019]s\s+eyes\s+(?:are|were)\s+([A-Za-z-]{2,20})\b", re.I)
    DECISION_RE = re.compile(r"\b(?:we|I)\s+(?:decided|agreed|settled on|chose|selected|rejected|assumed|will use|are using)\s+(?:that\s+)?(.{4,160}?)(?=[.!?]|$)", re.I)
    REQUIREMENT_RE = re.compile(r"\b(?:must|shall|required to|needs to|need to|should remain|has to)\s+(.{4,160}?)(?=[.!?]|$)", re.I)
    EXPLICIT_FACT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})\s+(?:has|owns|uses|lives in|works at|works for)\s+(.{2,100}?)(?=[.!?]|$)", re.I)

    _PARAM_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_ \-]{0,40}?)\s*[:=]\s*(.+?)\s*$", re.MULTILINE)

    # Markdown bold pattern: **Label:** value (closing ** AFTER colon)
    _PARAM_MARKDOWN = re.compile(
        r"^\s*\*\*([A-Za-z][A-Za-z0-9_ \-\/]{0,50}?):\*\*\s*(.+?)\s*$",
        re.MULTILINE,
    )

    # Markdown bullet pattern: * **Label:** value
    _PARAM_BULLET = re.compile(
        r"^\s*[\*\-]\s*\*\*([A-Za-z][A-Za-z0-9_ \-\/]{0,50}?):\*\*\s*(.+?)\s*$",
        re.MULTILINE,
    )
    _SUBJECT_MARKER = re.compile(r"^\s*\[\s*subject\s*:\s*([^\]]+)\]\s*$", re.MULTILINE | re.I)
    _PARAM_STOP_LABELS = {"note", "warning", "example", "todo", "see", "also", "ref", "reference", "source", "hint", "tip", "important", "remember", "caution", "danger"}

    def __init__(self, db: StateDatabase, use_spacy: bool = False, min_confidence: float = 0.70):
        self.db = db
        self.use_spacy = use_spacy
        self.min_confidence = min_confidence
        self._nlp = _get_nlp() if use_spacy else None
        self._is_system_message = False
        self.conversation_id = "default"

    def set_system_message(self, is_system: bool):
        self._is_system_message = is_system

    def _strip_bracketed_content(self, text: str) -> str:
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\u3010[^\u3011]*\u3011", " ", text)
        text = re.sub(r"\u300c[^\u300d]*\u300d", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clean_value(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?"))

    @staticmethod
    def _clean_subject(value: str) -> str:
        value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?"))
        return _normalize_subject(value) or value

    def _valid_subject(self, subject: str) -> bool:
        s = subject.strip().lower()
        if not s or s in self.STOP_SUBJECTS or len(s) < 2 or len(s) > 80:
            return False
        if self.HEDGE_RE.search(subject):
            return False
        return True

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Only emit entities attached to high-signal durable facts.

        No capitalized-word heuristic and no blanket spaCy NER: both produced
        database pollution in long narrative conversations.
        """
        return []

    def _fact(self, subject: str, predicate: str, obj: str, conf: float, from_assistant: bool, source: str = "heuristic") -> dict[str, Any] | None:
        subject = self._clean_subject(subject)
        obj = self._clean_value(obj)
        predicate = re.sub(r"[^a-z0-9_]+", "_", predicate.lower()).strip("_")
        if not self._valid_subject(subject) or not predicate or not obj or len(obj) > 180:
            return None
        if self.HEDGE_RE.search(f"{subject} {predicate} {obj}"):
            return None
        return {"subject": subject, "predicate": predicate, "object": obj, "confidence": conf, "source": source if not from_assistant else "assistant"}

    def extract_fact_candidates(self, text: str, *, from_assistant: bool = False) -> list[dict[str, Any]]:
        if self._is_system_message:
            return []
        text = strip_reasoning(text or "")
        clean = self._strip_bracketed_content(text)
        # A hedged statement is not durable memory. Reject the whole turn rather
        # than accidentally capturing the asserted-looking fragment inside it.
        if self.HEDGE_RE.search(clean):
            return []
        out: list[dict[str, Any]] = []

        def add(subject, pred, obj, conf=0.9, source="heuristic"):
            f = self._fact(subject, pred, obj, conf, from_assistant, source)
            if f: out.append(f)

        # Explicit locations: this is the important everyday-assistant case.
        for rx in (self.LOCATION_RE, self.POSSESSION_LOCATION_RE):
            for m in rx.finditer(clean):
                add(m.group(1), "location", m.group(2), 0.90)

        for m in self.AGE_RE.finditer(clean):
            add(m.group(1), "age", m.group(2), 0.95)
        for m in self.HAIR_RE.finditer(clean):
            add(m.group(1), "hair_color", m.group(2), 0.90)
        for m in self.HAIR2_RE.finditer(clean):
            add(m.group(1), "hair_color", m.group(2), 0.90)
        for m in self.EYES_RE.finditer(clean):
            add(m.group(1), "eye_color", m.group(2), 0.90)

        for m in self.DECISION_RE.finditer(clean):
            add("conversation", "decision", m.group(1), 0.90)
        for m in self.REQUIREMENT_RE.finditer(clean):
            add("conversation", "requirement", m.group(1), 0.88)
        for m in self.EXPLICIT_FACT_RE.finditer(clean):
            add(m.group(1), "association", m.group(2), 0.82)

        # Technical parameter pack remains opt-in and data-shaped only.
        out.extend(self.extract_params(text, from_assistant=from_assistant))

        # Deduplicate exact candidates.
        unique = {}
        for f in out:
            key = (f["subject"].lower(), f["predicate"], f["object"].lower())
            unique[key] = f
        return list(unique.values())

    @staticmethod
    def _normalize_param_label(label: str) -> str:
        label = label.strip().lower()
        label = re.sub(r"[\s\-]+", "_", label)
        return re.sub(r"[^a-z0-9_]", "", label)

    @staticmethod
    def _looks_like_parameter_value(value: str) -> bool:
        """Accept data-like values including proper nouns, reject prose."""
        value = value.strip()
        if not value:
            return False

        # Remove parenthetical content for validation
        value_clean = re.sub(r'\([^)]*\)', '', value).strip()
        if not value_clean:
            return False

        # Reject: pure lowercase prose (e.g., "she looks up")
        if re.match(r"^[a-z][a-z\s]+$", value_clean):
            return False

        # Accept: any number (including small integers)
        if re.match(r"^-?\d+(?:\.\d+)?$", value_clean):
            return True

        # Accept: number with unit (10mm, 12V, 1.8 deg)
        if re.search(r"\d+\s*[a-zA-Z\u00b0\u03a9\u00b5%]+", value_clean):
            return True

        # Accept: currency ($50k, EUR 12.50)
        if re.search(r"[\u20ac$\u00a3]\s*\d", value_clean):
            return True

        # Accept: version (v2, 1.2.3)
        if re.match(r"^v?\d+(?:\.\d+)+(?:[-_][\w-]+)?$", value_clean):
            return True

        # Accept: model identifier (NEMA17, BME280)
        if re.match(r"^[A-Z]+[\w-]*\d+[\w-]*$", value_clean):
            return True

        # Accept: capitalized words (proper nouns like Ayanna, Nymph)
        if re.match(r"^[A-Z][a-z]+$", value_clean):
            return True

        # Accept: multi-word with capitals (Black hair, Large eyes)
        if re.match(r"^[A-Z][a-zA-Z\s,\-\/]+$", value_clean):
            return True

        # Fallback: accept if has digit
        return bool(re.search(r"\d", value_clean))

    def preprocess_message(self, text: str) -> tuple[str, str]:
        markers = list(self._SUBJECT_MARKER.finditer(text))
        if not markers:
            return text, text
        clean_parts, last_end = [], 0
        for m in markers:
            clean_parts.append(text[last_end:m.start()]); last_end = m.end()
        clean_parts.append(text[last_end:])
        return "".join(clean_parts), text

    def _collect_param_matches(self, text: str) -> list[tuple[int, re.Match]]:
        """Collect all param matches with their start positions, in document order."""
        matches: list[tuple[int, re.Match]] = []
        for rx in (self._PARAM_LINE, self._PARAM_MARKDOWN, self._PARAM_BULLET):
            for m in rx.finditer(text):
                matches.append((m.start(), m))
        matches.sort(key=lambda x: x[0])
        return matches

    def _is_definition_line(self, label: str, value: str) -> bool:
        """True if this looks like 'Squarebox: 1m x 1m x 1m' -- a subject definition.

        Heuristic: the label is capitalized like a proper noun, it's not a
        known property name, and the value looks like data.
        """
        normalized = self._normalize_param_label(label)
        if not normalized:
            return False
        if normalized in _KNOWN_PROPERTY_LABELS:
            return False
        # Label must start with a capital letter (proper-noun-like)
        if not label[:1].isupper():
            return False
        # Must be a single token or two-token identifier, not a sentence
        if len(label.split()) > 2:
            return False
        # Value must look like data, not prose
        if not self._looks_like_parameter_value(value):
            return False
        return True

    def extract_params(self, text: str, *, from_assistant: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        # Walk the text in document order so that each [subject: X] marker
        # applies only to the params that follow it, up to the next marker.
        marker_positions = [(m.start(), self._clean_subject(m.group(1))) for m in self._SUBJECT_MARKER.finditer(text)]
        param_matches = self._collect_param_matches(text)

        def current_subject(pos: int) -> str:
            subject = "params"
            for m_pos, m_subject in marker_positions:
                if m_pos <= pos:
                    subject = m_subject
                else:
                    break
            return subject

        for pos, m in param_matches:
            # Skip if the whole match is actually a subject marker
            if self._SUBJECT_MARKER.match(m.group(0)):
                continue

            label = self._normalize_param_label(m.group(1))
            raw = self._clean_value(m.group(2))
            if not label or label in self._PARAM_STOP_LABELS:
                continue

            # Definition-line promotion: "Squarebox: 1m x 1m x 1m"
            if self._is_definition_line(m.group(1), raw):
                definition_subject = self._normalize_param_label(m.group(1))
                # Multi-value: split on x, comma, semicolon, pipe
                parts = [self._clean_value(x) for x in re.split(r"[x\u00d7,;|]", raw) if self._clean_value(x)]
                valid = [x for x in parts if self._looks_like_parameter_value(x)]
                if valid:
                    f = self._fact(definition_subject, "value", " ".join(valid), 0.85, from_assistant, "params")
                    if f: out.append(f)
                continue

            # Attribute line: use current subject (from marker, or "params")
            subject = current_subject(pos)

            # Markdown patterns keep the full value; plain patterns split on separators
            is_markdown = '**' in m.group(0)
            if is_markdown:
                parts = [raw]
            else:
                parts = [self._clean_value(x) for x in re.split(r"[,;|]", raw)] if any(x in raw for x in ",;|") else [raw]

            valid = [x for x in parts if self._looks_like_parameter_value(x)]
            for value in valid:
                f = self._fact(subject, label, value, 0.90, from_assistant, "params")
                if f: out.append(f)
        return out

    def process_turn_heuristic(self, user_text: str, assistant_text: str, turn_number: int, conversation_id: str):
        self.conversation_id = conversation_id
        facts = self.extract_fact_candidates(user_text, from_assistant=False) + self.extract_fact_candidates(assistant_text, from_assistant=True)
        # Never store entities from prose. Entities are created only for explicit fact subjects.
        subjects = {f["subject"] for f in facts if f["subject"].lower() != "conversation" and f["subject"].lower() != "params"}
        for subject in subjects:
            self.db.upsert_entity(conversation_id, subject, "unknown", {"source": "fact_subject"}, 0.80)
        results = []
        for fact in facts:
            results.append(self.db.commit_fact(conversation_id, fact["subject"], fact["predicate"], fact["object"], fact["confidence"], turn_number, fact.get("source", "heuristic")))
        return len(subjects), len(results)


class MemoryPipeline:
    """Extract only after the response is validated; optionally use the same LLM."""
    def __init__(self, db: StateDatabase, extractor: EntityExtractor, *, mode="heuristic", min_confidence=0.70, on_conflict="reconcile", llm_every_n_turns=0, llm_max_tokens=300, llm_temperature=0.0):
        self.db = db
        self.extractor = extractor
        self.mode = mode
        self.min_confidence = min_confidence
        self.on_conflict = on_conflict
        self.llm_every_n_turns = llm_every_n_turns
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    async def process_turn(self, conversation_id: str, user_text: str, assistant_text: str, turn_number: int, llm_complete=None):
        # Deterministic path first. The proxy has already performed its response
        # consistency pass for JSON requests; streaming uses this post-pass.
        return self.extractor.process_turn_heuristic(user_text, assistant_text, turn_number, conversation_id)
