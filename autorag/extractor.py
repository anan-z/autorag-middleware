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
    HAIR_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})['’]s\s+hair\s+(?:is|was)\s+([A-Za-z-]{2,20})\b", re.I)
    HAIR2_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})\s+has\s+([A-Za-z-]{2,20})\s+hair\b", re.I)
    EYES_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})['’]s\s+eyes\s+(?:are|were)\s+([A-Za-z-]{2,20})\b", re.I)
    DECISION_RE = re.compile(r"\b(?:we|I)\s+(?:decided|agreed|settled on|chose|selected|rejected|assumed|will use|are using)\s+(?:that\s+)?(.{4,160}?)(?=[.!?]|$)", re.I)
    REQUIREMENT_RE = re.compile(r"\b(?:must|shall|required to|needs to|need to|should remain|has to)\s+(.{4,160}?)(?=[.!?]|$)", re.I)
    EXPLICIT_FACT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})\s+(?:has|owns|uses|lives in|works at|works for)\s+(.{2,100}?)(?=[.!?]|$)", re.I)

    _PARAM_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_ \-]{0,40}?)\s*[:=]\s*(.+?)\s*$", re.MULTILINE)

    # Markdown bold pattern: **Label:** value
    _PARAM_MARKDOWN = re.compile(
        r"^\s*\*\*([A-Za-z][A-Za-z0-9_ \-\/]{0,50}?)\*\*\s*[:=]\s*(.+?)\s*$",
        re.MULTILINE,
    )

    # Markdown bullet pattern: * **Label:** value
    _PARAM_BULLET = re.compile(
        r"^\s*[\*\-]\s*\*\*([A-Za-z][A-Za-z0-9_ \-\/]{0,50}?)\*\*\s*[:=]\s*(.+?)\s*$",
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
        text = re.sub(r"\【[^\】]*\】", " ", text)
        text = re.sub(r"\〔[^\〕]*\〕", " ", text)
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
        value = value.strip()
        if not value or re.match(r"^[A-Za-z][A-Za-z\s]+$", value):
            return False
        return bool(
            re.match(r"^-?\d+(?:\.\d+)?$", value)
            or re.search(r"\d+\s*[a-zA-Z°Ωµ%]+", value)
            or re.search(r"[€$£]\s*\d", value)
            or re.match(r"^v?\d+(?:\.\d+)+(?:[-_][\w-]+)?$", value)
            or re.match(r"^[A-Z]+[\w-]*\d+[\w-]*$", value)
        )

    def preprocess_message(self, text: str) -> tuple[str, str]:
        markers = list(self._SUBJECT_MARKER.finditer(text))
        if not markers:
            return text, text
        clean_parts, last_end = [], 0
        for m in markers:
            clean_parts.append(text[last_end:m.start()]); last_end = m.end()
        clean_parts.append(text[last_end:])
        return "".join(clean_parts), text

    def extract_params(self, text: str, *, from_assistant: bool) -> list[dict[str, Any]]:
        out = []
        subject = "params"
        markers = list(self._SUBJECT_MARKER.finditer(text))
        if markers:
            subject = self._clean_subject(markers[-1].group(1))

        # Collect matches from all three patterns
        all_matches = []
        all_matches.extend(self._PARAM_LINE.finditer(text))
        all_matches.extend(self._PARAM_MARKDOWN.finditer(text))
        all_matches.extend(self._PARAM_BULLET.finditer(text))

        for m in all_matches:
            label = self._normalize_param_label(m.group(1))
            raw = self._clean_value(m.group(2))
            if not label or label in self._PARAM_STOP_LABELS or self._SUBJECT_MARKER.match(m.group(0)):
                continue
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
