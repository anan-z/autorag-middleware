"""Post-generation consistency pass — domain-neutral claim check vs memory.

RP (hair, inventory, etc.) is a stress test, not the product center.
Core path: candidate claims (heuristic + optional same-model JSON) → DB → policy.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

from .database import StateDatabase
from .extractor import EntityExtractor, strip_reasoning, _normalize_subject

logger = logging.getLogger("autorag.validator")

# Domain-neutral durable-claim patterns (cheap filter). Not RP-specific.
# Optional packs can add more via config later.
_GENERIC_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})['’]s\s+(red|black|blonde|brown|white|auburn|gray|grey|blue|green)\s+hair\b", re.I),
        "hair_color",
        "g1_g2",
    ),
    (
        re.compile(r"\b([A-Z][A-Za-z0-9_-]{1,40})['’]s\s+hair\s+(?:is|was)\s+(red|black|blonde|brown|white|auburn|gray|grey)\b", re.I),
        "hair_color",
        "g1_g2",
    ),
    # Name is N years old
    (
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+(\d{1,3})\s+years?\s+old\b"),
        "age",
        "g2",
    ),
    # X is located at / in / on Y  |  the X is on the Y
    (
        re.compile(
            r"\b(?:left|put|placed|stored|kept)\s+(?:the\s+|my\s+|our\s+)?(.{2,40}?)\s+"
            r"(?:on|in|at)\s+(?:the\s+)?(.{2,50})",
            re.I,
        ),
        "location",
        "g1_g2",
    ),
    # The reserve / budget / limit is €50k | 120 mm | v2
    (
        re.compile(
            r"\b(?:the\s+)?([A-Za-z][A-Za-z0-9_\-\s]{1,40}?)\s+"
            r"(?:is|are|was|were|must be|should be|remains?)\s+"
            r"([€$£]?\d[\d.,]*\s*(?:k|K|m|M|%|mm|cm|kg|hours?|days?)?|"
            r"v?\d+(?:\.\d+)*|[A-Za-z][A-Za-z0-9_\-]{1,30})",
            re.I,
        ),
        "value",
        "g1_g2",
    ),
    # We decided / agreed / assumed that ...
    (
        re.compile(
            r"\b(?:we\s+)?(?:decided|agreed|assumed|concluded|rejected)\s+(?:that\s+)?(.{5,80})",
            re.I,
        ),
        "decision",
        "g1_as_object",
    ),
]

# Optional RP / narrative stress-test patterns (enabled via validation.extra_patterns: rp)
_RP_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\b([A-Z][a-z]+)'s\s+(red|black|blonde|brown|white|auburn|gray|grey|blue|green)\s+hair\b"
        ),
        "hair_color",
        "g1_g2",
    ),
    (
        re.compile(
            r"\b([A-Z][a-z]+)'s\s+hair\s+(?:is|was)\s+(red|black|blonde|brown|white|auburn|gray|grey)\b"
        ),
        "hair_color",
        "g1_g2",
    ),
    (
        re.compile(
            r"\b([A-Z][a-z]+)'s\s+(blue|green|brown|hazel|gray|grey|black)\s+eyes\b"
        ),
        "eye_color",
        "g1_g2",
    ),
]



# Optional params / technical-spec patterns (enabled via validation.extra_patterns: params)
_PARAMS_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"^\s*([A-Za-z][A-Za-z0-9_ \-]{0,40}?)\s*[:=]\s*(.+?)\s*$",
            re.MULTILINE,
        ),
        "param",
        "label_value",
    ),
]

_INTENTIONAL_HINTS = re.compile(
    r"\b(now|had become|turned|changed|after the|since the|no longer|used to be|"
    r"once was|updated to|revised to|instead|we now|going forward|from now on|"
    r"dyed|painted|replaced|superseded)\b",
    re.I,
)

CLAIM_EXTRACT_SYSTEM = """You extract durable claims from an assistant draft that should stay consistent with long-term memory.
Return ONLY JSON:
{"claims":[{"subject":"string","predicate":"string","object":"string"}]}

Rules:
- Domain-agnostic: finance, engineering, science, product, everyday, or narrative.
- Durable only: decisions, constraints, quantities, identities, locations of things, assumptions, requirements.
- predicate: short snake_case (e.g. cash_reserve, fan_size, api_compat, location, age, status, assumed).
- Skip pure style, transient emotion, and one-off narration with no persistent state.
- Skip questions and hedges (maybe, I think, probably) unless clearly asserted as decided fact.
- If none: {"claims":[]}
- No markdown fences."""


def _parse_claims_json(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        logger.warning("Claim extract non-JSON: %r", text[:200])
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Claim extract JSON parse failed: %r", text[:200])
        return []
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list):
        return []
    out: list[dict[str, Any]] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        subj = _normalize_subject(str(c.get("subject") or "")) or str(c.get("subject") or "").strip()
        pred = str(c.get("predicate") or "").strip().lower().replace(" ", "_")
        obj = str(c.get("object") or "").strip()
        if not subj or not pred or not obj:
            continue
        out.append({"subject": subj, "predicate": pred, "object": obj, "span": f"{subj}.{pred}={obj}"})
    return out


class ResponseValidator:
    """Domain-neutral consistency: candidates → memory lookup → policy."""

    def __init__(
        self,
        db: StateDatabase,
        extractor: EntityExtractor | None = None,
        *,
        extra_patterns: list[str] | None = None,
        use_llm_claims: bool = True,
    ):
        self.db = db
        self.extractor = extractor
        self.extra_patterns = set(extra_patterns or [])
        self.use_llm_claims = use_llm_claims

    def _pattern_list(self) -> list[tuple[re.Pattern[str], str, str]]:
        patterns = list(_GENERIC_PATTERNS)
        if "rp" in self.extra_patterns:
            patterns.extend(_RP_PATTERNS)
        if "params" in self.extra_patterns:
            patterns.extend(_PARAMS_PATTERNS)
        return patterns

    def extract_claim_candidates(
        self,
        text: str,
        *,
        llm_claims: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        text = strip_reasoning(text or "")
        claims: list[dict[str, Any]] = []

        for pattern, predicate, mode in self._pattern_list():
            for m in pattern.finditer(text):
                try:
                    if mode == "g2":
                        subject = _normalize_subject(m.group(1)) or m.group(1).strip()
                        value = m.group(2).strip()
                    elif mode == "g1_g2":
                        subject = _normalize_subject(m.group(1)) or m.group(1).strip()
                        value = m.group(2).strip()
                    elif mode == "g1_as_object":
                        subject = "decision"
                        value = m.group(1).strip()
                        predicate = "statement"
                    else:
                        continue
                except IndexError:
                    continue
                if not subject or not value:
                    continue
                # Skip very generic subjects
                if subject.lower() in {"it", "this", "that", "there", "he", "she", "they"}:
                    continue
                claims.append(
                    {
                        "subject": subject[:80],
                        "predicate": predicate,
                        "object": value[:120],
                        "span": m.group(0)[:120],
                        "source": "pattern",
                    }
                )

        if self.extractor is not None:
            for f in self.extractor.extract_fact_candidates(text, from_assistant=True):
                claims.append(
                    {
                        "subject": f["subject"],
                        "predicate": f["predicate"],
                        "object": f["object"],
                        "span": f"{f['subject']}.{f['predicate']}={f['object']}",
                        "source": "heuristic",
                    }
                )

        if llm_claims:
            for c in llm_claims:
                claims.append({**c, "source": c.get("source", "llm")})

        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for c in claims:
            key = (c["subject"].lower(), c["predicate"].lower())
            merged[key] = c
        return list(merged.values())

    async def extract_claims_llm(
        self,
        draft: str,
        known_facts: list[dict[str, Any]],
        llm_complete: Callable[..., Awaitable[str]],
    ) -> list[dict[str, Any]]:
        known = "\n".join(
            f"- {f['subject']}.{f['predicate']} = {f['object']}" for f in known_facts[:20]
        ) or "(none)"
        messages = [
            {"role": "system", "content": CLAIM_EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"KNOWN MEMORY:\n{known}\n\n"
                    f"ASSISTANT DRAFT:\n{strip_reasoning(draft)[:3000]}\n\n"
                    "Extract durable claims as JSON."
                ),
            },
        ]
        try:
            raw = await llm_complete(messages, max_tokens=400, temperature=0.1)
        except Exception:
            logger.exception("LLM claim extract failed")
            return []
        return _parse_claims_json(raw or "")

    def find_conflicts(
        self,
        conversation_id: str,
        text: str,
        *,
        llm_claims: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        for claim in self.extract_claim_candidates(text, llm_claims=llm_claims):
            existing = self.db.get_active_fact(
                conversation_id, claim["subject"], claim["predicate"]
            )
            if not existing:
                # Also try looser subject match: token overlap on subject
                existing = self._fuzzy_fact(
                    conversation_id, claim["subject"], claim["predicate"]
                )
            if not existing:
                continue
            if existing["object"].strip().lower() == str(claim["object"]).strip().lower():
                continue
            span = claim.get("span") or ""
            conflicts.append(
                {
                    "subject": claim["subject"],
                    "predicate": claim["predicate"],
                    "existing": existing["object"],
                    "generated": claim["object"],
                    "span": span,
                    "intentional_hint": bool(
                        _INTENTIONAL_HINTS.search(span)
                        or _INTENTIONAL_HINTS.search(text or "")
                    ),
                    "matched_subject": existing.get("subject", claim["subject"]),
                }
            )
        return conflicts

    def _fuzzy_fact(
        self, conversation_id: str, subject: str, predicate: str
    ) -> dict[str, Any] | None:
        """Match fact if predicate matches and subjects share a meaningful token."""
        pred = predicate.lower().replace(" ", "_")
        sub_tokens = set(re.findall(r"[a-z0-9]{3,}", subject.lower()))
        if not sub_tokens:
            return None
        for f in self.db.list_active_facts(conversation_id, limit=100):
            if f["predicate"].lower() != pred:
                continue
            ft = set(re.findall(r"[a-z0-9]{3,}", f["subject"].lower()))
            if sub_tokens & ft:
                return f
        return None

    def validate(
        self,
        conversation_id: str,
        ai_response: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        conflicts = self.find_conflicts(conversation_id, ai_response)
        errors = [
            (
                f"{c['subject']}.{c['predicate']} is recorded as {c['existing']!r}, "
                f"but the reply says {c['generated']!r}."
            )
            for c in conflicts
            if not c.get("intentional_hint")
        ]
        return len(errors) == 0, errors

    def build_correction_prompt(
        self,
        original_system: str,
        errors: list[str],
        *,
        policy: str = "hard",
    ) -> str:
        bullets = "\n".join(f"- {e}" for e in errors)
        if policy == "soft":
            body = (
                "[MEMORY CONFLICT — soft]\n"
                "The draft conflicts with established long-term memory for this conversation. "
                "If this is an intentional update, keep it; otherwise stay consistent with:\n"
                f"{bullets}\n"
                "[/MEMORY CONFLICT]"
            )
        else:
            body = (
                "[SYSTEM CORRECTION]\n"
                "Your draft contradicts established conversation memory. "
                "Regenerate so you stay consistent with these facts "
                "(unless the text clearly marks an intentional change):\n"
                f"{bullets}\n"
                "[/SYSTEM CORRECTION]"
            )
        if original_system.strip():
            return f"{original_system.strip()}\n\n{body}"
        return body

    async def reality_check(
        self,
        conversation_id: str,
        draft: str,
        *,
        policy: str = "hard",
        llm_complete: Callable[..., Awaitable[str]] | None = None,
        original_messages: list[dict] | None = None,
    ) -> dict[str, Any]:
        if policy in ("off", None, ""):
            return {"ok": True, "conflicts": [], "text": draft, "action": "skip"}

        llm_claims: list[dict[str, Any]] = []
        if self.use_llm_claims and llm_complete is not None:
            known = self.db.list_active_facts(conversation_id, limit=20)
            llm_claims = await self.extract_claims_llm(draft, known, llm_complete)

        conflicts = self.find_conflicts(
            conversation_id, draft, llm_claims=llm_claims or None
        )
        if not conflicts:
            return {"ok": True, "conflicts": [], "text": draft, "action": "none"}

        hard_conflicts = [c for c in conflicts if not c.get("intentional_hint")]
        intentional = [c for c in conflicts if c.get("intentional_hint")]

        if policy == "flag":
            for c in conflicts:
                logger.info(
                    "Response conflict %s.%s: %r vs %r",
                    c["subject"],
                    c["predicate"],
                    c["existing"],
                    c["generated"],
                )
            return {
                "ok": len(hard_conflicts) == 0,
                "conflicts": conflicts,
                "text": draft,
                "action": "flag",
            }

        if not hard_conflicts and intentional:
            return {
                "ok": True,
                "conflicts": conflicts,
                "text": draft,
                "action": "intentional_accept",
            }

        if llm_complete is None or not original_messages:
            return {
                "ok": False,
                "conflicts": conflicts,
                "text": draft,
                "action": "flag_no_llm",
            }

        if policy == "reconcile":
            resolved = await self._reconcile_response(
                draft, conflicts, llm_complete, original_messages
            )
            return {
                "ok": True,
                "conflicts": conflicts,
                "text": resolved or draft,
                "action": "reconcile",
            }

        errors = [
            f"{c['subject']}.{c['predicate']}: memory={c['existing']!r}, draft={c['generated']!r}"
            for c in hard_conflicts or conflicts
        ]
        correction = self.build_correction_prompt("", errors, policy=policy)
        messages = list(original_messages) + [{"role": "system", "content": correction}]
        try:
            new_text = await llm_complete(messages, max_tokens=1024, temperature=0.35)
        except Exception:
            logger.exception("Reality-check regenerate failed")
            return {
                "ok": False,
                "conflicts": conflicts,
                "text": draft,
                "action": "regenerate_failed",
            }

        again = self.find_conflicts(conversation_id, new_text or "")
        still_hard = [c for c in again if not c.get("intentional_hint")]
        return {
            "ok": len(still_hard) == 0,
            "conflicts": conflicts,
            "remaining": again,
            "text": new_text or draft,
            "action": policy,
        }

    async def _reconcile_response(
        self,
        draft: str,
        conflicts: list[dict[str, Any]],
        llm_complete: Callable[..., Awaitable[str]],
        original_messages: list[dict],
    ) -> str | None:
        bullets = "\n".join(
            f"- {c['subject']}.{c['predicate']}: memory={c['existing']!r}, draft={c['generated']!r}"
            f"{' (possible intentional change)' if c.get('intentional_hint') else ''}"
            for c in conflicts
        )
        prompt = (
            "Your previous draft conflicts with long-term conversation memory.\n"
            f"{bullets}\n\n"
            "Rewrite the reply to stay consistent with established memory, "
            "unless the draft clearly narrates an intentional change—then keep the change.\n"
            "Output ONLY the revised reply, no preamble."
        )
        messages = list(original_messages) + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": prompt},
        ]
        try:
            return await llm_complete(messages, max_tokens=1024, temperature=0.3)
        except Exception:
            logger.exception("Reconcile response failed")
            return None
