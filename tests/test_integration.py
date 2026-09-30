"""Tests for AutoRAG 0.2.2."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autorag.config import AutoRAGConfig
from autorag.database import StateDatabase, _simple_embed
from autorag.extractor import (
    EntityExtractor,
    MemoryPipeline,
    _normalize_subject,
    _parse_facts_json,
    strip_reasoning,
)
from autorag.injector import ContextInjector
from autorag.proxy import AutoRAGProxy


@pytest.fixture
def db(tmp_path: Path):
    return StateDatabase(tmp_path / "test.db")


def test_conversation_isolation(db: StateDatabase):
    db.commit_fact("conv_a", "car keys", "location", "finger-take", turn_number=1)
    db.commit_fact("conv_b", "car keys", "location", "kitchen counter", turn_number=1)
    a = db.get_active_fact("conv_a", "car keys", "location")
    b = db.get_active_fact("conv_b", "car keys", "location")
    assert a["object"] == "finger-take"
    assert b["object"] == "kitchen counter"


def test_supersede_and_rollback(db: StateDatabase):
    db.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    db.commit_fact("c1", "Maya", "hair_color", "red", turn_number=5)
    assert db.get_active_fact("c1", "Maya", "hair_color")["object"] == "red"
    n = db.deactivate_facts_from_turn("c1", 5)
    assert n >= 1
    # after rollback of turn 5+, black should still be active if not deactivated
    # (black was turn 1; only turn>=5 deactivated — red gone, black still active)
    active = db.get_active_fact("c1", "Maya", "hair_color")
    assert active is not None
    assert active["object"] == "black"


def test_normalize_subject():
    assert _normalize_subject("my car keys") == "Car Keys" or _normalize_subject("my car keys").lower() == "car keys"
    assert _normalize_subject("he") == ""
    assert _normalize_subject("the keys") != ""


def test_no_re_i_on_tired(db: StateDatabase):
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_fact_candidates("I'm Tired today")
    # should not invent Tired as subject of age/location
    assert not any(f["subject"].lower() == "tired" for f in facts)
    ents = ext.extract_entities("I'm Tired today")
    # "Tired" may appear as weak entity but confidence low / stop list
    strong = [e for e in ents if e["name"] == "Tired" and e["confidence"] >= 0.55]
    assert strong == []


def test_hedge_skips_speculation(db: StateDatabase):
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_fact_candidates(
        "I think I left my keys on the kitchen counter, maybe."
    )
    # hedge window should skip
    assert not any(f["predicate"] == "location" for f in facts)


def test_location_assertion(db: StateDatabase):
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_fact_candidates(
        "I left my car keys on the finger-take near the door."
    )
    locs = [f for f in facts if f["predicate"] == "location"]
    assert locs
    assert any("key" in f["subject"].lower() for f in locs)


def test_no_owner_from_possessive_dog(db: StateDatabase):
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_fact_candidates("Maya's dog barked at John's car.")
    assert not any(f["predicate"] == "owner" for f in facts)


def test_parse_facts_json():
    raw = '{"facts":[{"subject":"car keys","predicate":"location","object":"home","confidence":0.9}]}'
    facts = _parse_facts_json(raw)
    assert facts[0]["subject"].lower().find("key") >= 0 or facts[0]["subject"]


def test_strip_reasoning():
    t = strip_reasoning("Hello <think>secret</think> world")
    assert "secret" not in t
    assert "Hello" in t and "world" in t


def test_search_facts_token_score(db: StateDatabase):
    db.commit_fact("c1", "car keys", "location", "finger-take", turn_number=1)
    db.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    hits = db.search_facts("c1", "Where did I leave my car keys?")
    assert hits
    assert hits[0]["subject"].lower().find("key") >= 0 or "key" in hits[0]["object"].lower() or hits[0]["predicate"] == "location"


def test_embed_stable():
    assert _simple_embed("Maya") == _simple_embed("Maya")


def test_config_defaults():
    cfg = AutoRAGConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.extraction.mode in ("heuristic", "llm", "hybrid")


def test_ensure_system_keeps_later_system_messages():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "system", "content": "Author note: be brief."},
        {"role": "user", "content": "Hi"},
    ]
    out = AutoRAGProxy._ensure_system_message(msgs, "[MEMORY]\nx\n[/MEMORY]")
    system_msgs = [m for m in out if m["role"] == "system"]
    assert len(system_msgs) == 2
    assert "MEMORY" in system_msgs[0]["content"]
    assert system_msgs[1]["content"] == "Author note: be brief."


def test_user_turn_index():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert AutoRAGProxy._user_turn_index(msgs) == 2


def test_fingerprint_stable():
    msgs = [
        {"role": "system", "content": "You are Maya."},
        {"role": "user", "content": "Hello"},
    ]
    a = AutoRAGProxy._fingerprint_messages(msgs)
    b = AutoRAGProxy._fingerprint_messages(msgs)
    assert a == b
    assert a.startswith("fp_")


def test_injector_label(db: StateDatabase):
    db.commit_fact("c1", "keys", "location", "table", turn_number=1)
    block = ContextInjector(db, AutoRAGConfig()).build_system_prompt("c1", "where are keys")
    assert "ESTABLISHED FACTS" in block or "keys" in block


def test_pipeline_heuristic(db: StateDatabase):
    ext = EntityExtractor(db, use_spacy=False, min_confidence=0.5)
    pipe = MemoryPipeline(db, ext, mode="heuristic", on_conflict="flag")
    result = asyncio.run(
        pipe.process_turn(
            "rp1",
            "I left my wallet on the desk.",
            "Got it.",
            turn_number=1,
            llm_complete=None,
        )
    )
    assert isinstance(result["facts_committed"], list)


def test_pipeline_conflict_keep(db: StateDatabase):
    db.commit_fact("rp1", "Maya", "hair_color", "black", confidence=0.95, turn_number=1)

    async def fake_llm(messages, max_tokens=150, temperature=0.0):
        return '{"action":"keep_existing","object":"black","confidence":0.9}'

    pipe = MemoryPipeline(
        db, EntityExtractor(db), mode="heuristic", on_conflict="reconcile"
    )
    result = asyncio.run(
        pipe._validate_and_commit(
            "rp1",
            {
                "subject": "Maya",
                "predicate": "hair_color",
                "object": "red",
                "confidence": 0.8,
                "source": "heuristic",
            },
            turn_number=10,
            llm_complete=fake_llm,
        )
    )
    assert result.get("status") == "rejected"
    assert db.get_active_fact("rp1", "Maya", "hair_color")["object"] == "black"


def test_find_hair_conflict(db: StateDatabase):
    from autorag.validator import ResponseValidator
    from autorag.extractor import EntityExtractor

    db.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    v = ResponseValidator(db, EntityExtractor(db), extra_patterns=["rp"])
    conflicts = v.find_conflicts(
        "c1", "A breeze swept through the room, lifting Maya's red hair."
    )
    assert conflicts
    assert conflicts[0]["existing"] == "black"
    assert conflicts[0]["generated"].lower() == "red"


def test_no_conflict_matching_hair(db: StateDatabase):
    from autorag.validator import ResponseValidator
    from autorag.extractor import EntityExtractor

    db.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    v = ResponseValidator(db, EntityExtractor(db), extra_patterns=["rp"])
    conflicts = v.find_conflicts(
        "c1", "A breeze lifted Maya's black hair."
    )
    assert conflicts == []


def test_sunset_not_checked(db: StateDatabase):
    from autorag.validator import ResponseValidator

    v = ResponseValidator(db)
    claims = v.extract_claim_candidates(
        "The beautiful sunset painted the room gold."
    )
    assert claims == []


def test_reality_check_hard(db: StateDatabase):
    from autorag.validator import ResponseValidator
    from autorag.extractor import EntityExtractor

    db.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    v = ResponseValidator(db, EntityExtractor(db), extra_patterns=["rp"], use_llm_claims=False)

    async def fake_llm(messages, max_tokens=1024, temperature=0.4):
        return "A breeze swept through the room, lifting Maya's black hair."

    result = asyncio.run(
        v.reality_check(
            "c1",
            "A breeze swept through the room, lifting Maya's red hair.",
            policy="hard",
            llm_complete=fake_llm,
            original_messages=[{"role": "user", "content": "Describe the scene."}],
        )
    )
    assert "black" in result["text"].lower()
    assert result["action"] == "hard"


def test_finance_value_conflict(db: StateDatabase):
    from autorag.validator import ResponseValidator

    db.commit_fact("fin", "cash reserve", "value", "50000", turn_number=1)
    v = ResponseValidator(db, use_llm_claims=False)
    # LLM claims path would catch natural language; inject claim directly via find with llm_claims
    conflicts = v.find_conflicts(
        "fin",
        "We should keep the cash reserve at 100000 euros.",
        llm_claims=[{
            "subject": "cash reserve",
            "predicate": "value",
            "object": "100000",
            "span": "cash reserve value 100000",
        }],
    )
    assert conflicts
    assert conflicts[0]["existing"] == "50000"


def test_decision_intentional_hint(db: StateDatabase):
    from autorag.validator import ResponseValidator

    db.commit_fact("eng", "api", "compat", "v2", turn_number=1)
    v = ResponseValidator(db, use_llm_claims=False)
    conflicts = v.find_conflicts(
        "eng",
        "Going forward the api compat is v3.",
        llm_claims=[{
            "subject": "api",
            "predicate": "compat",
            "object": "v3",
            "span": "Going forward the api compat is v3",
        }],
    )
    assert conflicts
    assert conflicts[0]["intentional_hint"] is True


# --- Params pattern pack tests (0.3.3) ---

def test_params_single_value(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_params("Width: 10\nLength: 20", from_assistant=False)
    assert any(f["predicate"] == "width" and f["object"] == "10" for f in facts)
    assert any(f["predicate"] == "length" and f["object"] == "20" for f in facts)


def test_params_multi_value(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_params("Dimensions: 42mm, 42mm, 48mm", from_assistant=False)
    dims = [f for f in facts if f["predicate"] == "dimensions"]
    assert len(dims) == 3  # Multiple facts, not JSON blob
    assert any(f["object"] == "42mm" for f in dims)
    assert any(f["object"] == "48mm" for f in dims)


def test_params_noise_rejection(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_params("Note: see above\nAyanna: she smiles", from_assistant=False)
    assert len(facts) == 0


def test_params_unit_preservation(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_params("Voltage: 12V", from_assistant=False)
    assert facts[0]["object"] == "12V"


def test_params_supersession(db: StateDatabase):
    db.commit_fact("c1", "params", "width", "10", turn_number=1)
    db.commit_fact("c1", "params", "width", "12", turn_number=2)
    assert db.get_active_fact("c1", "params", "width")["object"] == "12"


def test_params_label_normalization(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_params("Step Angle: 1.8°", from_assistant=False)
    assert facts[0]["predicate"] == "step_angle"


def test_params_subject_marker(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    text = "[subject: motor_a]\nWidth: 10\n[subject: motor_b]\nWidth: 15"
    facts = ext.extract_params(text, from_assistant=False)

    motor_a_width = [f for f in facts if f["subject"] == "motor_a" and f["predicate"] == "width"]
    motor_b_width = [f for f in facts if f["subject"] == "motor_b" and f["predicate"] == "width"]

    assert len(motor_a_width) == 1 and motor_a_width[0]["object"] == "10"
    assert len(motor_b_width) == 1 and motor_b_width[0]["object"] == "15"


def test_params_preprocess_strips_markers(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    text = "[subject: motor_a]\nWidth: 10"
    clean, original = ext.preprocess_message(text)

    # Clean text should not contain marker
    assert "[subject:" not in clean
    assert "Width: 10" in clean

    # Original text retains marker for extraction
    assert "[subject: motor_a]" in original


def test_params_small_integers_accepted(db: StateDatabase):
    from autorag.extractor import EntityExtractor
    ext = EntityExtractor(db, use_spacy=False)
    facts = ext.extract_params("Iterations: 12\nCount: 5", from_assistant=False)
    assert any(f["predicate"] == "iterations" and f["object"] == "12" for f in facts)
    assert any(f["predicate"] == "count" and f["object"] == "5" for f in facts)


def test_params_in_validator(db: StateDatabase):
    from autorag.validator import ResponseValidator
    from autorag.extractor import EntityExtractor

    db.commit_fact("c1", "params", "voltage", "12V", turn_number=1)
    v = ResponseValidator(db, EntityExtractor(db), extra_patterns=["params"])
    patterns = v._pattern_list()
    # Should have generic + params patterns
    assert len(patterns) > len(__import__('autorag.validator', fromlist=['_GENERIC_PATTERNS'])._GENERIC_PATTERNS)
