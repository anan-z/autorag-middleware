from __future__ import annotations

import asyncio
from pathlib import Path

from autorag.config import AutoRAGConfig, LLMBackend
from autorag.database import StateDatabase, _simple_embed
from autorag.extractor import EntityExtractor, MemoryPipeline, strip_reasoning
from autorag.injector import ContextInjector
from autorag.proxy import AutoRAGProxy
from autorag.validator import ResponseValidator


def db(tmp_path: Path) -> StateDatabase:
    return StateDatabase(tmp_path / "state.db")


def test_conversation_isolation(tmp_path):
    d = db(tmp_path)
    d.commit_fact("home", "car keys", "location", "finger-take", turn_number=1)
    d.commit_fact("work", "car keys", "location", "kitchen counter", turn_number=1)
    assert d.get_active_fact("home", "car keys", "location")["object"] == "finger-take"
    assert d.get_active_fact("work", "car keys", "location")["object"] == "kitchen counter"
    assert d.search_facts("work", "where are my car keys")[0]["object"] == "kitchen counter"


def test_conservative_extractor_car_keys(tmp_path):
    ext = EntityExtractor(db(tmp_path), use_spacy=False)
    facts = ext.extract_fact_candidates("I left my car keys on the finger-take near the door.")
    locs = [f for f in facts if f["predicate"] == "location"]
    assert len(locs) == 1
    assert locs[0]["subject"].lower() == "car keys"
    assert "finger-take" in locs[0]["object"].lower()


def test_narrative_does_not_pollute(tmp_path):
    ext = EntityExtractor(db(tmp_path), use_spacy=False)
    text = "A beautiful sunset painted the room gold while Maya smiled and the wind lifted her red hair."
    facts = ext.extract_fact_candidates(text, from_assistant=True)
    assert not any(f["predicate"] in {"location", "association", "decision", "requirement"} for f in facts)
    assert not ext.extract_entities(text)


def test_hedged_location_is_not_memory(tmp_path):
    ext = EntityExtractor(db(tmp_path), use_spacy=False)
    facts = ext.extract_fact_candidates("I think I left my keys on the kitchen counter, maybe.")
    assert not any(f["predicate"] == "location" for f in facts)


def test_hair_fact_is_high_signal(tmp_path):
    ext = EntityExtractor(db(tmp_path), use_spacy=False)
    facts = ext.extract_fact_candidates("Maya's hair is black.")
    assert [(f["subject"], f["predicate"], f["object"]) for f in facts] == [("Maya", "hair_color", "black")]


def test_parameter_pack_is_data_shaped(tmp_path):
    ext = EntityExtractor(db(tmp_path), use_spacy=False)
    facts = ext.extract_fact_candidates("[subject: motor_a]\nWidth: 42mm\nNote: use a quiet fan\nIterations: 12")
    vals = {(f["subject"].lower(), f["predicate"], f["object"]) for f in facts}
    assert ("motor_a", "width", "42mm") in vals
    assert ("motor_a", "iterations", "12") in vals
    assert not any(f["predicate"] == "note" for f in facts)


def test_pipeline_commits_only_high_signal(tmp_path):
    d = db(tmp_path)
    ext = EntityExtractor(d, use_spacy=False)
    pipe = MemoryPipeline(d, ext)
    result = asyncio.run(pipe.process_turn("c1", "I left my car keys on the finger-take.", "A beautiful sunset painted the room gold.", 1))
    assert result[1] == 1
    assert d.fact_count("c1") == 1
    assert d.get_active_fact("c1", "car keys", "location")["object"] == "finger-take"


def test_supersede_and_rollback(tmp_path):
    d = db(tmp_path)
    d.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    d.commit_fact("c1", "Maya", "hair_color", "red", turn_number=5)
    assert d.get_active_fact("c1", "Maya", "hair_color")["object"] == "red"
    d.deactivate_facts_from_turn("c1", 5)
    assert d.get_active_fact("c1", "Maya", "hair_color")["object"] == "black"


def test_reality_check_detects_maya_conflict(tmp_path):
    d = db(tmp_path)
    d.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    v = ResponseValidator(d, EntityExtractor(d), use_llm_claims=False)
    conflicts = v.find_conflicts("c1", "A breeze swept through the room, lifting Maya's red hair.")
    assert conflicts and conflicts[0]["existing"] == "black" and conflicts[0]["generated"].lower() == "red"


def test_reality_check_ignores_matching_fact(tmp_path):
    d = db(tmp_path)
    d.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    v = ResponseValidator(d, EntityExtractor(d), use_llm_claims=False)
    assert v.find_conflicts("c1", "A breeze lifted Maya's black hair.") == []


def test_deterministic_vectors_survive_restart(tmp_path):
    first = _simple_embed("Maya black hair")
    second = _simple_embed("Maya black hair")
    assert first == second
    d = db(tmp_path)
    d.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    d2 = StateDatabase(tmp_path / "state.db")
    assert d2.get_active_fact("c1", "Maya", "hair_color")["object"] == "black"


def test_purge_only_one_conversation(tmp_path):
    d = db(tmp_path)
    d.commit_fact("a", "car keys", "location", "finger-take", turn_number=1)
    d.commit_fact("b", "car keys", "location", "desk", turn_number=1)
    d.ensure_conversation("a")
    d.ensure_conversation("b")
    deleted = d.purge_conversation("a")
    assert deleted["facts"] == 1
    assert d.get_active_fact("a", "car keys", "location") is None
    assert d.get_active_fact("b", "car keys", "location")["object"] == "desk"


def test_config_is_conservative():
    cfg = AutoRAGConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.extraction.mode == "heuristic"
    assert cfg.validation.use_llm_claims is False


def test_proxy_can_initialize(tmp_path):
    cfg = AutoRAGConfig(database_path=tmp_path / "proxy.db")
    proxy = AutoRAGProxy(cfg)
    assert proxy.db.fact_count() == 0
    asyncio.run(proxy.close())


def test_system_messages_are_preserved():
    msgs = [{"role":"system","content":"You are helpful."},{"role":"system","content":"Author note."},{"role":"user","content":"Hi"}]
    out = AutoRAGProxy._ensure_system_message(msgs, "[MEMORY]\nkeys\n[/MEMORY]")
    systems = [m for m in out if m["role"] == "system"]
    assert len(systems) == 2
    assert "MEMORY" in systems[0]["content"]
    assert systems[1]["content"] == "Author note."


def test_strip_reasoning():
    assert strip_reasoning("hello <think>private</think> world") == "hello  world"


def test_json_proxy_reality_check_can_correct_draft(tmp_path):
    import httpx
    cfg = AutoRAGConfig(database_path=tmp_path / "proxy.db")
    cfg.backends["default"] = LLMBackend(api_base="http://fake/v1", api_key="not-needed", model="local-model")
    proxy = AutoRAGProxy(cfg)
    proxy.db.commit_fact("c1", "Maya", "hair_color", "black", turn_number=1)
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            body = {"choices": [{"message": {"role": "assistant", "content": "A breeze lifted Maya's red hair."}}]}
        else:
            body = {"choices": [{"message": {"role": "assistant", "content": "A breeze lifted Maya's black hair."}}]}
        return httpx.Response(200, json=body)

    async def run():
        await proxy.client.aclose()
        proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result = await proxy._json_completion(
            "http://fake/v1/chat/completions",
            {"Content-Type": "application/json"},
            {"model": "local-model", "messages": [{"role": "user", "content": "Describe the scene."}]},
            cfg.backends["default"],
            "c1", "Describe the scene.", "", 2,
        )
        return result

    result = asyncio.run(run())
    assert result.body is not None
    assert b"black hair" in result.body
    assert len(calls) == 2
    asyncio.run(proxy.close())
