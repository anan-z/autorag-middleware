"""OpenAI-compatible FastAPI proxy with conversation-scoped memory (0.2.2)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import AutoRAGConfig, LLMBackend
from .database import StateDatabase
from .extractor import EntityExtractor, MemoryPipeline, strip_reasoning
from .injector import ContextInjector
from .validator import ResponseValidator

logger = logging.getLogger("autorag.proxy")

_proxy: "AutoRAGProxy | None" = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proxy
    config = AutoRAGConfig.load(os.environ.get("AUTORAG_CONFIG"))
    _proxy = AutoRAGProxy(config)
    logger.info(
        "AutoRAG %s ready  db=%s  backends=%s  extract=%s  conflict=%s",
        __version__,
        config.resolved_db_path(),
        list(config.backends.keys()),
        config.extraction.mode,
        config.validation.on_conflict,
    )
    yield
    if _proxy is not None:
        await _proxy.close()
        _proxy = None


app = FastAPI(
    title="AutoRAG Middleware",
    version=__version__,
    description="Conversation-scoped OpenAI-compatible memory proxy",
    lifespan=lifespan,
)


class AutoRAGProxy:
    def __init__(self, config: AutoRAGConfig):
        self.config = config
        self.db = StateDatabase(config.resolved_db_path())
        self.extractor = EntityExtractor(
            self.db,
            use_spacy=config.extraction.use_spacy,
            min_confidence=config.extraction.min_confidence,
        )
        self.pipeline = MemoryPipeline(
            self.db,
            self.extractor,
            mode=config.extraction.mode,
            min_confidence=config.extraction.min_confidence,
            on_conflict=config.validation.on_conflict,
            llm_every_n_turns=config.extraction.llm_every_n_turns,
            llm_max_tokens=config.extraction.llm_max_tokens,
            llm_temperature=config.extraction.llm_temperature,
        )
        self.injector = ContextInjector(self.db, config)
        self.validator = ResponseValidator(
            self.db,
            extractor=self.extractor,
            extra_patterns=list(config.validation.extra_patterns),
            use_llm_claims=config.validation.use_llm_claims,
        )
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
        # Per-conversation locks so background extract finishes before next turn uses memory
        self._conv_locks: dict[str, asyncio.Lock] = {}
        self._bg_tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        # Wait briefly for background tasks
        pending = [t for t in self._bg_tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=30)
        await self.client.aclose()

    def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        if conversation_id not in self._conv_locks:
            self._conv_locks[conversation_id] = asyncio.Lock()
        return self._conv_locks[conversation_id]

    def _resolve_backend(self, model: str | None) -> LLMBackend | None:
        model = model or "default"
        backend = self.config.backends.get(model)
        if backend is None:
            backend = self.config.backends.get("default")
            if backend is None and self.config.backends:
                backend = next(iter(self.config.backends.values()))
        return backend

    @staticmethod
    def _fingerprint_messages(messages: list[dict]) -> str:
        """Stable chat id when client sends no conversation_id."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role")
            if role not in ("system", "user"):
                continue
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            parts.append(f"{role}:{str(content)[:400]}")
            if role == "user" and len([p for p in parts if p.startswith("user:")]) >= 1:
                # first system (if any) + first user is enough
                if any(p.startswith("system:") for p in parts) or len(parts) >= 1:
                    break
        raw = "\n".join(parts) or "empty"
        return "fp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _conversation_id(self, request: Request, body: dict, messages: list[dict]) -> str:
        hdr = request.headers.get("x-conversation-id") or request.headers.get(
            "x-session-id"
        )
        if hdr and hdr.strip():
            return hdr.strip()
        for key in ("conversation_id", "session_id", "chat_id"):
            val = body.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        meta = body.get("metadata") or {}
        if isinstance(meta, dict):
            for key in ("conversation_id", "session_id", "chat_id"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        # Prefer fingerprint over shared "default" bucket when messages exist
        if messages:
            return self._fingerprint_messages(messages)
        return self.config.default_conversation_id

    @staticmethod
    def _user_turn_index(messages: list[dict]) -> int:
        """1-based count of user messages = logical turn for this request."""
        n = sum(1 for m in messages if m.get("role") == "user")
        return max(1, n)


    def _resolve_turn(self, conversation_id: str, messages: list[dict]) -> int:
        """Map request to a turn index; only rollback on regenerate/rewind.

        - Full history (assistant msgs or multiple user msgs): turn = user-message count.
          If that turn is <= max stored turn, treat as regenerate/swipe and deactivate.
        - Single user message, no history (typical curl/API one-shot): allocate next
          sequential turn — do NOT wipe prior facts.
        """
        user_count = sum(1 for m in messages if m.get("role") == "user")
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        max_turn = self.db.max_turn(conversation_id)

        if has_assistant or user_count > 1:
            turn = max(1, user_count)
            if turn <= max_turn:
                n = self.db.deactivate_facts_from_turn(conversation_id, turn)
                if n:
                    logger.info(
                        "Rollback %s: deactivated %d facts from turn >= %d",
                        conversation_id,
                        n,
                        turn,
                    )
            return turn

        # One-shot / no history
        return max_turn + 1 if max_turn >= 0 else 1

    @staticmethod
    def _last_user_message(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, list):
                    parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    return " ".join(parts)
                return str(content)
        return ""

    @staticmethod
    def _ensure_system_message(messages: list[dict], system_content: str) -> list[dict]:
        """Merge memory into the *first* system message; keep all other system messages."""
        if not system_content.strip():
            return list(messages)
        out: list[dict] = []
        merged = False
        for msg in messages:
            if msg.get("role") == "system" and not merged:
                existing = msg.get("content") or ""
                combined = (
                    f"{existing}\n\n{system_content}".strip()
                    if existing
                    else system_content
                )
                out.append({**msg, "content": combined})
                merged = True
            else:
                out.append(msg)
        if not merged:
            out.insert(0, {"role": "system", "content": system_content})
        return out

    async def _llm_complete(
        self,
        backend: LLMBackend,
        messages: list[dict],
        *,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> str:
        url = backend.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {backend.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": backend.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def _make_llm_fn(self, backend: LLMBackend):
        async def fn(messages, max_tokens=400, temperature=0.1):
            return await self._llm_complete(
                backend, messages, max_tokens=max_tokens, temperature=temperature
            )
        return fn

    async def _post_process(
        self,
        backend: LLMBackend,
        conversation_id: str,
        user_message: str,
        ai_message: str,
        turn_number: int,
    ) -> None:
        if not self.config.extraction.enabled:
            return
        llm_fn = None
        if self.config.extraction.mode in ("llm", "hybrid"):
            llm_fn = self._make_llm_fn(backend)
        async with self._lock_for(conversation_id):
            try:
                await self.pipeline.process_turn(
                    conversation_id,
                    user_message,
                    ai_message,
                    turn_number,
                    llm_complete=llm_fn,
                )
            except Exception:
                logger.exception("Memory pipeline failed for %s", conversation_id)

    def _schedule_post_process(
        self,
        backend: LLMBackend,
        conversation_id: str,
        user_message: str,
        ai_message: str,
        turn_number: int,
    ) -> None:
        task = asyncio.create_task(
            self._post_process(
                backend, conversation_id, user_message, ai_message, turn_number
            )
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def handle_chat_completion(self, request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        model = body.get("model")
        backend = self._resolve_backend(model)
        if backend is None:
            return JSONResponse(
                {"error": f"No backend configured for model '{model}'."},
                status_code=400,
            )

        messages: list[dict] = list(body.get("messages") or [])
        conversation_id = self._conversation_id(request, body, messages)
        self.db.ensure_conversation(conversation_id)

        turn_number = self._resolve_turn(conversation_id, messages)

        # Wait if prior turn's background extract still running for this chat
        lock = self._lock_for(conversation_id)
        if lock.locked():
            await lock.acquire()
            lock.release()

        user_message = self._last_user_message(messages)
        stream = bool(body.get("stream", False))

        enhanced = self.injector.build_system_prompt(
            conversation_id, user_message, messages[:-1] if messages else None
        )
        messages = self._ensure_system_message(messages, enhanced)

        target_url = backend.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {backend.api_key}",
            "Content-Type": "application/json",
        }
        payload = {**body, "messages": messages, "model": backend.model}
        for k in ("conversation_id", "session_id", "chat_id"):
            payload.pop(k, None)

        if stream:
            return await self._stream_completion(
                target_url,
                headers,
                payload,
                backend,
                conversation_id,
                user_message,
                enhanced,
                turn_number,
            )
        return await self._json_completion(
            target_url,
            headers,
            payload,
            backend,
            conversation_id,
            user_message,
            enhanced,
            turn_number,
        )

    async def _json_completion(
        self,
        url: str,
        headers: dict,
        payload: dict,
        backend: LLMBackend,
        conversation_id: str,
        user_message: str,
        system_block: str,
        turn_number: int,
    ):
        try:
            resp = await self.client.post(
                url, json={**payload, "stream": False}, headers=headers
            )
        except httpx.RequestError as exc:
            logger.exception("Backend request failed")
            return JSONResponse({"error": f"Backend unreachable: {exc}"}, status_code=502)

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = {"error": resp.text}
            return JSONResponse(detail, status_code=resp.status_code)

        result = resp.json()
        ai_message = ""
        try:
            ai_message = result["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            pass

        policy = self.config.validation.response_policy
        if (
            self.config.validation.enabled
            and ai_message
            and policy not in ("off", None, "")
        ):
            llm_fn = self._make_llm_fn(backend)
            check = await self.validator.reality_check(
                conversation_id,
                ai_message,
                policy=policy,
                llm_complete=llm_fn,
                original_messages=list(payload["messages"]),
            )
            if check.get("text") and check["text"] != ai_message:
                ai_message = check["text"]
                # Patch returned payload content
                try:
                    result["choices"][0]["message"]["content"] = ai_message
                except (KeyError, IndexError, TypeError):
                    pass
                logger.info(
                    "Reality check action=%s conflicts=%d",
                    check.get("action"),
                    len(check.get("conflicts") or []),
                )

        self._schedule_post_process(
            backend, conversation_id, user_message, ai_message, turn_number
        )
        return JSONResponse(result)

    async def _stream_completion(
        self,
        url: str,
        headers: dict,
        payload: dict,
        backend: LLMBackend,
        conversation_id: str,
        user_message: str,
        system_block: str,
        turn_number: int,
    ):
        async def event_generator() -> AsyncIterator[bytes]:
            accumulated: list[str] = []
            try:
                async with self.client.stream(
                    "POST", url, json={**payload, "stream": True}, headers=headers
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        err = {
                            "error": {
                                "message": body.decode("utf-8", errors="replace"),
                                "type": "backend_error",
                                "code": resp.status_code,
                            }
                        }
                        yield f"data: {json.dumps(err)}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                break
                            try:
                                chunk = json.loads(data)
                                for choice in chunk.get("choices") or []:
                                    delta = choice.get("delta") or {}
                                    if delta.get("content"):
                                        accumulated.append(delta["content"])
                            except json.JSONDecodeError:
                                pass
                            yield (line + "\n\n").encode()
                        else:
                            yield (line + "\n").encode()
            except httpx.RequestError as exc:
                err = {
                    "error": {
                        "message": f"Backend unreachable: {exc}",
                        "type": "connection_error",
                    }
                }
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            # Schedule after stream ends — client already has [DONE]
            ai_message = strip_reasoning("".join(accumulated))
            self._schedule_post_process(
                backend, conversation_id, user_message, ai_message, turn_number
            )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if _proxy is None:
        return JSONResponse({"error": "Proxy not initialized"}, status_code=503)
    return await _proxy.handle_chat_completion(request)


@app.get("/v1/models")
async def list_models():
    if _proxy is None:
        return JSONResponse({"object": "list", "data": []})
    data = [
        {"id": name, "object": "model", "owned_by": "autorag"}
        for name in _proxy.config.backends.keys()
    ]
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": __version__,
        "entities": _proxy.db.entity_count() if _proxy else 0,
        "facts": _proxy.db.fact_count() if _proxy else 0,
        "vec_enabled": bool(_proxy and _proxy.db._vec_available),
        "extraction_mode": _proxy.config.extraction.mode if _proxy else None,
    }


@app.get("/v1/state/entities")
async def list_entities(conversation_id: str = "default", q: str = "", limit: int = 20):
    if _proxy is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    ents = _proxy.db.search_entities(conversation_id, q, limit=min(limit, 100))
    return {"conversation_id": conversation_id, "entities": ents, "count": len(ents)}


@app.get("/v1/state/facts")
async def list_facts(conversation_id: str = "default", q: str = "", limit: int = 50):
    if _proxy is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    if q.strip():
        facts = _proxy.db.search_facts(conversation_id, q, limit=min(limit, 100))
    else:
        facts = _proxy.db.list_active_facts(conversation_id, limit=min(limit, 100))
    return {"conversation_id": conversation_id, "facts": facts, "count": len(facts)}


@app.get("/v1/state/conversations")
async def list_conversations():
    if _proxy is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return {"conversations": _proxy.db.list_conversations()}
