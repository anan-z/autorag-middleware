# AutoRAG Middleware 0.4.0

Conversation-scoped external working memory for OpenAI-compatible local LLMs.

The 0.4 design is deliberately **conservative**: polluted memory is worse than missed memory. The default extractor stores only high-signal durable facts and does not harvest arbitrary capitalized words, spaCy entities, adjectives, or narrative prose.

## What is remembered

Examples of high-signal facts:

- `I left my car keys on the finger-take.` → `car keys.location = finger-take`
- `Maya's hair is black.` → `Maya.hair_color = black`
- `We decided to use PostgreSQL.` → `conversation.decision = use PostgreSQL`
- `[subject: motor_a] Width: 42mm` → `motor_a.width = 42mm` (parameter pack)

A sentence such as `A beautiful sunset painted the room gold.` creates **no memory**.

## Conversation isolation

Every fact is scoped to a conversation. Supply a stable ID with:

```http
X-Conversation-Id: my-chat-42
```

or a body field `conversation_id` / `session_id` / `chat_id`.

Do not use the default bucket for multiple unrelated chats. A client integration should provide a stable per-chat ID.

## Reality checking

For JSON/non-streaming requests, the proxy can check the draft against existing memory before returning it. For example, if memory says `Maya.hair_color = black` and a draft says `Maya's red hair`, the configured policy can ask the same model to correct the draft.

The default semantic claim extractor is OFF because it costs another model call and can itself introduce noise. Deterministic checks remain enabled.

Streaming responses cannot be rewritten after bytes have already reached the client; their memory extraction happens after the stream. If you need hard response correction, use non-streaming mode.

## LLM extraction

If deterministic extraction misses too much, explicitly opt into the same-model structured extractor:

```yaml
extraction:
  mode: "llm"
  llm_every_n_turns: 1
```

This is intentionally not the default.

## Purging one conversation

Use the supplied script instead of an ad-hoc SQL snippet:

```powershell
python scripts/purge_conversation.py fp_bb11ae3ba671c312
```

It asks for confirmation. For automation:

```powershell
python scripts/purge_conversation.py fp_bb11ae3ba671c312 --yes
```

Optional explicit DB path:

```powershell
python scripts/purge_conversation.py fp_bb11ae3ba671c312 --db '%LOCALAPPDATA%\autorag-middleware\state.db' --yes
```

Only that conversation's entities, facts, events, relationships, conversation state, vectors, and metadata are removed.

## Quick start

```powershell
python -m venv venv
venv\\Scripts\\activate
pip install -e .
python -m autorag
```

Point LM Studio clients / SillyTavern / Open WebUI at `http://127.0.0.1:8000/v1`.

## Tests

```powershell
pip install -e ".[dev]"
pytest -q
```

## Architecture

```text
client
  │
  ▼
AutoRAG proxy
  ├── retrieve relevant facts for THIS conversation
  ├── inject compact memory
  ▼
LM Studio / Gemma
  │
  ├── optional deterministic reality check
  └── response
  │
  ▼
conservative extractor
  │
  ▼
SQLite (conversation-scoped facts)
```

The project is intended for long-running finance, product design, physics, research, everyday-assistant, and RP conversations. RP is a useful stress test, not a special-case product mode.
