# AutoRAG Middleware 0.3.1

**Domain-neutral long-conversation memory** (finance, product design, physics, research, everyday assistant).  
Interactive fiction / RP is an optional stress pack (`validation.extra_patterns: [rp]`), not the core product.

### Reality check (post-generation)

1. Draft from the model  
2. **LLM claim extract** (same backend, JSON) + generic durable-claim patterns  
3. Compare to conversation-scoped SQLite facts  
4. Policy: `hard` | `soft` | `reconcile` | `flag` | `off`  
5. Background extract → validate → commit  

RP-only patterns (hair/eyes) are **off by default**. Enable with:

```yaml
validation:
  use_llm_claims: true
  extra_patterns: [rp]   # optional narrative stress-test patterns
  response_policy: hard
```

# AutoRAG Middleware 0.2.2

Hardening over 0.2.1 (Claude review items):

- Keep **all** system messages (only merge memory into the first)
- **Background** extract/reconcile (reply returns immediately; per-conversation lock)
- **Regenerate/swipe rollback**: deactivate facts for current turn+, restore superseded
- **Per-request turn** = user-message count (not a global counter)
- Extractor cleanup: no `re.I` on names, no possessive→owner noise, skip hedges, normalize subjects
- Token-scored fact retrieval; fingerprint chat id when no `X-Conversation-Id`
- Strip `<think>` before extraction; lifespan handler; correction prompt no longer doubles memory

# AutoRAG Middleware 0.2.1

Patch over 0.2.0: `--config` reaches the FastAPI process, deterministic embeddings (no `hash()` seed drift), default bind `127.0.0.1`, config fall-through when an explicit path is missing.

# AutoRAG Middleware 0.2.0

OpenAI-compatible **proxy** between your chat client (SillyTavern, Open WebUI, LMSA, …) and an LLM backend (LM Studio, Ollama, cloud).

**What 0.2.0 adds**

- **Conversation-scoped memory** — no cross-chat contamination  
- **Structured facts** (`subject.predicate = object`) with supersession  
- **Extract → Validate → Commit** pipeline  
- **Optional same-model structured extraction** (hybrid / llm modes)  
- **Conflict reconcile** via a tiny second call to the same backend  
- Streaming + JSON completions  

## Architecture

```
Client  ──►  AutoRAG Proxy  ──►  LLM Backend (Gemma / etc.)
                │                      │
                │                      │ optional extract / reconcile
                ▼                      │
         SQLite (per conversation)  ◄──┘
         entities + facts + vectors
```

### Memory lifecycle

1. **Inject** relevant facts/entities for the active conversation into the system prompt  
2. **Generate** (stream or JSON) via your backend  
3. **Extract** durable facts (heuristic and/or same-model JSON)  
4. **Validate** against existing facts; optional reconcile call on conflict  
5. **Commit** (or keep existing / supersede)

## Quick start

```bash
unzip autorag-middleware.zip
cd autorag-middleware

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -e .
python -m autorag
```

Point the client at `http://127.0.0.1:8000/v1`.

### Isolate chats

Send header:

```http
X-Conversation-Id: my-rp-session-42
```

Or body field `conversation_id` / `session_id`.  
Default id is `default` (everything shares one bucket if you omit it).

### Config (`configs/default.yaml` or platformdirs config path)

```yaml
extraction:
  mode: hybrid          # heuristic | llm | hybrid
  llm_every_n_turns: 1

validation:
  enabled: true
  on_conflict: reconcile  # off | flag | reconcile

backends:
  default:
    api_base: "http://127.0.0.1:1234/v1"
    api_key: "not-needed"
    model: "your-model-id"
```

| `extraction.mode` | Behavior |
|---|---|
| `heuristic` | Regex/patterns only — no extra LLM calls |
| `llm` | Structured JSON extract via same backend |
| `hybrid` | Heuristic + LLM when candidates exist (or every N turns) |

| `validation.on_conflict` | Behavior |
|---|---|
| `off` | Always supersede with the new value |
| `flag` | Log conflict, still supersede |
| `reconcile` | Extra same-model call; may keep existing |

## Inspect memory

```bash
curl "http://127.0.0.1:8000/v1/state/facts?conversation_id=default"
curl "http://127.0.0.1:8000/v1/state/entities?conversation_id=default"
curl "http://127.0.0.1:8000/health"
```

## Example (car keys)

1. Chat (with `X-Conversation-Id: home`): *“I can’t find my car keys.”* → model replies they are on the finger-take.  
2. Hybrid extract stores `car keys.location = finger-take …` under conversation `home`.  
3. Many turns later: *“Where did I leave my car keys?”* → injector puts that fact into the system block → model answers from memory.  
4. Conversation `work` can store a different location for the same subject without clobbering `home`.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Limitations

- LLM extract quality depends on your local model following the JSON schema.  
- Heuristics alone will miss many free-form facts.  
- Streaming path does not run the *response rewrite* continuity pass (fact pipeline still runs after the stream).  
- Not a full lorebook manager — it is externalized working memory.  

## License

MIT
