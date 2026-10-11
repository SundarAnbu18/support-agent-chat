# Phase 8 — After it works

Only start these once phases 0–7 are green. Each is small and independent.

## 8.1 See inside the graph: LangSmith tracing (15 minutes)
Add to `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=support-agent
```
and `os.environ` must actually contain them, so in `app/config.py` add fields
`langsmith_tracing`, `langsmith_api_key`, `langsmith_project` (all optional) and at the
bottom of `app/llm.py` export them with `os.environ.setdefault(...)` when set.
Every chat turn then appears as a tree: router call, tool calls, final answer, with
latency and token counts.

## 8.2 Schema migrations: Alembic
`create_all` won't add a column to an existing table. When you first need to change a
model: `uv add alembic`, `uv run alembic init migrations`, point `env.py` at
`app.db.Base.metadata` and `settings.database_url`, then
`uv run alembic revision --autogenerate -m "..."` and `uv run alembic upgrade head`.

## 8.3 Long conversations
State grows forever. In `build_agent_node`, replace `history = list(state["messages"])`
with `langchain_core.messages.trim_messages(..., max_tokens=..., token_counter=llm, start_on="human", include_system=False)`.
Trim at the model-call boundary, not in the state, so the log stays complete.

## 8.4 New node types
Each one is: a field or two in `NodeSpec`, a builder in `nodes.py`, one line in `build_node`.
- `handoff`: appends a fixed AI message ("A human will take over") and writes a `handoff` row. Routers can route to it.
- `summarizer`: writes a one-line summary of the conversation into `conversations.title`.
- `guard`: a cheap model that checks the incoming message for prompt-injection or abuse and routes to a refusal.

## 8.5 Per-client knowledge upload
Accept `.md`/`.txt` file uploads (`python-multipart`) instead of pasting; later PDFs via `pypdf`.

## 8.6 Evals
`tests/evals/cases.jsonl` with `{"message": ..., "expect_route": ...}` lines and a pytest
that runs only the router node (cheap) and asserts the label. Run it whenever you change
a triage prompt. Routing accuracy is the single most important number in this system.

## 8.7 Ship it
- Dockerfile for the API (`uv sync --frozen`, `uvicorn app.main:app --host 0.0.0.0`), add it to `docker-compose.yml` next to `db`.
- Put the real admin password and OpenAI key in the host's env, never in the image.
- Add `ALLOWED_ORIGINS` and CORS middleware only if the frontend moves to its own domain.
