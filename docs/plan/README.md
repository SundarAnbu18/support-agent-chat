# Support Agent — build plan

A chat-only, multi-agent customer-support platform. One admin panel onboards people
and clients. Each client gets its own **DAG file** (a JSON pipeline spec) that is
compiled at runtime into a **LangGraph** graph. Postgres with **pgvector** stores
everything: clients, users, DAG versions, knowledge-base chunks (embeddings), and
LangGraph's own conversation memory. One OpenAI key in `.env` serves every client.

You type all application code yourself. These docs give the code and explain it.
I never write files under `app/`, `scripts/`, `static/`, or `examples/`.

## Rules of the game

1. One phase per sitting. Each phase ends with a **Verify** block that proves it works.
2. Commit after every green Verify: `git add -A && git commit -m "phase N: ..."`.
3. If a Verify fails, paste the exact error to me. Do not skip ahead.
4. Type the code; read the explanation *after* typing, then re-read the code.
5. Every phase lists **What you learn** so you can check your own understanding.

## Phases

| # | Phase | You end up with |
|---|-------|-----------------|
| 0 | [Setup](phase-0-setup.md) | uv project, Python 3.12, Postgres+pgvector in Docker, settings |
| 1 | [Database](phase-1-database.md) | SQLAlchemy models, tables created, vector index |
| 2 | [Auth + admin API](phase-2-auth-admin.md) | login, tokens, create clients and users via curl |
| 3 | [DAG spec](phase-3-dag-spec.md) | Pydantic DAG schema with cycle detection, versioned DAG API, tests |
| 4 | [Knowledge base](phase-4-knowledge.md) | documents → chunks → embeddings → similarity search |
| 5 | [Graph compiler](phase-5-graph-compiler.md) | spec → LangGraph, agents with tools, hot reload, terminal chat |
| 6 | [Chat API](phase-6-chat-api.md) | conversations + streaming SSE endpoint |
| 7 | [Frontend](phase-7-frontend.md) | admin panel + chat page, vanilla HTML/JS |
| 8 | [What's next](phase-8-next.md) | tracing, migrations, handoff, evals |

Target: phases 0–5 give you a working multi-agent chat in the terminal. 6–7 put it on the web.

## Architecture

```
  Browser: static/chat.html  and  static/admin.html
                 │  JSON over HTTP, replies streamed as SSE
  ┌──────────────▼──────────────────────────────────────────────┐
  │ FastAPI (app/main.py)                                        │
  │   /auth/*      login, me                                     │
  │   /admin/*     clients, users, dag versions, documents       │
  │   /chat/*      conversations, messages (streaming)           │
  │                                                              │
  │   GraphRegistry (app/dag/registry.py)                        │
  │     reads the client's ACTIVE DagVersion                     │
  │     compile_dag(spec) ──► LangGraph StateGraph (cached)      │
  │        nodes: retriever │ router │ agent(+tools)             │
  │        tools: search_knowledge (pgvector), current_time      │
  └───────┬──────────────────────────────┬───────────────────────┘
          │ SQLAlchemy                   │ LangGraph PostgresSaver
  ┌───────▼──────────────────────────────▼───────────────────────┐
  │ Postgres 16 + pgvector (Docker)                               │
  │  clients · users · auth_tokens · dag_versions(JSONB)          │
  │  documents · chunks(embedding vector(1536))                   │
  │  conversations · messages · checkpoints (LangGraph memory)    │
  └──────────────────────────────────────────────────────────────┘
          ▲
     OpenAI API: chat model per client (clients.model), embeddings
     One key: OPENAI_API_KEY in .env
```

## Life of one chat message

1. `POST /chat/conversations/{id}/messages` arrives with a Bearer token.
2. The token resolves to a user → their client → the client's active `DagVersion`.
3. `GraphRegistry.get()` returns the compiled graph for that version (compiles on first use or after a DAG change).
4. The graph runs with `thread_id = conversation id`. LangGraph's Postgres checkpointer loads the previous messages for that thread automatically.
5. Nodes run in DAG order: `retrieve` (pgvector) → `triage` (router, structured output) → one `agent` (LLM, may call tools) → END.
6. Tokens and node events stream back as SSE. The final answer is also saved to `messages` for the admin's view.

## Decisions (and why)

| Decision | Why |
|----------|-----|
| DAG spec is JSON stored in a JSONB column, versioned | Editable from the admin panel, diffable, rollback by activating an older version. No file system needed. |
| Strict DAG at the top level (no cycles) | You asked for a DAG. Cycles live only *inside* an agent node (its tool loop), which has a hard round limit. |
| Agent tool loop written by hand (~20 lines) | You see the real ReAct loop instead of a black box, and the API won't shift under you. |
| Router uses structured output with a `Literal` of labels | The model can only answer with a valid label, so the conditional edge can't fall off the map. |
| Sync SQLAlchemy + sync FastAPI endpoints | Less to learn, no event-loop foot-guns. FastAPI runs `def` endpoints in a threadpool. |
| LangGraph `PostgresSaver` for memory | Conversation history is handled by the framework, keyed by `thread_id`. Zero custom code. |
| One `OPENAI_API_KEY`, model chosen per client | Exactly what you asked for. Per-node model override is in the spec for later. |
| Vanilla HTML/JS frontend | No node toolchain in the critical path. Replace with React later if you want. |
