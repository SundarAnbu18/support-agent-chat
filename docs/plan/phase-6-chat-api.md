# Phase 6 — Chat API with streaming

**Goal:** members (and admins) create conversations and send messages over HTTP; the
reply streams back token by token as Server-Sent Events, with node events so the UI
can show which agent is answering.

**What you learn:** SSE over a POST with `StreamingResponse`, LangGraph's
`stream_mode=["updates", "messages"]`, filtering token streams by node, keeping a
conversation log separate from graph state.

## 6.1 Chat router

`app/routers/chat.py`
```python
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dag.nodes import message_text
from app.dag.registry import registry
from app.db import SessionLocal, get_db
from app.models import Client, Conversation, Message, User
from app.schemas import ConversationCreate, ConversationOut, MessageIn, MessageOut
from app.security import current_user

router = APIRouter(prefix="/chat", tags=["chat"])


def conversation_or_404(db: Session, conversation_id: uuid.UUID, user: User) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or (user.role != "admin" and conv.user_id != user.id):
        raise HTTPException(404, "Conversation not found")
    return conv


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    stmt = select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.created_at.desc())
    return db.scalars(stmt).all()


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(
    body: ConversationCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    client_id = body.client_id if (user.role == "admin" and body.client_id) else user.client_id
    if client_id is None:
        raise HTTPException(422, "Admins must pass client_id; members are bound to their client")
    if db.get(Client, client_id) is None:
        raise HTTPException(404, "Client not found")
    conv = Conversation(client_id=client_id, user_id=user.id, title="New conversation")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(
    conversation_id: uuid.UUID,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conversation_or_404(db, conversation_id, user)
    stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    return db.scalars(stmt).all()


@router.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: uuid.UUID,
    body: MessageIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conv = conversation_or_404(db, conversation_id, user)
    client = db.get(Client, conv.client_id)
    try:
        graph, spec = registry.get(db, client)
    except LookupError as err:
        raise HTTPException(409, str(err))
    agent_nodes = {n.id for n in spec.nodes if n.type == "agent"}

    db.add(Message(conversation_id=conv.id, role="user", content=body.content))
    if conv.title == "New conversation":
        conv.title = body.content[:60]
    db.commit()

    config = {"configurable": {"thread_id": str(conv.id)}}

    def events():
        reply: list[str] = []
        try:
            stream = graph.stream(
                {"messages": [HumanMessage(body.content)]},
                config,
                stream_mode=["updates", "messages"],
            )
            for mode, data in stream:
                if mode == "updates":
                    for node_name, delta in data.items():
                        yield sse({"type": "node", "name": node_name, "route": (delta or {}).get("route")})
                else:
                    chunk, meta = data
                    if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") in agent_nodes:
                        text = message_text(chunk)
                        if text:
                            reply.append(text)
                            yield sse({"type": "token", "text": text})
        except Exception as err:  # noqa: BLE001 - surface anything to the UI instead of a dead stream
            yield sse({"type": "error", "message": str(err)})
            return

        with SessionLocal() as session:
            session.add(Message(conversation_id=conv.id, role="assistant", content="".join(reply)))
            session.commit()
        yield sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Explanation.**

- `stream_mode=["updates", "messages"]` with a list makes LangGraph yield `(mode, data)` pairs. `"updates"` gives node completions (used for the "which agent" badges). `"messages"` gives `(message_chunk, metadata)` for every LLM token emitted inside *any* node, including the router's structured output. `metadata["langgraph_node"]` tells you where it came from, so we only forward tokens from agent nodes.
- The generator is a plain sync function; Starlette iterates it in a threadpool, so the LLM calls don't block the event loop. The request-scoped `db` session may already be closed by then, which is why the assistant message is saved with a fresh `SessionLocal()`.
- The `thread_id` is the conversation id. LangGraph loads the previous state for that thread before running, so memory is per conversation and per client with no extra code.
- The `messages` table is the human-readable log: the UI reloads history from it, the admin can read it. LangGraph's checkpoints are the machine memory.

## 6.2 Register the router

In `app/main.py`:
```python
from app.routers import admin, auth, chat, dags, knowledge
...
for module in (auth, admin, dags, knowledge, chat):
    app.include_router(module.router)
```

## Verify

Server running. Log in as the member from phase 2:

```bash
MTOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'content-type: application/json' -d '{"email":"sam@acme.com","password":"sam-password-1"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])') && echo "$MTOKEN"
```

```bash
CONV=$(curl -s -X POST localhost:8000/chat/conversations -H "authorization: Bearer $MTOKEN" -H 'content-type: application/json' -d '{}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])') && echo "$CONV"
```

```bash
curl -N -X POST "localhost:8000/chat/conversations/$CONV/messages" -H "authorization: Bearer $MTOKEN" -H 'content-type: application/json' -d '{"content":"how much is the Team plan?"}'
```

Expected: a `node` event for `retrieve`, a `node` event for `triage` with a `route`,
then many `token` events, then a `node` event for the agent, then `done`. Then:

```bash
curl -s "localhost:8000/chat/conversations/$CONV/messages" -H "authorization: Bearer $MTOKEN"
```

shows the user message and the full assistant reply.

Commit: `git add -A && git commit -m "phase 6: chat api with sse streaming"`
