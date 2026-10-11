# Phase 5 — Graph compiler (the heart)

**Goal:** turn a validated `DagSpec` into a running LangGraph graph: three node
builders, two tools, a registry that caches compiled graphs and recompiles when the
active DAG or the client's model changes, Postgres-backed conversation memory, and a
terminal chat to prove it all works.

**What you learn:** LangGraph `StateGraph`, state reducers (`add_messages`),
conditional edges, closures as node factories, the raw tool-calling loop, structured
output with a dynamic `Literal`, checkpointers and `thread_id`, hot reload by cache key.

## 5.1 State

`app/dag/state.py`
```python
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # full conversation, persisted by the checkpointer
    route: str                                             # label chosen by the last router node
    context: list[str]                                     # chunks found by the last retriever node
```

**Explanation.** A node returns a *partial* dict. For most keys the new value
replaces the old one. `messages` is annotated with the `add_messages` reducer, so a
node returning `{"messages": [ai]}` *appends* instead of overwriting. The
checkpointer saves this whole state after every node, keyed by `thread_id`.

## 5.2 Tools (replace the phase 3 stub)

`app/dag/tools.py`
```python
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

from app import knowledge
from app.db import SessionLocal


def make_search_knowledge(client_id: uuid.UUID):
    @tool
    def search_knowledge(query: str) -> str:
        """Search this company's own knowledge base (policies, prices, product details).
        Use it whenever the answer depends on company-specific facts."""
        with SessionLocal() as db:
            hits = knowledge.search(db, client_id, query, k=4)
        if not hits:
            return "No relevant documents found."
        return "\n\n---\n\n".join(h.content for h in hits)

    return search_knowledge


def make_current_time(client_id: uuid.UUID):
    @tool
    def current_time() -> str:
        """Return the current date and time in UTC."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    return current_time


# name used in the DAG file -> factory(client_id) -> LangChain tool
TOOL_REGISTRY = {
    "search_knowledge": make_search_knowledge,
    "current_time": make_current_time,
}
```

**Explanation.** `@tool` turns a function into something the model can call; the
docstring becomes the tool description the model reads, and the type hints become its
JSON schema. Each factory closes over `client_id`, so the tool is already scoped to
the right tenant and the model never sees or chooses a client id. Adding a tool is:
write a factory, add one line to the registry, name it in a DAG file.

## 5.3 Node builders

`app/dag/nodes.py`
```python
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import Field, create_model

from app import knowledge
from app.dag.spec import NodeSpec
from app.dag.state import ChatState
from app.dag.tools import TOOL_REGISTRY
from app.db import SessionLocal
from app.llm import chat_model
from app.models import Client


def message_text(message: BaseMessage) -> str:
    """Content can be a plain string or a list of content blocks; return the text either way."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return message_text(m)
    return ""


def plain_turns(messages: list[BaseMessage], limit: int) -> list[BaseMessage]:
    """Only human messages and final AI answers, no tool traffic. Safe to slice: a ToolMessage
    without its preceding tool call would make the OpenAI API reject the request."""
    kept = [
        m for m in messages
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return kept[-limit:]


def build_retriever_node(node: NodeSpec, client: Client):
    def retrieve(state: ChatState) -> dict:
        query = last_human_text(state["messages"])
        with SessionLocal() as db:
            hits = knowledge.search(db, client.id, query, k=node.top_k)
        return {"context": [h.content for h in hits]}

    return retrieve


def build_router_node(node: NodeSpec, client: Client):
    Decision = create_model(
        "Decision",
        label=(Literal[tuple(node.labels)], Field(description="The single best label")),
    )
    llm = chat_model(node.model or client.model).with_structured_output(Decision)
    instructions = f"{node.prompt}\n\nAnswer with exactly one label from: {', '.join(node.labels)}."

    def route(state: ChatState) -> dict:
        decision = llm.invoke([SystemMessage(instructions), *plain_turns(state["messages"], 6)])
        return {"route": decision.label}

    return route


def build_agent_node(node: NodeSpec, client: Client):
    tools = [TOOL_REGISTRY[name](client.id) for name in node.tools]
    tools_by_name = {t.name: t for t in tools}
    base_llm = chat_model(node.model or client.model)
    llm = base_llm.bind_tools(tools) if tools else base_llm

    def system_text(state: ChatState) -> str:
        text = node.system_prompt
        context = state.get("context") or []
        if context:
            excerpts = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))
            text += f"\n\n# Knowledge base excerpts\n{excerpts}"
        return text

    def agent(state: ChatState) -> dict:
        system = SystemMessage(system_text(state))
        history = list(state["messages"])
        new: list[BaseMessage] = []
        for _ in range(node.max_tool_rounds + 1):
            ai = llm.invoke([system, *history, *new])
            new.append(ai)
            if not ai.tool_calls:
                break
            for call in ai.tool_calls:
                result = tools_by_name[call["name"]].invoke(call["args"])
                new.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        if isinstance(new[-1], ToolMessage):
            # ran out of rounds with tool results pending: force a final answer without tools
            new.append(base_llm.invoke([system, *history, *new]))
        return {"messages": new}

    return agent


def build_node(node: NodeSpec, client: Client):
    builders = {
        "retriever": build_retriever_node,
        "router": build_router_node,
        "agent": build_agent_node,
    }
    return builders[node.type](node, client)
```

**Explanation.**

- Each builder is a closure factory: it does the expensive setup once (model object, tools, the structured-output schema) and returns the function LangGraph will call per turn. That function receives the state and returns a partial update.
- **Router.** `create_model` builds a Pydantic class at runtime whose `label` field is `Literal["billing", "technical", "general"]`. `with_structured_output` makes the model fill that schema, so the answer can only be one of the labels. The conditional edge then maps that label to a node.
- **Agent.** This is the ReAct loop with no magic. Call the model; if it asked for tools, run each, append `ToolMessage`s, call again. `max_tool_rounds` bounds it. The tail check handles the edge case where the limit hits while tool results are pending: the OpenAI API requires every tool call to be answered, and we also want the customer to get a reply, so we make one last call with the tool-less model.
- The retriever's hits go into `state.context`; the agent's system prompt gets them appended as numbered excerpts. The agent can *also* call `search_knowledge` with a better query of its own. Two retrieval styles, both in the DAG file, both scoped by client.

## 5.4 Compiler

`app/dag/compiler.py`
```python
from langgraph.graph import END, START, StateGraph

from app.dag.nodes import build_node
from app.dag.spec import END as SPEC_END
from app.dag.spec import DagSpec
from app.dag.state import ChatState
from app.models import Client


def compile_dag(spec: DagSpec, client: Client, checkpointer):
    graph = StateGraph(ChatState)

    for node in spec.nodes:
        graph.add_node(node.id, build_node(node, client))

    graph.add_edge(START, spec.entry)
    for edge in spec.edges:
        if edge.routes is not None:
            graph.add_conditional_edges(
                edge.source,
                pick_route,
                {label: target(t) for label, t in edge.routes.items()},
            )
        else:
            graph.add_edge(edge.source, target(edge.target))

    return graph.compile(checkpointer=checkpointer)


def target(name: str):
    return END if name == SPEC_END else name


def pick_route(state: ChatState) -> str:
    return state["route"]
```

**Explanation.** This is the whole translation: spec nodes → `add_node`, plain
edges → `add_edge`, routes → `add_conditional_edges(source, function, mapping)`.
The function returns whatever the router wrote into `state["route"]`; the mapping
turns that label into the next node. Validation already guaranteed the mapping is
complete, so the graph can't dead-end.

## 5.5 Registry (cache + hot reload + memory)

`app/dag/registry.py`
```python
import uuid

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.dag.compiler import compile_dag
from app.dag.spec import DagSpec
from app.models import Client, DagVersion


class GraphRegistry:
    def __init__(self):
        self._pool: ConnectionPool | None = None
        self.checkpointer: PostgresSaver | None = None
        self._cache: dict[uuid.UUID, tuple[tuple, object, DagSpec]] = {}

    def start(self) -> None:
        self._pool = ConnectionPool(
            conninfo=settings.psycopg_url,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=True,
        )
        self.checkpointer = PostgresSaver(self._pool)
        self.checkpointer.setup()   # creates LangGraph's checkpoint tables if missing

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.close()

    def get(self, db: Session, client: Client):
        """Return (compiled_graph, spec) for the client's active DAG, compiling only when needed."""
        active = db.scalar(
            select(DagVersion).where(DagVersion.client_id == client.id, DagVersion.is_active.is_(True))
        )
        if active is None:
            raise LookupError(f"client '{client.slug}' has no active DAG")

        key = (active.id, client.model)
        cached = self._cache.get(client.id)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        spec = DagSpec.model_validate(active.spec)
        graph = compile_dag(spec, client, self.checkpointer)
        self._cache[client.id] = (key, graph, spec)
        return graph, spec


registry = GraphRegistry()
```

**Explanation.** The cache key is `(active version id, client model)`. Save a new
DAG version → new id → recompile on the next message. Activate an old version → its
id → recompile. Change the client's model in the admin panel → recompile. Nothing
else to invalidate, no restart. The checkpointer uses its own psycopg pool because
LangGraph talks to Postgres directly, not through SQLAlchemy; `autocommit=True` is
what it requires.

## 5.6 Wire it into the app

Replace `app/main.py`:
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dag.registry import registry
from app.routers import admin, auth, dags, knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.start()
    yield
    registry.stop()


app = FastAPI(title="Support Agent", lifespan=lifespan)
for module in (auth, admin, dags, knowledge):
    app.include_router(module.router)


@app.get("/health")
def health():
    return {"ok": True}
```

## 5.7 Terminal chat

`scripts/chat_cli.py`
```python
import sys
import uuid

from langchain_core.messages import HumanMessage
from sqlalchemy import select

from app.dag.nodes import message_text
from app.dag.registry import registry
from app.db import SessionLocal
from app.models import Client


def main(slug: str) -> None:
    registry.start()
    thread_id = str(uuid.uuid4())
    print(f"chatting with '{slug}'  (thread {thread_id})  Ctrl-C to quit")
    try:
        while True:
            text = input("\nyou> ").strip()
            if not text:
                continue
            with SessionLocal() as db:
                client = db.scalar(select(Client).where(Client.slug == slug))
                if client is None:
                    sys.exit(f"no client with slug '{slug}'")
                graph, _spec = registry.get(db, client)
            config = {"configurable": {"thread_id": thread_id}}
            for update in graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
                for node_name, delta in update.items():
                    route = (delta or {}).get("route")
                    print(f"  [{node_name}]" + (f" -> {route}" if route else ""))
                    for m in (delta or {}).get("messages", []):
                        if m.type == "ai" and not m.tool_calls:
                            print(f"\nbot> {message_text(m)}")
                        elif m.type == "tool":
                            print(f"  (tool result: {message_text(m)[:60]}...)")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    finally:
        registry.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "acme")
```

**Explanation.** `stream_mode="updates"` yields one dict per finished node:
`{node_name: what_it_returned}`. That's why you see `[retrieve]`, `[triage] -> billing`,
`[billing]` in order: you're watching the DAG execute. `registry.get` runs on every
turn, so if you save a new DAG in another terminal the *next* message already uses it.
Memory works because every call uses the same `thread_id`.

## Verify

Make sure phase 3's DAG is active and phase 4's document is loaded, then:

```bash
uv run python -m scripts.chat_cli acme
```

Try, in order:

1. `hi, what plans do you have?` — expect `[triage] -> general` or `billing`, and prices from the document.
2. `I bought the annual plan 20 days ago, can I get a refund?` — expect `-> billing`, a pro-rated answer, probably a `(tool result ...)` line.
3. `and what about 100 days?` — expect it to remember the context from message 2 (memory via checkpointer).
4. `the app crashes when I upload a file` — expect `-> technical`.

Hot reload test: in a second terminal, save a new DAG version where the `general`
node's `system_prompt` ends with "Always end with a joke." (POST the edited JSON as in
phase 3). Send another message in the CLI: the next `general` answer should end with a joke, with no restart.

Confirm memory landed in Postgres:
```bash
docker compose exec db psql -U support -d support -c 'select thread_id, count(*) from checkpoints group by thread_id'
```

Commit: `git add -A && git commit -m "phase 5: graph compiler, registry, terminal chat"`
