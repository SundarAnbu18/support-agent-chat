# Phase 3 — The DAG spec

**Goal:** a strict, validated schema for the "DAG file", a versioned API to save and
activate DAGs per client, and unit tests for the validator.

**What you learn:** Pydantic model validators, graph theory you can type in 20 lines
(Kahn's algorithm for cycle detection), why validation at save time beats failing at
chat time, versioning as append-only rows.

## 3.1 What a DAG file looks like

`examples/support_dag.json`
```json
{
  "name": "acme-support-v1",
  "entry": "retrieve",
  "nodes": [
    {"id": "retrieve", "type": "retriever", "top_k": 4},
    {
      "id": "triage",
      "type": "router",
      "prompt": "You are the triage desk of a software company's support team. Classify the customer's latest message.",
      "labels": ["billing", "technical", "general"]
    },
    {
      "id": "billing",
      "type": "agent",
      "system_prompt": "You are Acme's billing specialist. Answer only billing questions (invoices, refunds, plans). Use the knowledge base excerpts and the search_knowledge tool before answering. If you cannot answer, say exactly what you will escalate.",
      "tools": ["search_knowledge"]
    },
    {
      "id": "technical",
      "type": "agent",
      "system_prompt": "You are Acme's technical support engineer. Diagnose step by step, ask one clarifying question at a time, and cite the knowledge base when you use it.",
      "tools": ["search_knowledge", "current_time"]
    },
    {
      "id": "general",
      "type": "agent",
      "system_prompt": "You are Acme's friendly front desk. Answer general questions briefly and point the customer to billing or technical support when needed."
    }
  ],
  "edges": [
    {"source": "retrieve", "target": "triage"},
    {"source": "triage", "routes": {"billing": "billing", "technical": "technical", "general": "general"}},
    {"source": "billing", "target": "END"},
    {"source": "technical", "target": "END"},
    {"source": "general", "target": "END"}
  ]
}
```

Three node types:

| type | what it does | fields |
|------|--------------|--------|
| `retriever` | pgvector search on the latest user message, puts hits in `state.context` | `top_k` |
| `router` | asks the LLM for exactly one label, puts it in `state.route` | `prompt`, `labels` |
| `agent` | an LLM persona with optional tools; appends its answer to `state.messages` | `system_prompt`, `tools`, `max_tool_rounds` |

Every node may set `model` to override the client's model. Edges are either plain
(`target`) or conditional (`routes`: label → node). `"END"` is the exit.

## 3.2 Schema + validator

`app/dag/__init__.py` — empty.

`app/dag/spec.py`
```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator

NodeType = Literal["agent", "router", "retriever"]
END = "END"


class NodeSpec(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: NodeType
    model: str | None = None            # override the client's model for this node only
    # agent
    system_prompt: str | None = None
    tools: list[str] = []
    max_tool_rounds: int = Field(default=3, ge=0, le=10)
    # router
    prompt: str | None = None
    labels: list[str] = []
    # retriever
    top_k: int = Field(default=4, ge=1, le=20)

    @model_validator(mode="after")
    def check_type_fields(self):
        if self.type == "agent" and not self.system_prompt:
            raise ValueError(f"agent node '{self.id}' needs a system_prompt")
        if self.type == "router":
            if not self.prompt:
                raise ValueError(f"router node '{self.id}' needs a prompt")
            if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
                raise ValueError(f"router node '{self.id}' needs at least two distinct labels")
        return self


class EdgeSpec(BaseModel):
    source: str
    target: str | None = None               # plain edge
    routes: dict[str, str] | None = None    # conditional edge: label -> node id or "END"

    def targets(self) -> list[str]:
        return [self.target] if self.target is not None else list(self.routes.values())


class DagSpec(BaseModel):
    name: str
    entry: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]

    @model_validator(mode="after")
    def check_graph(self):
        from app.dag.tools import TOOL_REGISTRY  # local import: tools.py imports the DB layer

        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids")
        by_id = {n.id: n for n in self.nodes}
        if self.entry not in by_id:
            raise ValueError(f"entry '{self.entry}' is not a node")

        for n in self.nodes:
            for t in n.tools:
                if t not in TOOL_REGISTRY:
                    raise ValueError(f"node '{n.id}' uses unknown tool '{t}'")

        outgoing: dict[str, list[EdgeSpec]] = {i: [] for i in ids}
        for e in self.edges:
            if e.source not in by_id:
                raise ValueError(f"edge source '{e.source}' is not a node")
            if (e.target is None) == (e.routes is None):
                raise ValueError(f"edge from '{e.source}' needs exactly one of target / routes")
            for t in e.targets():
                if t != END and t not in by_id:
                    raise ValueError(f"edge from '{e.source}' points to unknown node '{t}'")
            outgoing[e.source].append(e)

        for n in self.nodes:
            outs = outgoing[n.id]
            if len(outs) != 1:
                raise ValueError(f"node '{n.id}' must have exactly one outgoing edge, has {len(outs)}")
            e = outs[0]
            if n.type == "router":
                if e.routes is None:
                    raise ValueError(f"router '{n.id}' needs a routes edge")
                if set(e.routes) != set(n.labels):
                    raise ValueError(f"routes of '{n.id}' must cover exactly its labels {n.labels}")
            elif e.routes is not None:
                raise ValueError(f"only router nodes may have routes ('{n.id}')")

        assert_acyclic(outgoing)
        return self

    def node(self, node_id: str) -> NodeSpec:
        return next(n for n in self.nodes if n.id == node_id)


def assert_acyclic(outgoing: dict[str, list[EdgeSpec]]) -> None:
    """Kahn's algorithm: repeatedly remove nodes with no incoming edges.
    If some nodes can never be removed, they sit on a cycle."""
    successors = {i: set() for i in outgoing}
    indegree = {i: 0 for i in outgoing}
    for src, edges in outgoing.items():
        for e in edges:
            for t in e.targets():
                if t != END and t not in successors[src]:
                    successors[src].add(t)
                    indegree[t] += 1
    ready = [i for i, d in indegree.items() if d == 0]
    removed = 0
    while ready:
        n = ready.pop()
        removed += 1
        for t in successors[n]:
            indegree[t] -= 1
            if indegree[t] == 0:
                ready.append(t)
    if removed != len(outgoing):
        stuck = sorted(i for i, d in indegree.items() if d > 0)
        raise ValueError(f"DAG contains a cycle through {stuck}")
```

**Explanation.**

- `model_validator(mode="after")` runs after field parsing, so you can reason about the whole object. Raising `ValueError` turns into a normal Pydantic validation error that FastAPI reports as 422.
- "Exactly one outgoing edge per node" is deliberately strict. It keeps the graph readable and means every node's next step is unambiguous; branching only happens via a router's `routes`.
- The cycle check is the heart of "DAG". Kahn's algorithm is also how you'd compute a topological order, which is handy if you later want to draw the graph.
- Tools are validated against a registry (phase 5 defines it) so a typo in the DAG file fails at save time, not when a customer is waiting.

Until phase 5 exists, create a stub so the import works:

`app/dag/tools.py` (temporary; phase 5 replaces it)
```python
TOOL_REGISTRY: dict = {"search_knowledge": None, "current_time": None}
```

## 3.3 Versioned DAG API

`app/routers/dags.py`
```python
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.dag.spec import DagSpec
from app.db import get_db
from app.models import DagVersion
from app.routers.deps import get_client_or_404
from app.schemas import DagIn, DagOut
from app.security import require_admin

router = APIRouter(prefix="/admin", tags=["dags"], dependencies=[Depends(require_admin)])


def validation_messages(err: ValidationError) -> list[str]:
    return [e["msg"].removeprefix("Value error, ") for e in err.errors()]


@router.post("/dags/validate")
def validate_dag(spec: dict[str, Any]):
    try:
        DagSpec.model_validate(spec)
    except ValidationError as err:
        return {"ok": False, "errors": validation_messages(err)}
    return {"ok": True, "errors": []}


@router.get("/clients/{client_id}/dags", response_model=list[DagOut])
def list_dags(client_id: uuid.UUID, db: Session = Depends(get_db)):
    get_client_or_404(db, client_id)
    stmt = select(DagVersion).where(DagVersion.client_id == client_id).order_by(DagVersion.version.desc())
    return db.scalars(stmt).all()


@router.post("/clients/{client_id}/dags", response_model=DagOut, status_code=201)
def create_dag(client_id: uuid.UUID, body: DagIn, db: Session = Depends(get_db)):
    client = get_client_or_404(db, client_id)
    try:
        DagSpec.model_validate(body.spec)
    except ValidationError as err:
        raise HTTPException(422, detail=validation_messages(err))
    latest = db.scalar(select(func.max(DagVersion.version)).where(DagVersion.client_id == client.id)) or 0
    dag = DagVersion(client_id=client.id, version=latest + 1, spec=body.spec, is_active=False)
    db.add(dag)
    if body.activate:
        activate(db, dag)
    db.commit()
    db.refresh(dag)
    return dag


@router.post("/clients/{client_id}/dags/{version}/activate", response_model=DagOut)
def activate_dag(client_id: uuid.UUID, version: int, db: Session = Depends(get_db)):
    get_client_or_404(db, client_id)
    dag = db.scalar(select(DagVersion).where(DagVersion.client_id == client_id, DagVersion.version == version))
    if dag is None:
        raise HTTPException(404, "Version not found")
    activate(db, dag)
    db.commit()
    return dag


def activate(db: Session, dag: DagVersion) -> None:
    db.execute(update(DagVersion).where(DagVersion.client_id == dag.client_id).values(is_active=False))
    dag.is_active = True
```

Register it in `app/main.py`:
```python
from app.routers import admin, auth, dags
...
app.include_router(dags.router)
```

**Explanation.** Saving never overwrites: it inserts `version + 1` and flips the
active flag. "Update the DAG" in the admin panel is therefore "save a new version";
rolling back is "activate an old one". The registry (phase 5) keys its cache on the
active version's id, so activation is the hot-reload trigger.

## 3.4 Tests

`tests/__init__.py` — empty.

`tests/test_spec.py`
```python
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.dag.spec import DagSpec

EXAMPLE = json.loads((Path(__file__).parent.parent / "examples" / "support_dag.json").read_text())


def test_example_is_valid():
    spec = DagSpec.model_validate(EXAMPLE)
    assert spec.entry == "retrieve"
    assert spec.node("triage").labels == ["billing", "technical", "general"]


def test_cycle_is_rejected():
    bad = copy.deepcopy(EXAMPLE)
    bad["edges"][2] = {"source": "billing", "target": "retrieve"}  # billing -> retrieve -> triage -> billing
    with pytest.raises(ValidationError, match="cycle"):
        DagSpec.model_validate(bad)


def test_router_must_cover_all_labels():
    bad = copy.deepcopy(EXAMPLE)
    del bad["edges"][1]["routes"]["general"]
    with pytest.raises(ValidationError, match="cover exactly"):
        DagSpec.model_validate(bad)


def test_node_needs_exactly_one_outgoing_edge():
    bad = copy.deepcopy(EXAMPLE)
    bad["edges"].append({"source": "general", "target": "billing"})
    with pytest.raises(ValidationError, match="exactly one outgoing edge"):
        DagSpec.model_validate(bad)


def test_unknown_tool_is_rejected():
    bad = copy.deepcopy(EXAMPLE)
    bad["nodes"][2]["tools"] = ["launch_rockets"]
    with pytest.raises(ValidationError, match="unknown tool"):
        DagSpec.model_validate(bad)
```

## Verify

```bash
uv run pytest -q
```

Then with the server running (`uv run uvicorn app.main:app --reload --port 8000`) and
`$TOKEN` from phase 2 (log in again if the shell is new), replace the client id:

```bash
curl -s -X POST localhost:8000/admin/clients/PASTE-CLIENT-ID/dags -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d "{\"spec\": $(cat examples/support_dag.json), \"activate\": true}" | head -c 300
```

```bash
curl -s -X POST localhost:8000/admin/dags/validate -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{"name":"x","entry":"a","nodes":[{"id":"a","type":"retriever"}],"edges":[{"source":"a","target":"a"}]}'
```

Expected: 5 tests pass; the first curl returns a version 1 with `is_active: true`;
the second returns `{"ok": false, "errors": ["DAG contains a cycle through ['a']"]}`.

Commit: `git add -A && git commit -m "phase 3: dag spec, validation, versions"`
