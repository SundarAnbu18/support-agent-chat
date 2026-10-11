# Phase 4 — Knowledge base (pgvector)

**Goal:** admins paste documents per client; the system chunks, embeds and stores
them; a search function returns the most similar chunks for a query.

**What you learn:** what an embedding is in practice, chunking with overlap, cosine
distance in SQL via pgvector, tenant-scoped retrieval.

## 4.1 OpenAI clients

`app/llm.py`
```python
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings


def chat_model(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(model=model or settings.default_model, api_key=settings.openai_api_key)


@lru_cache(maxsize=1)
def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
```

**Explanation.** We don't set `temperature`: some newer OpenAI models reject any value
other than their default, and the client's model is user-chosen. One embeddings client
is cached because it holds an HTTP connection pool.

## 4.2 Chunk, embed, search

`app/knowledge.py`
```python
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm import embeddings
from app.models import Chunk, Document


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Fixed-size windows with overlap so a sentence cut at a boundary still appears whole in one chunk."""
    text = text.strip()
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return pieces


def add_document(db: Session, client_id: uuid.UUID, title: str, text: str) -> Document:
    pieces = chunk_text(text)
    vectors = embeddings().embed_documents(pieces) if pieces else []
    doc = Document(client_id=client_id, title=title)
    for position, (piece, vector) in enumerate(zip(pieces, vectors)):
        doc.chunks.append(Chunk(client_id=client_id, position=position, content=piece, embedding=vector))
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def search(db: Session, client_id: uuid.UUID, query: str, k: int = 4) -> list[Chunk]:
    query_vector = embeddings().embed_query(query)
    stmt = (
        select(Chunk)
        .where(Chunk.client_id == client_id)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(k)
    )
    return list(db.scalars(stmt))
```

**Explanation.** `embed_documents` sends all chunks in one request and returns one
vector per chunk. `cosine_distance` renders as `embedding <=> $1` in SQL; ordering
by it ascending puts the most similar chunks first. The `where client_id` clause is
the tenant wall: Acme's bot can never see Globex's documents. 800 characters is a
sane default for support docs; make it a per-client setting later if you want.

## 4.3 Knowledge API

`app/routers/knowledge.py`
```python
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import knowledge
from app.db import get_db
from app.models import Document
from app.routers.deps import get_client_or_404
from app.schemas import DocumentIn, DocumentOut, SearchHit
from app.security import require_admin

router = APIRouter(prefix="/admin", tags=["knowledge"], dependencies=[Depends(require_admin)])


@router.get("/clients/{client_id}/documents", response_model=list[DocumentOut])
def list_documents(client_id: uuid.UUID, db: Session = Depends(get_db)):
    get_client_or_404(db, client_id)
    stmt = select(Document).where(Document.client_id == client_id).order_by(Document.created_at.desc())
    return db.scalars(stmt).all()


@router.post("/clients/{client_id}/documents", response_model=DocumentOut, status_code=201)
def add_document(client_id: uuid.UUID, body: DocumentIn, db: Session = Depends(get_db)):
    client = get_client_or_404(db, client_id)
    return knowledge.add_document(db, client.id, body.title, body.text)


@router.delete("/clients/{client_id}/documents/{document_id}", status_code=204)
def delete_document(client_id: uuid.UUID, document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None or doc.client_id != client_id:
        raise HTTPException(404, "Document not found")
    db.delete(doc)
    db.commit()


@router.get("/clients/{client_id}/search", response_model=list[SearchHit])
def search_documents(
    client_id: uuid.UUID,
    q: str = Query(min_length=1),
    k: int = Query(default=4, ge=1, le=20),
    db: Session = Depends(get_db),
):
    get_client_or_404(db, client_id)
    return knowledge.search(db, client_id, q, k)
```

Register in `app/main.py`:
```python
from app.routers import admin, auth, dags, knowledge
...
app.include_router(knowledge.router)
```

`examples/acme_refunds.md` — a document to test with:
```
Acme refund policy

Monthly plans can be cancelled at any time; the current month is not refunded.
Annual plans are refundable in full within 14 days of purchase, and pro-rated after that
until day 90. After day 90 annual plans are not refundable.
Refunds are issued to the original payment method within 5 business days.
To request a refund, email billing@acme.example with your invoice number.

Acme plans and prices

Starter: 19 USD per month, 1 user, 5 GB storage.
Team: 49 USD per month, up to 10 users, 100 GB storage, priority support.
Business: 199 USD per month, unlimited users, 1 TB storage, SSO, dedicated manager.
```

## Verify

Server running, `$TOKEN` set, client id pasted:

```bash
curl -s -X POST localhost:8000/admin/clients/PASTE-CLIENT-ID/documents -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d "$(python3 -c 'import json; print(json.dumps({"title": "Refund policy", "text": open("examples/acme_refunds.md").read()}))')"
```

```bash
curl -s "localhost:8000/admin/clients/PASTE-CLIENT-ID/search?q=can%20I%20get%20my%20money%20back%20on%20a%20yearly%20plan&k=2" -H "authorization: Bearer $TOKEN"
```

Expected: a document JSON, then two hits whose `content` contains the refund paragraph. Check the rows exist:

```bash
docker compose exec db psql -U support -d support -c 'select position, left(content, 40) from chunks'
```

Commit: `git add -A && git commit -m "phase 4: knowledge base with pgvector"`
