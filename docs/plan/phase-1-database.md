# Phase 1 — Database

**Goal:** every table the system needs, created by one script, including the
`vector` extension and an HNSW index on embeddings.

**What you learn:** SQLAlchemy 2.0 declarative models, UUID keys, JSONB, the pgvector
column type, why `expire_on_commit=False`, approximate-nearest-neighbour indexes.

## 1.1 Engine and session

`app/db.py`
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Explanation.** `pool_pre_ping` checks a pooled connection is alive before using it,
so a restarted Docker DB doesn't produce one mysterious failure. `expire_on_commit=False`
means objects keep their loaded values after `commit()`; without it, touching
`conv.id` after the session closes (which we do inside a streaming response) raises.

## 1.2 Models

`app/models.py`
```python
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uuid_pk():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def created_at_col():
    return mapped_column(DateTime(timezone=True), default=utcnow)


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = uuid_pk()
    slug: Mapped[str] = mapped_column(String(64), unique=True)   # the "client ID" people type
    name: Mapped[str] = mapped_column(String(200))
    model: Mapped[str] = mapped_column(String(100))              # OpenAI model for this client
    created_at: Mapped[datetime] = created_at_col()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20))                # "admin" | "member"
    client_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("clients.id"), nullable=True)
    created_at: Mapped[datetime] = created_at_col()


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = created_at_col()

    user: Mapped[User] = relationship()


class DagVersion(Base):
    __tablename__ = "dag_versions"
    __table_args__ = (UniqueConstraint("client_id", "version"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    spec: Mapped[dict] = mapped_column(JSONB)                    # the DAG file itself
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = created_at_col()


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = uuid_pk()
    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = created_at_col()

    chunks: Mapped[list["Chunk"]] = relationship(cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dims))


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = uuid_pk()
    client_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = created_at_col()


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))                # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_col()
```

**Explanation.**

- `Client.slug` is the human "client ID" (e.g. `acme`); `Client.id` is the UUID used in foreign keys. `Client.model` is the OpenAI model name you pick in the admin panel.
- `DagVersion` holds the DAG file as JSONB. Every save creates a new row with `version + 1`; exactly one row per client has `is_active=True`. That gives you history and rollback for free.
- `Chunk.client_id` is duplicated from the document on purpose: similarity search filters by client, and a join on every query would be slower and easier to get wrong (tenant isolation).
- `Chunk.embedding` is a `Vector(1536)`. pgvector stores it as a native type and gives you `<=>` (cosine distance) in SQL.
- `Message` is a plain log for humans (admin view, conversation list). LangGraph keeps its own richer state (tool calls etc.) in its checkpoint tables.

## 1.3 Create-tables script

`scripts/__init__.py` — empty file (so `python -m scripts.init_db` resolves imports from the repo root).

`scripts/init_db.py`
```python
from sqlalchemy import text

from app import models  # noqa: F401  (importing registers the tables on Base.metadata)
from app.db import Base, engine

with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

Base.metadata.create_all(engine)

with engine.begin() as conn:
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    ))

print("tables:", ", ".join(sorted(Base.metadata.tables)))
```

**Explanation.** `create_all` is idempotent: it creates missing tables and leaves existing ones alone (it does *not* alter columns; when you change a model later, phase 8 introduces Alembic migrations). The HNSW index makes nearest-neighbour search approximate but fast; with a few thousand chunks you wouldn't notice the difference, but it's the right habit.

## Verify

```bash
uv run python -m scripts.init_db
```

```bash
docker compose exec db psql -U support -d support -c '\dt' -c '\d chunks'
```

Expected: eight tables listed (`auth_tokens, chunks, clients, conversations, dag_versions, documents, messages, users`), and `chunks` shows `embedding | vector(1536)` and the `chunks_embedding_idx` index.

Commit: `git add -A && git commit -m "phase 1: database models"`
