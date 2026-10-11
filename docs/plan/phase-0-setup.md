# Phase 0 — Setup

**Goal:** a uv-managed Python 3.12 project, Postgres + pgvector running in Docker,
and a typed settings object that loads `.env`.

**What you learn:** uv projects, why Python 3.12 (wheels for every dependency exist), docker compose, pydantic-settings.

## 0.1 Python + project

```bash
cd ~/Desktop/Support-Agent
uv python install 3.12
uv init --python 3.12 --name support_agent
rm -f main.py
```

`uv init` created `pyproject.toml` and `.python-version`. Now add dependencies:

```bash
uv add fastapi "uvicorn[standard]" sqlalchemy "psycopg[binary,pool]" pgvector pydantic-settings bcrypt langgraph langchain-core langchain-openai langgraph-checkpoint-postgres
uv add --dev pytest httpx
```

What each one is for:

| Package | Role |
|---------|------|
| fastapi, uvicorn | HTTP API and the server that runs it |
| sqlalchemy, psycopg | ORM and the Postgres driver (psycopg 3). `pool` extra gives the connection pool LangGraph's checkpointer uses |
| pgvector | SQLAlchemy `Vector` column type + distance operators |
| pydantic-settings | Load `.env` into a typed `Settings` object |
| bcrypt | Password hashing |
| langgraph, langchain-core | Graph runtime, message types, tool decorator |
| langchain-openai | `ChatOpenAI`, `OpenAIEmbeddings` |
| langgraph-checkpoint-postgres | Conversation memory stored in Postgres |
| pytest, httpx | Tests (httpx is FastAPI's test client transport) |

## 0.2 Files

`.gitignore`
```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

`.env.example` (commit this; copy it to `.env` and fill the key)
```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg://support:support@127.0.0.1:5433/support
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change-me-now
DEFAULT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

```bash
cp .env.example .env
# then edit .env and paste your real OpenAI key
```

`docker-compose.yml`
```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: support
      POSTGRES_PASSWORD: support
      POSTGRES_DB: support
    ports:
      - "127.0.0.1:5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

Port 5433 on the host avoids colliding with any local Postgres on 5432. The image is
plain Postgres 16 with the `vector` extension compiled in; you still have to run
`CREATE EXTENSION vector` once (phase 1 does it).

`app/__init__.py` — empty file.

`app/config.py`
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    database_url: str
    admin_email: str
    admin_password: str
    default_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536

    @property
    def psycopg_url(self) -> str:
        """Plain libpq URL for tools that don't speak SQLAlchemy (the LangGraph checkpointer)."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


settings = Settings()
```

**Explanation.** `BaseSettings` reads each field from the environment (case-insensitive),
falling back to `.env`. It does *not* export the values into `os.environ`, which is why
later we pass `settings.openai_api_key` to `ChatOpenAI` explicitly instead of hoping
the library finds the env var. `embedding_dims` must match the embedding model:
`text-embedding-3-small` produces 1536 floats, and the vector column is declared with that size.

## Verify

```bash
open -a Docker
```

Wait until the whale icon is steady, then:

```bash
docker compose up -d && docker compose ps
```

```bash
uv run python -c "from app.config import settings; print(settings.default_model, settings.psycopg_url)"
```

Expected: the container shows `running`, and the second command prints
`gpt-4.1-mini postgresql://support:support@127.0.0.1:5433/support`.

Commit: `git add -A && git commit -m "phase 0: project setup"`
