# Phase 2 — Auth + admin API

**Goal:** an admin can log in, create clients (name, slug, model) and onboard users
(members bound to a client). All through HTTP, tested with curl.

**What you learn:** password hashing, opaque bearer tokens, FastAPI dependencies
(`Depends`) as the auth mechanism, router-level guards, Pydantic request/response schemas.

## 2.1 Security helpers

`app/security.py`
```python
import secrets

import bcrypt
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AuthToken, User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_token(db: Session, user: User) -> str:
    token = AuthToken(token=secrets.token_urlsafe(32), user_id=user.id)
    db.add(token)
    db.commit()
    return token.token


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = db.get(AuthToken, authorization.removeprefix("Bearer "))
    if token is None:
        raise HTTPException(401, "Invalid token")
    return token.user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    return user
```

**Explanation.** A token is just 32 random bytes stored in a table; logging out is
deleting the row. No JWT, no clock skew, no secret rotation. `current_user` is a
dependency: any endpoint that declares `user: User = Depends(current_user)` gets the
user or a 401 before its own code runs. `require_admin` stacks on top of it.

## 2.2 API schemas

`app/schemas.py`
```python
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FromORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# auth
class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    token: str
    role: str
    client_id: uuid.UUID | None


# users
class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    role: Literal["admin", "member"] = "member"
    client_id: uuid.UUID | None = None


class UserOut(FromORM):
    id: uuid.UUID
    email: str
    role: str
    client_id: uuid.UUID | None


# clients
class ClientCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    name: str
    model: str | None = None


class ClientUpdate(BaseModel):
    name: str | None = None
    model: str | None = None


class ClientOut(FromORM):
    id: uuid.UUID
    slug: str
    name: str
    model: str
    created_at: datetime


# dag versions (used in phase 3)
class DagIn(BaseModel):
    spec: dict[str, Any]
    activate: bool = True


class DagOut(FromORM):
    id: uuid.UUID
    version: int
    is_active: bool
    created_at: datetime
    spec: dict[str, Any]


# knowledge (phase 4)
class DocumentIn(BaseModel):
    title: str
    text: str = Field(min_length=1)


class DocumentOut(FromORM):
    id: uuid.UUID
    title: str
    created_at: datetime


class SearchHit(BaseModel):
    document_id: uuid.UUID
    position: int
    content: str


# chat (phase 6)
class ConversationCreate(BaseModel):
    client_id: uuid.UUID | None = None


class ConversationOut(FromORM):
    id: uuid.UUID
    client_id: uuid.UUID
    title: str
    created_at: datetime


class MessageIn(BaseModel):
    content: str = Field(min_length=1)


class MessageOut(FromORM):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
```

**Explanation.** `from_attributes=True` lets FastAPI build a response model straight
from a SQLAlchemy object. Schemas for later phases are here already so you type this
file once.

## 2.3 Routers

`app/routers/__init__.py` — empty.

`app/routers/deps.py`
```python
import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Client


def get_client_or_404(db: Session, client_id: uuid.UUID) -> Client:
    client = db.get(Client, client_id)
    if client is None:
        raise HTTPException(404, "Client not found")
    return client
```

`app/routers/auth.py`
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.schemas import LoginIn, TokenOut, UserOut
from app.security import create_token, current_user, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Bad email or password")
    return TokenOut(token=create_token(db, user), role=user.role, client_id=user.client_id)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user
```

`app/routers/admin.py`
```python
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Client, User
from app.routers.deps import get_client_or_404
from app.schemas import ClientCreate, ClientOut, ClientUpdate, UserCreate, UserOut
from app.security import hash_password, require_admin

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/clients", response_model=list[ClientOut])
def list_clients(db: Session = Depends(get_db)):
    return db.scalars(select(Client).order_by(Client.created_at)).all()


@router.post("/clients", response_model=ClientOut, status_code=201)
def create_client(body: ClientCreate, db: Session = Depends(get_db)):
    if db.scalar(select(Client).where(Client.slug == body.slug)):
        raise HTTPException(409, "Slug already used")
    client = Client(slug=body.slug, name=body.name, model=body.model or settings.default_model)
    db.add(client)
    db.commit()
    db.refresh(client)
    return client


@router.patch("/clients/{client_id}", response_model=ClientOut)
def update_client(client_id: uuid.UUID, body: ClientUpdate, db: Session = Depends(get_db)):
    client = get_client_or_404(db, client_id)
    if body.name is not None:
        client.name = body.name
    if body.model is not None:
        client.model = body.model
    db.commit()
    return client


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.scalars(select(User).order_by(User.created_at)).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    if body.role == "member" and body.client_id is None:
        raise HTTPException(422, "Members must belong to a client")
    if body.client_id is not None:
        get_client_or_404(db, body.client_id)
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already used")
    user = User(
        email=email,
        password_hash=hash_password(body.password),
        role=body.role,
        client_id=body.client_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
```

**Explanation.** `dependencies=[Depends(require_admin)]` on the router guards every
route in the file. Changing a client's `model` here is all it takes to switch the LLM
for that client: phase 5's registry includes the model in its cache key, so the next
message recompiles the graph with the new model.

## 2.4 App entry point and admin seed

`app/main.py` (phase 5 replaces this with a version that starts the graph registry)
```python
from fastapi import FastAPI

from app.routers import admin, auth

app = FastAPI(title="Support Agent")
app.include_router(auth.router)
app.include_router(admin.router)


@app.get("/health")
def health():
    return {"ok": True}
```

`scripts/seed_admin.py`
```python
from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import User
from app.security import hash_password

with SessionLocal() as db:
    email = settings.admin_email.lower()
    if db.scalar(select(User).where(User.email == email)):
        print("admin already exists:", email)
    else:
        db.add(User(email=email, password_hash=hash_password(settings.admin_password), role="admin"))
        db.commit()
        print("admin created:", email)
```

## Verify

Terminal 1:
```bash
uv run python -m scripts.seed_admin && uv run uvicorn app.main:app --reload --port 8000
```

Terminal 2 (replace the password with the one in your `.env`):
```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'content-type: application/json' -d '{"email":"admin@example.com","password":"change-me-now"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])') && echo "$TOKEN"
```

```bash
curl -s -X POST localhost:8000/admin/clients -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{"slug":"acme","name":"Acme Inc","model":"gpt-4.1-mini"}'
```

Copy the `id` from that output into the next command:
```bash
curl -s -X POST localhost:8000/admin/users -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' -d '{"email":"sam@acme.com","password":"sam-password-1","role":"member","client_id":"PASTE-CLIENT-ID"}'
```

Expected: a client JSON with `slug: acme`, a user JSON with `role: member`, and
`curl localhost:8000/admin/clients` without a token returns 401.
Open http://localhost:8000/docs to see FastAPI's generated API explorer.

Commit: `git add -A && git commit -m "phase 2: auth and admin api"`
