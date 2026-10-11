# Phase 7 — Frontend: admin panel + chat

**Goal:** two static pages served by FastAPI. `/admin` onboards people, creates
clients, edits DAG files and knowledge. `/` is the chat. No build step.

**What you learn:** consuming SSE from a POST with `fetch` + `ReadableStream`,
keeping a token in `localStorage`, how little JS a working UI needs.

## 7.1 Serve the pages

In `app/main.py` add:
```python
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).resolve().parent.parent / "static"
```
and after the `include_router` loop:
```python
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(STATIC / "chat.html")


@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(STATIC / "admin.html")
```

`GET /admin` (the page) does not clash with `/admin/clients` (the API): FastAPI matches full paths.

## 7.2 Shared helper

`static/common.js`
```javascript
const TOKEN_KEY = "support_token";

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function setToken(t) { if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY); }

async function api(path, { method = "GET", body } = {}) {
  const headers = { "content-type": "application/json" };
  const token = getToken();
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (res.status === 401) { setToken(null); location.reload(); return; }
  if (res.status === 204) return null;
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

async function login(email, password) {
  const data = await api("/auth/login", { method: "POST", body: { email, password } });
  setToken(data.token);
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "onclick") node.onclick = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

const css = `
  body { font: 14px/1.4 system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1d2330; }
  header { background: #1d2330; color: #fff; padding: 10px 16px; display: flex; gap: 12px; align-items: center; }
  header a { color: #cfd6e4; }
  button { padding: 6px 10px; border: 1px solid #c3cad6; background: #fff; border-radius: 6px; cursor: pointer; }
  button.primary { background: #2f6df6; color: #fff; border-color: #2f6df6; }
  input, textarea, select { padding: 6px 8px; border: 1px solid #c3cad6; border-radius: 6px; font: inherit; width: 100%; box-sizing: border-box; }
  textarea { min-height: 220px; font-family: ui-monospace, monospace; }
  .card { background: #fff; border: 1px solid #e1e5ec; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .row { display: flex; gap: 8px; margin-bottom: 8px; }
  .row > * { flex: 1; }
  .list button { display: block; width: 100%; text-align: left; margin-bottom: 4px; }
  .list button.active { border-color: #2f6df6; background: #e9f0ff; }
  .error { color: #b00020; white-space: pre-wrap; }
  .badge { display: inline-block; background: #e9f0ff; color: #2f6df6; border-radius: 10px; padding: 1px 8px; font-size: 12px; margin-right: 4px; }
`;
document.head.append(el("style", {}, css));
```

## 7.3 Admin panel

`static/admin.html`
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Support Agent — Admin</title>
<script src="/static/common.js"></script>
<style>
  main { display: grid; grid-template-columns: 260px 1fr; gap: 16px; padding: 16px; }
  nav button { margin-right: 4px; }
</style>
</head>
<body>
<header><strong>Support Agent · Admin</strong><a href="/">chat</a><span style="flex:1"></span><button id="logout">Log out</button></header>
<div id="app"></div>
<script>
const app = document.getElementById("app");
let clients = [], users = [], selected = null, tab = "dag";

async function boot() {
  if (!getToken()) return renderLogin();
  const me = await api("/auth/me");
  if (me.role !== "admin") { app.replaceChildren(el("p", { class: "error" }, "Admins only. Log out and use an admin account.")); return; }
  await reload();
}

async function reload() {
  [clients, users] = await Promise.all([api("/admin/clients"), api("/admin/users")]);
  if (selected) selected = clients.find(c => c.id === selected.id) || null;
  render();
}

function renderLogin() {
  const email = el("input", { placeholder: "email" }), pw = el("input", { placeholder: "password", type: "password" });
  const err = el("div", { class: "error" });
  app.replaceChildren(el("div", { class: "card", style: "max-width:360px;margin:60px auto" },
    el("h3", {}, "Admin login"), el("div", { class: "row" }, email), el("div", { class: "row" }, pw), err,
    el("button", { class: "primary", onclick: async () => {
      try { await login(email.value, pw.value); boot(); } catch (e) { err.textContent = e.message; }
    } }, "Log in")));
}

function render() {
  app.replaceChildren(el("main", {}, renderSidebar(), selected ? renderClient() : el("p", {}, "Select or create a client.")));
}

function renderSidebar() {
  const slug = el("input", { placeholder: "slug (e.g. acme)" }), name = el("input", { placeholder: "Client name" }), model = el("input", { placeholder: "model (blank = default)" });
  const err = el("div", { class: "error" });
  return el("div", {},
    el("div", { class: "card list" }, el("h3", {}, "Clients"),
      ...clients.map(c => el("button", { class: selected?.id === c.id ? "active" : "", onclick: () => { selected = c; render(); } }, `${c.name} (${c.slug})`))),
    el("div", { class: "card" }, el("h4", {}, "New client"),
      el("div", { class: "row" }, slug), el("div", { class: "row" }, name), el("div", { class: "row" }, model), err,
      el("button", { class: "primary", onclick: async () => {
        try { selected = await api("/admin/clients", { method: "POST", body: { slug: slug.value, name: name.value, model: model.value || null } }); await reload(); }
        catch (e) { err.textContent = e.message; }
      } }, "Create")));
}

function renderClient() {
  const tabs = { settings: renderSettings, users: renderUsers, dag: renderDag, knowledge: renderKnowledge };
  const nav = el("nav", { class: "card" }, ...Object.keys(tabs).map(t => el("button", { class: t === tab ? "primary" : "", onclick: () => { tab = t; render(); } }, t)));
  const body = el("div", { class: "card" });
  tabs[tab](body);
  return el("div", {}, el("h2", {}, `${selected.name} `, el("small", {}, `slug: ${selected.slug} · model: ${selected.model}`)), nav, body);
}

function renderSettings(box) {
  const name = el("input", { value: selected.name }), model = el("input", { value: selected.model }), err = el("div", { class: "error" });
  box.append(el("h3", {}, "Settings"), el("div", { class: "row" }, el("label", {}, "Name", name), el("label", {}, "OpenAI model", model)), err,
    el("button", { class: "primary", onclick: async () => {
      try { await api(`/admin/clients/${selected.id}`, { method: "PATCH", body: { name: name.value, model: model.value } }); await reload(); }
      catch (e) { err.textContent = e.message; }
    } }, "Save"));
}

function renderUsers(box) {
  const mine = users.filter(u => u.client_id === selected.id);
  const email = el("input", { placeholder: "email" }), pw = el("input", { placeholder: "password (8+ chars)", type: "password" }), err = el("div", { class: "error" });
  box.append(el("h3", {}, "People"),
    el("ul", {}, ...mine.map(u => el("li", {}, `${u.email} · ${u.role}`))),
    el("h4", {}, "Onboard a member"), el("div", { class: "row" }, email, pw), err,
    el("button", { class: "primary", onclick: async () => {
      try { await api("/admin/users", { method: "POST", body: { email: email.value, password: pw.value, role: "member", client_id: selected.id } }); await reload(); }
      catch (e) { err.textContent = e.message; }
    } }, "Create member"));
}

async function renderDag(box) {
  const versions = await api(`/admin/clients/${selected.id}/dags`);
  const active = versions.find(v => v.is_active);
  const editor = el("textarea", {}, active ? JSON.stringify(active.spec, null, 2) : "");
  const out = el("div", { class: "error" });
  const parse = () => { try { return JSON.parse(editor.value); } catch (e) { out.textContent = "Not valid JSON: " + e.message; return null; } };
  box.append(el("h3", {}, "DAG file ", active ? el("span", { class: "badge" }, `active: v${active.version}`) : el("span", { class: "badge" }, "no active DAG")),
    editor, out,
    el("div", { class: "row", style: "margin-top:8px" },
      el("button", { onclick: async () => {
        const spec = parse(); if (!spec) return;
        const r = await api("/admin/dags/validate", { method: "POST", body: spec });
        out.textContent = r.ok ? "Valid ✓" : r.errors.join("\n");
      } }, "Validate"),
      el("button", { class: "primary", onclick: async () => {
        const spec = parse(); if (!spec) return;
        try { await api(`/admin/clients/${selected.id}/dags`, { method: "POST", body: { spec, activate: true } }); render(); }
        catch (e) { out.textContent = e.message; }
      } }, "Save as new version & activate")),
    el("h4", {}, "Versions"),
    el("ul", {}, ...versions.map(v => el("li", {}, `v${v.version} · ${v.spec.name} · ${new Date(v.created_at).toLocaleString()} `,
      v.is_active ? el("span", { class: "badge" }, "active") : el("button", { onclick: async () => { await api(`/admin/clients/${selected.id}/dags/${v.version}/activate`, { method: "POST" }); render(); } }, "activate")))));
}

async function renderKnowledge(box) {
  const docs = await api(`/admin/clients/${selected.id}/documents`);
  const title = el("input", { placeholder: "Title" }), text = el("textarea", { placeholder: "Paste the document text" }), err = el("div", { class: "error" });
  const q = el("input", { placeholder: "Test a search query" }), hits = el("div", {});
  box.append(el("h3", {}, "Knowledge base"),
    el("ul", {}, ...docs.map(d => el("li", {}, `${d.title} `, el("button", { onclick: async () => { await api(`/admin/clients/${selected.id}/documents/${d.id}`, { method: "DELETE" }); render(); } }, "delete")))),
    el("h4", {}, "Add document"), el("div", { class: "row" }, title), text, err,
    el("button", { class: "primary", onclick: async () => {
      try { await api(`/admin/clients/${selected.id}/documents`, { method: "POST", body: { title: title.value, text: text.value } }); render(); }
      catch (e) { err.textContent = e.message; }
    } }, "Embed & save"),
    el("h4", {}, "Test search"), el("div", { class: "row" }, q, el("button", { onclick: async () => {
      const r = await api(`/admin/clients/${selected.id}/search?q=${encodeURIComponent(q.value)}&k=3`);
      hits.replaceChildren(...r.map(h => el("pre", { class: "card" }, h.content)));
    } }, "Search")), hits);
}

document.getElementById("logout").onclick = () => { setToken(null); location.reload(); };
boot();
</script>
</body>
</html>
```

## 7.4 Chat page

`static/chat.html`
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Support Agent — Chat</title>
<script src="/static/common.js"></script>
<style>
  main { display: grid; grid-template-columns: 240px 1fr; gap: 16px; padding: 16px; height: calc(100vh - 44px); box-sizing: border-box; }
  #thread { display: flex; flex-direction: column; }
  #messages { flex: 1; overflow-y: auto; }
  .msg { max-width: 72%; padding: 8px 12px; border-radius: 12px; margin: 6px 0; white-space: pre-wrap; }
  .user { background: #2f6df6; color: #fff; margin-left: auto; }
  .assistant { background: #fff; border: 1px solid #e1e5ec; }
  .trace { font-size: 12px; color: #667; margin: 2px 0 6px; }
</style>
</head>
<body>
<header><strong>Support Agent</strong><a href="/admin">admin</a><span style="flex:1"></span><button id="logout">Log out</button></header>
<div id="app"></div>
<script>
const app = document.getElementById("app");
let me = null, conversations = [], current = null, clientsForAdmin = [];

async function boot() {
  if (!getToken()) return renderLogin();
  me = await api("/auth/me");
  if (me.role === "admin") clientsForAdmin = await api("/admin/clients");
  conversations = await api("/chat/conversations");
  render();
}

function renderLogin() {
  const email = el("input", { placeholder: "email" }), pw = el("input", { placeholder: "password", type: "password" }), err = el("div", { class: "error" });
  app.replaceChildren(el("div", { class: "card", style: "max-width:360px;margin:60px auto" },
    el("h3", {}, "Sign in"), el("div", { class: "row" }, email), el("div", { class: "row" }, pw), err,
    el("button", { class: "primary", onclick: async () => { try { await login(email.value, pw.value); boot(); } catch (e) { err.textContent = e.message; } } }, "Sign in")));
}

function render() {
  const clientPick = me.role === "admin"
    ? el("select", {}, ...clientsForAdmin.map(c => el("option", { value: c.id }, `${c.name}`)))
    : null;
  const side = el("div", { class: "card list" },
    el("h3", {}, "Conversations"),
    ...(clientPick ? [el("div", { class: "row" }, clientPick)] : []),
    el("button", { class: "primary", onclick: async () => {
      current = await api("/chat/conversations", { method: "POST", body: { client_id: clientPick ? clientPick.value : null } });
      conversations = await api("/chat/conversations"); render(); } }, "+ New conversation"),
    ...conversations.map(c => el("button", { class: current?.id === c.id ? "active" : "", onclick: () => { current = c; render(); } }, c.title)));
  const thread = el("div", { id: "thread", class: "card" });
  app.replaceChildren(el("main", {}, side, thread));
  if (current) renderThread(thread);
  else thread.append(el("p", {}, "Start a new conversation."));
}

async function renderThread(thread) {
  const messages = el("div", { id: "messages" });
  const input = el("input", { placeholder: "Type your message and press Enter" });
  const history = await api(`/chat/conversations/${current.id}/messages`);
  for (const m of history) messages.append(bubble(m.role, m.content));
  thread.append(messages, el("div", { class: "row", style: "margin-top:8px" }, input));
  messages.scrollTop = messages.scrollHeight;
  input.focus();
  input.onkeydown = async (ev) => {
    if (ev.key !== "Enter" || !input.value.trim()) return;
    const text = input.value.trim(); input.value = ""; input.disabled = true;
    messages.append(bubble("user", text));
    const trace = el("div", { class: "trace" });
    const reply = bubble("assistant", "");
    messages.append(trace, reply);
    await streamReply(text, trace, reply, messages);
    input.disabled = false; input.focus();
    if (current.title === "New conversation") { conversations = await api("/chat/conversations"); current = conversations.find(c => c.id === current.id); render(); }
  };
}

function bubble(role, text) { return el("div", { class: `msg ${role}` }, text); }

async function streamReply(text, trace, reply, messages) {
  const res = await fetch(`/chat/conversations/${current.id}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${getToken()}` },
    body: JSON.stringify({ content: text }),
  });
  if (!res.ok) { reply.textContent = `Error ${res.status}: ${await res.text()}`; return; }
  const reader = res.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "node") trace.append(el("span", { class: "badge" }, ev.route ? `${ev.name} → ${ev.route}` : ev.name));
        else if (ev.type === "token") reply.textContent += ev.text;
        else if (ev.type === "error") reply.textContent += `\n[error] ${ev.message}`;
        messages.scrollTop = messages.scrollHeight;
      }
    }
  }
}

document.getElementById("logout").onclick = () => { setToken(null); location.reload(); };
boot();
</script>
</body>
</html>
```

**Explanation.**

- The SSE stream comes from a POST, so the browser's `EventSource` (GET only) can't be used. Instead we read the response body as a stream and split it on the blank line that ends each SSE frame. The decoder is called with `{stream: true}` so a multi-byte character split across two network chunks is not corrupted.
- The trace line above each answer shows the DAG executing: `retrieve`, `triage → billing`, `billing`. That is your debugging view for routing problems.
- Admins pick a client from a dropdown and can test any client's bot; members are bound to their client by the API, not by the UI.

## Verify

```bash
uv run uvicorn app.main:app --reload --port 8000
```

1. Open http://localhost:8000/admin, log in as admin. Create a second client (e.g. `globex`), paste the example DAG with different system prompts, add a document, onboard a member.
2. Open http://localhost:8000/ in a private window, log in as that member, start a conversation, ask something from the document. Watch the trace badges and the streamed answer.
3. Back in the admin panel, edit the DAG (change a system prompt), save. Send another message in the chat: the behaviour changes without a restart.
4. Change the client's model in Settings, send another message: still works, now on the new model.

Commit: `git add -A && git commit -m "phase 7: admin panel and chat ui"`

You now have the thing you set out to build. Phase 8 is the backlog.
