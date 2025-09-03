#!/usr/bin/env bash
set -euo pipefail

# ========= Settings =========
PROJECT_DIR="${HOME}/rag-demo"
VENV_DIR="${PROJECT_DIR}/.venv"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
INDEX_FILE="${PROJECT_DIR}/index.py"
CHAT_FILE="${PROJECT_DIR}/chat_cli.py"
ENV_EXAMPLE="${PROJECT_DIR}/.env.example"
GITIGNORE_FILE="${PROJECT_DIR}/.gitignore"

# Defaults (user can later change in .env)
EMBED_MODEL_DEFAULT="nomic-embed-text"
CHAT_MODEL_DEFAULT="llama3.1:8b"
OLLAMA_ENDPOINT_DEFAULT="http://localhost:11434"

# ========= Helpers =========
log()  { echo -e "\033[1;32m[+]\033[0m $*"; }
warn() { echo -e "\033[1;33m[!]\033[0m $*"; }
err()  { echo -e "\033[1;31m[✗]\033[0m $*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

# ========= Preflight =========
log "Starting RAG demo installer (Ubuntu/Linux)."
if [[ "$(uname -s)" != "Linux" ]]; then
  err "This script targets Ubuntu/Linux."; exit 1
fi

# ========= Apt basics =========
log "Updating apt & installing base packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip curl unzip ca-certificates jq

# Git (optional)
if ! need_cmd git; then
  log "Installing git..."
  sudo apt-get install -y git
fi

# ========= AWS CLI v2 =========
if need_cmd aws && aws --version 2>/dev/null | grep -q "aws-cli/2"; then
  log "AWS CLI v2 already installed."
else
  log "Installing AWS CLI v2..."
  sudo rm -rf /tmp/aws /tmp/awscliv2.zip || true
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install --update
fi
aws --version >/dev/null || { err "AWS CLI v2 installation failed."; exit 1; }

# ========= Ollama install / start =========
if ! need_cmd ollama; then
  log "Ollama not found. Installing via official script..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Try to enable & start service (non-fatal if not systemd env)
if command -v systemctl >/dev/null 2>&1; then
  log "Enabling and starting Ollama service..."
  sudo systemctl enable ollama || true
  sudo systemctl start ollama || true
fi

# Reachability check
if curl -fsS "${OLLAMA_ENDPOINT_DEFAULT}/api/tags" >/dev/null 2>&1; then
  log "Ollama reachable at ${OLLAMA_ENDPOINT_DEFAULT}."
else
  warn "Could not reach Ollama at ${OLLAMA_ENDPOINT_DEFAULT}. If you're not in a systemd environment, start Ollama manually (run: ollama serve) in another session."
fi

# ========= Project skeleton =========
log "Creating project at ${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# .gitignore to avoid committing secrets/db
cat > "${GITIGNORE_FILE}" <<'GIT'
.env
kb.sqlite
__pycache__/
*.pyc
GIT

# ========= Python venv & deps =========
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating Python venv..."
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

log "Writing requirements.txt..."
cat > "${REQ_FILE}" <<'REQS'
boto3
pypdf
python-dotenv
numpy
requests
rich
prompt_toolkit
REQS

log "Installing Python deps..."
pip install --upgrade pip
pip install -r "${REQ_FILE}"

# ========= .env.example =========
log "Writing .env.example (copy to .env and fill values)..."
cat > "${ENV_EXAMPLE}" <<ENV
# Copy this file to .env and fill your values
# Cloudian/HyperStore S3 endpoint (https://host[:port])
S3_ENDPOINT=https://<YOUR-CLOUDIAN-ENDPOINT>
S3_KEY=<YOUR-ACCESS-KEY>
S3_SECRET=<YOUR-SECRET-KEY>
S3_BUCKET=kb-demo
S3_PREFIX=docs/

# Ollama endpoint
OLLAMA=${OLLAMA_ENDPOINT_DEFAULT}

# Models
EMBED_MODEL=${EMBED_MODEL_DEFAULT}
CHAT_MODEL=${CHAT_MODEL_DEFAULT}
ENV

# ========= index.py =========
log "Writing index.py..."
cat > "${INDEX_FILE}" <<'PY'
import os, io, json, sqlite3
import boto3
import requests
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_KEY      = os.getenv("S3_KEY")
S3_SECRET   = os.getenv("S3_SECRET")
S3_BUCKET   = os.getenv("S3_BUCKET")
S3_PREFIX   = os.getenv("S3_PREFIX", "docs/")
OLLAMA      = os.getenv("OLLAMA", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

def chunk_text(text, chunk_size=1600, overlap=250):
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        yield text[i:i+chunk_size]
        i += step

def extract_text_from_bytes(key, blob):
    key_lower = key.lower()
    if key_lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(blob))
            return "\n".join([(page.extract_text() or "") for page in reader.pages])
        except Exception as e:
            print(f"  [warn] PDF parse failed for {key}: {e}")
            return ""
    else:
        return blob.decode("utf-8", errors="ignore")

def embed(text):
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]

def main():
    if not all([S3_ENDPOINT, S3_KEY, S3_SECRET, S3_BUCKET]):
        raise SystemExit("Missing S3 configuration. Fill .env first.")

    s3 = boto3.client("s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET)

    conn = sqlite3.connect("kb.sqlite")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        s3_key TEXT,
        pos INTEGER,
        text TEXT,
        vec TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_key ON chunks(s3_key);")

    paginator = s3.get_paginator('list_objects_v2')
    total = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for o in page.get("Contents", []):
            key = o["Key"]
            if key.endswith("/"):
                continue
            print(f"Indexing: {key}")
            body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            text = extract_text_from_bytes(key, body)
            if not text.strip():
                print("  (no extractable text)")
                continue
            cur.execute("DELETE FROM chunks WHERE s3_key=?", (key,))
            pos = 0
            for part in chunk_text(text):
                vec = embed(part)
                cur.execute(
                    "INSERT INTO chunks (s3_key, pos, text, vec) VALUES (?, ?, ?, ?)",
                    (key, pos, part, json.dumps(vec))
                )
                pos += 1
                total += 1
            conn.commit()

    print(f"Done. Total chunks indexed: {total}. DB: kb.sqlite")

if __name__ == "__main__":
    main()
PY

# ========= chat_cli.py =========
log "Writing chat_cli.py..."
cat > "${CHAT_FILE}" <<'PY'
#!/usr/bin/env python3
# chat_cli.py
import os
import sys
import json
import time
import sqlite3
from typing import List, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

load_dotenv()
OLLAMA      = os.getenv("OLLAMA", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "llama3.1:8b")

DB_PATH = "kb.sqlite"
TOP_K = 8  # retrieve more context for richer answers

# generation options (tweak with :mode)
LONG_OPTS = {"num_predict": 800, "temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.05}

console = Console()

def ensure_db():
    if not os.path.exists(DB_PATH):
        console.print(f"[red]Database {DB_PATH} not found. Run index.py first.[/red]")
        sys.exit(1)

def embed(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": EMBED_MODEL, "input": text},
        timeout=120
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def retrieve(query: str, k: int = TOP_K) -> List[Tuple[float, str, str, int]]:
    qv = embed(query)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT s3_key, pos, text, vec FROM chunks")
    results = []
    for s3_key, pos, text, vec_json in cur.fetchall():
        v = np.array(json.loads(vec_json), dtype=np.float32)
        score = cosine(qv, v)
        results.append((score, text, s3_key, pos))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]

def build_context(chunks: List[Tuple[float, str, str, int]]) -> str:
    blocks = []
    for score, text, s3_key, pos in chunks:
        blocks.append(f"[DOC:{s3_key} CHUNK:{pos} SCORE:{score:.3f}]\n{text}")
    return "\n\n---\n".join(blocks)

def stream_llm(prompt: str):
    resp = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": LONG_OPTS,
        },
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    stats = {}
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "response" in data and not data.get("done"):
            yield ("chunk", data["response"])
        if data.get("done"):
            stats = data
    yield ("done", stats)

def header():
    meta = Table.grid(expand=True)
    meta.add_column(justify="left")
    meta.add_column(justify="right")
    meta.add_row(
        f"[bold]RAG Chat Demo[/bold] • [green]{CHAT_MODEL}[/green] + [cyan]{EMBED_MODEL}[/cyan]",
        f"Ollama: [magenta]{OLLAMA}[/magenta]"
    )
    console.print(Panel(meta, border_style="blue", padding=(1,2)))

def help_panel():
    t = Table(title="Commands", show_header=False, box=None, padding=(0,1))
    t.add_row(":help", "Show this help")
    t.add_row(":sources", "Show sources used for the last answer")
    t.add_row(":clear", "Clear the screen")
    t.add_row(":mode long|short", "Adjust verbosity & length")
    t.add_row(":exit", "Exit")
    t.add_row("", "")
    t.add_row("[dim]Keys[/dim]", "[dim]Enter = send • Ctrl+J = newline[/dim]")
    console.print(Panel(t, border_style="cyan"))

def show_sources(chunks):
    tbl = Table(title="Top Retrieved Sources", show_lines=True)
    tbl.add_column("#", justify="right", width=2)
    tbl.add_column("S3 Key", overflow="fold")
    tbl.add_column("Chunk", justify="right", width=5)
    tbl.add_column("Score", justify="right", width=8)
    for i, (score, _text, key, pos) in enumerate(chunks, 1):
        tbl.add_row(str(i), key, str(pos), f"{score:.3f}")
    console.print(tbl)

def main():
    global LONG_OPTS
    ensure_db()
    console.clear()
    header()
    help_panel()

    kb = KeyBindings()
    @kb.add("enter")
    def _(event):
        event.app.current_buffer.validate_and_handle()
    @kb.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(history=InMemoryHistory(), key_bindings=kb)

    last_chunks = []
    last_stats = {}

    while True:
        try:
            user_q = session.prompt("\n[You] ", multiline=True)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]Bye![/bold yellow]")
            break

        cmd = user_q.strip()
        if cmd == "":
            continue
        if cmd.lower() in (":exit", ":quit"):
            console.print("[bold yellow]Bye![/bold yellow]")
            break
        if cmd.lower() == ":help":
            help_panel(); continue
        if cmd.lower() == ":clear":
            console.clear(); header(); continue
        if cmd.lower() == ":sources":
            if last_chunks: show_sources(last_chunks)
            else: console.print("[italic]No sources yet — ask something first.[/italic]")
            continue
        if cmd.lower().startswith(":mode"):
            parts = cmd.lower().split()
            if len(parts) == 2 and parts[1] in ("long", "short"):
                if parts[1] == "long":
                    LONG_OPTS = {"num_predict": 1200, "temperature": 0.35, "top_p": 0.92, "repeat_penalty": 1.05}
                    console.print("[green]Mode set to: long[/green]")
                else:
                    LONG_OPTS = {"num_predict": 300, "temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.1}
                    console.print("[yellow]Mode set to: short[/yellow]")
            else:
                console.print("Usage: :mode long|short")
            continue

        with Live(Spinner("dots", text=" retrieving from vector store..."), refresh_per_second=12):
            chunks = retrieve(cmd, k=TOP_K)
        last_chunks = chunks

        context = build_context(chunks)
        sys_prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Write a detailed, well-structured answer with clear paragraphs and bullet points when useful. "
            "When you rely on a specific snippet, add a short citation like [DOC:<key> CHUNK:<n>].\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {cmd}\n\nANSWER:"
        )

        console.print()
        console.rule("[bold green]Answer[/bold green]")
        with Live("", refresh_per_second=18) as live:
            answer_text = ""
            start = time.time()
            stats = {}
            for kind, payload in stream_llm(sys_prompt):
                if kind == "chunk":
                    answer_text += payload
                    live.update(Markdown(answer_text))
                else:
                    stats = payload
            dur = time.time() - start

        console.print(Markdown(answer_text))
        console.rule("[bold blue]Details[/bold blue]")
        show_sources(chunks)

        last_stats = stats if isinstance(stats, dict) else {}
        if last_stats:
            meta = Table.grid(padding=(0,1))
            meta.add_column(justify="left")
            meta.add_column(justify="right")
            meta.add_row("Duration (s)", f"{dur:.2f}")
            meta.add_row("Eval count", str(last_stats.get("eval_count", "—")))
            meta.add_row("Prompt eval count", str(last_stats.get("prompt_eval_count", "—")))
            console.print(Panel(meta, border_style="magenta"))

        console.print("[dim]Tips: Enter sends • Ctrl+J newline • :mode long|short • :sources shows retrieved chunks.[/dim]")

if __name__ == "__main__":
    main()
PY
chmod +x "${CHAT_FILE}"

# ========= Pull Ollama models (idempotent) =========
log "Ensuring Ollama models are available..."
HAVE_EMBED=0
HAVE_CHAT=0
if ollama list 2>/dev/null | grep -q "${EMBED_MODEL_DEFAULT}"; then HAVE_EMBED=1; fi
if ollama list 2>/dev/null | grep -q "${CHAT_MODEL_DEFAULT}"; then HAVE_CHAT=1; fi

if [[ "${HAVE_EMBED}" -eq 0 ]]; then
  log "Pulling embedding model: ${EMBED_MODEL_DEFAULT}"
  ollama pull "${EMBED_MODEL_DEFAULT}" || warn "Could not pull ${EMBED_MODEL_DEFAULT}. Try again later."
else
  log "Embedding model already present: ${EMBED_MODEL_DEFAULT}"
fi

if [[ "${HAVE_CHAT}" -eq 0 ]]; then
  log "Pulling chat model: ${CHAT_MODEL_DEFAULT}"
  ollama pull "${CHAT_MODEL_DEFAULT}" || warn "Could not pull ${CHAT_MODEL_DEFAULT}. Try again later."
else
  log "Chat model already present: ${CHAT_MODEL_DEFAULT}"
fi

log "Installation complete ✅"

cat <<NEXT

Next steps:
  1) Create your .env:
       cp ${ENV_EXAMPLE} ${PROJECT_DIR}/.env
       nano ${PROJECT_DIR}/.env
     Fill:
       S3_ENDPOINT (e.g., https://object.example.com:443)
       S3_KEY / S3_SECRET
       S3_BUCKET (e.g., kb-demo)
       S3_PREFIX (e.g., docs/)
       OLLAMA (leave default unless remote)

  2) Upload a few PDFs or .txt files to your bucket/prefix:
       aws --endpoint-url https://<YOUR-ENDPOINT> s3 mb s3://<BUCKET>
       aws --endpoint-url https://<YOUR-ENDPOINT> s3 cp ./sample.pdf s3://<BUCKET>/<PREFIX>/

  3) Build the SQLite index:
       source ${VENV_DIR}/bin/activate
       cd ${PROJECT_DIR}
       python index.py

  4) Chat with your docs:
       python chat_cli.py

Tips:
  - Enter = send, Ctrl+J = newline
  - Commands: :help, :sources, :mode long|short, :clear, :exit
NEXT