# 🧠 RAG Demo Bootstrap (Ollama + Cloudian/Any S3)

This project gives you a one-command installer that sets up a **Retrieval-Augmented Generation (RAG)** demo on an Ubuntu server.

- **Object storage (Cloudian HyperStore)holds your documents (PDF or text).
- **Ollama** runs locally to generate **embeddings** and **answers** (no external AI services).
- **SQLite** stores document chunks + vectors.
- A **Rich TUI** (terminal user interface) lets you chat with your documents.

> The repo intentionally contains only **two** files:
> 1) `install_rag_demo.sh` – the one-shot installer  
> 2) `README.md` – this guide

Everything else (code, venv, models, etc.) is created by the installer under `~/rag-demo/`.

---

## ✅ Requirements

- Ubuntu Server 24.04.3 LTS (root/sudo access)
- Internet connectivity (to install packages and pull models)
- An S3-compatible endpoint (Cloudian HyperStore recommended) + bucket credentials  
- ~10 GB free disk space (for models + dependencies)

The installer will **install Ollama** (if missing), **enable/start** the Ollama service, **pull the models**, and set up the entire demo.

---

## 🚀 Quick Start (TL;DR)

```bash
# 1) Clone and run installer
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
chmod +x install_rag_demo.sh
bash ./install_rag_demo.sh

# 2) Create and edit your environment file
cp ~/rag-demo/.env.example ~/rag-demo/.env
nano ~/rag-demo/.env   # fill values (see detailed guide below)

# 3) Upload a few PDFs or .txt files to your bucket/prefix
aws --endpoint-url https://<ENDPOINT> s3 mb s3://kb-demo
aws --endpoint-url https://<ENDPOINT> s3 cp ./sample.pdf s3://kb-demo/docs/

# 4) Build the local index (embeddings into SQLite)
source ~/rag-demo/.venv/bin/activate
python ~/rag-demo/index.py

# 5) Start the chat UI
python ~/rag-demo/chat_cli.py