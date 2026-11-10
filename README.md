# Local RAG Chat

A fully local chat application that answers questions like a normal AI
assistant while using your documents as supporting evidence. Documents help
verify and enrich answers; they do not restrict what the assistant can say.

- Normal, conversational answers from `qwen3:4b`.
- Documents are used for verification and support, not as a hard boundary.
- The assistant is instructed not to refuse with "I don't know based on the
  provided documents"; if the documents are not relevant, it answers from its
  own knowledge.
- When document excerpts are relevant, the answer cites the source file name.
- Add documents by path, from chat, or at startup.
- Persistent chat history with automatic session IDs or named sessions.
- Chat messages are stored with embeddings and relevant older messages are
  recalled as conversation memory.
- Runs locally with Ollama. No cloud APIs and no PostgreSQL required.

## How it works

```text
User question
     │
     ├──> embed question ──> search document vectors (.index/index.json)
     │                              │
     │                              └──> relevant excerpts (support only)
     │
     ├──> search earlier session messages by vector similarity
     │
     └──> Ollama qwen3:4b ──> normal answer + supporting document sources
```

1. Documents in `documents/` are split into chunks and embedded with
   `nomic-embed-text` via Ollama.
2. The document vectors are cached in `.index/index.json`.
3. Each question is embedded once and used to retrieve relevant document
   chunks and relevant earlier chat messages.
4. The model answers normally. Document excerpts are passed as supporting
   evidence, and only chunks above the relevance threshold are used.
5. The question and answer are saved in the session database with their
   embeddings.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally
- Ollama models: `qwen3:4b` and `nomic-embed-text`
- Python packages: `numpy`, `requests`, `pypdf`

## Setup

```bash
# Start Ollama if it is not already running.
ollama serve

# Pull the models once.
ollama pull qwen3:4b
ollama pull nomic-embed-text

# Install Python dependencies.
python -m pip install -r requirements.txt
```

The helper script uses `$HOME/miniconda3/envs/local-rag-app/bin/python` when
that environment exists, otherwise it uses `python3`. Override it with the
`PYTHON` environment variable if needed.

## Usage

### Interactive chat

```bash
./scripts/run.sh
```

This creates a new auto-generated session ID and starts a normal chat.

Resume or create a named session:

```bash
./scripts/run.sh --session my-study
```

If the name does not exist, a new named session is created. If it exists, the
previous conversation is loaded.

### One-shot question

```bash
./scripts/run.sh ask "What is MVCC?"
./scripts/run.sh ask "What is the capital of Japan?" --session geography
```

The question and answer are saved in the session, so you can resume it later
with `--session geography`.

### Add documents

```bash
# Copy a document into documents/ and rebuild the index.
./scripts/run.sh add "/path/to/notes.md"
./scripts/run.sh add "/path/to/report.pdf"

# Add one or more documents before starting chat or asking a question.
./scripts/run.sh --document "/path/to/report.pdf" --session my-study
./scripts/run.sh ask "Summarize the report" --document "/path/to/report.pdf"

# Add a document from inside chat.
/add /path/to/report.pdf
```

Supported formats: `.txt`, `.md`, `.markdown`, and text-based `.pdf`. PDF
text is extracted with `pypdf`. Scanned/image-only PDFs need OCR first because
they contain no extractable text.

### Sessions

```bash
# List recent sessions with ids, names, message counts, and timestamps.
./scripts/run.sh sessions

# Resume a session by id or name.
./scripts/run.sh --session my-study
./scripts/run.sh --session e1098a966961
```

Sessions are stored in `.data/sessions.db` (SQLite). Each message is stored
with its embedding, and semantically relevant earlier messages are recalled
when answering new questions. The last few messages are also included as
normal chat history for fluent multi-turn conversation.

### Index maintenance

```bash
./scripts/run.sh ingest   # rebuild the document index
./scripts/run.sh reset    # delete the cached document index
```

The index is rebuilt automatically when documents change. `reset` only removes
the document index; sessions are kept.

## Chat commands

| Command | Description |
| --- | --- |
| `/add <path>` | Copy a `.txt`/`.md`/`.pdf` document into `documents/` and reindex |
| `/docs` | List loaded documents |
| `/reindex` | Rebuild the document index |
| `/session` | Show the current session id and name |
| `/rename <name>` | Give the current session a name |
| `/new [name]` | Start a new session, optionally named |
| `/sessions` | List recent sessions |
| `/help` | Show available commands |
| `/exit` | Quit |

## Documents and verification

Documents are treated as supporting evidence, not as a restriction:

- If a retrieved chunk is relevant, the model uses it to verify and enrich
  the answer and cites the source file.
- If no chunk passes the relevance threshold (`MIN_DOC_SCORE`), the model
  answers normally from its own knowledge.
- The assistant is explicitly instructed never to say that information is
  unavailable in the documents and never to fabricate document content.

Relevance filtering helps avoid injecting unrelated document text into the
conversation. You can tune it with `MIN_DOC_SCORE` (default `0.5`).

## Configuration

All settings are environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen3:4b` | Chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_BACKEND` | `ollama` | `ollama` or `sentence-transformers` |
| `DOCUMENTS_DIR` | `./documents` | Folder with source documents |
| `INDEX_DIR` | `./.index` | Folder for the document vector index |
| `DATA_DIR` | `./.data` | Folder for local app data |
| `SESSIONS_DB` | `./.data/sessions.db` | SQLite session database |
| `CHUNK_SIZE` | `500` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Chunk overlap in characters |
| `TOP_K` | `3` | Number of document chunks to retrieve |
| `MIN_DOC_SCORE` | `0.5` | Minimum similarity for document support |
| `MEMORY_MESSAGES` | `3` | Number of earlier messages recalled by vector search |

Example:

```bash
OLLAMA_MODEL=qwen3:4b TOP_K=5 MIN_DOC_SCORE=0.45 ./scripts/run.sh
```

## Project structure

```text
app/
  main.py                 CLI, chat loop, sessions, prompting
  config.py               Environment-based configuration
  ingestion/
    loader.py             Load and save documents
    chunker.py            Sentence-aware chunking with overlap
  embeddings/
    embedder.py           Ollama or sentence-transformers embeddings
  retrieval/
    retriever.py          Local vector index and similarity search
  llm/
    ollama_client.py      Ollama chat client
  database/
    sessions.py           SQLite sessions and vector message history
documents/                Your source documents
scripts/run.sh            Convenience launcher
tests/                    Existing embedding experiment
```

The current implementation uses local JSON and SQLite storage, so
PostgreSQL/pgvector is not required. The database package is intentionally
small and can be replaced with PostgreSQL/pgvector later.

## Troubleshooting

**`Could not reach Ollama ...`**

Start Ollama:

```bash
ollama serve
```

**`Ollama returned an unexpected embedding response` or missing model**

Install the models:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

**The assistant ignores my document**

The question may not be similar enough to any chunk. Lower `MIN_DOC_SCORE`,
increase `TOP_K`, or run `/reindex` after adding documents.

**A PDF is skipped or has no text**

The PDF is probably scanned/image-only. Run OCR on it first, then add the
resulting text or searchable PDF. Password-protected PDFs are skipped too.

**I want a completely fresh start**

```bash
./scripts/run.sh reset
rm .data/sessions.db
```

The first command removes the document index; the second removes all saved
chat sessions.
