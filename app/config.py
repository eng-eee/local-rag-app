import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / ".index"))
INDEX_FILE = INDEX_DIR / "index.json"

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / ".data"))
SESSIONS_DB = Path(os.getenv("SESSIONS_DB", DATA_DIR / "sessions.db"))

# Embeddings: "ollama" uses nomic-embed-text, "sentence-transformers"
# uses a local HuggingFace model (needs torch installed).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ollama")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_DOC_SCORE = float(os.getenv("MIN_DOC_SCORE", "0.5"))
MEMORY_MESSAGES = int(os.getenv("MEMORY_MESSAGES", "3"))

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}
