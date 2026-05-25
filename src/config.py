import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
CHROMA_DB_DIR = DATA_DIR / "chroma"

# Subfolders in Knowledge Base for organization
KNOWLEDGE_SUBDIRS = {
    "docs": KNOWLEDGE_BASE_DIR / "docs",
    "papers": KNOWLEDGE_BASE_DIR / "papers",
    "notes": KNOWLEDGE_BASE_DIR / "notes"
}

# RAG Configurations
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # local embedding via sentence-transformers
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Cross-Encoder model
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:e4b")  # default to user's preloaded model

# Grounding and Chunking Configurations
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RELEVANCE_THRESHOLD = 0.65  # Default baseline cosine similarity threshold
TOP_K_CHUNKS = 5
LOGS_DIR = DATA_DIR / "logs"

def validate_indexing_path(path_str: str) -> Path:
    """
    Validates that a path is safe to index:
    1. Must exist.
    2. Must be within the designated project root to prevent path traversal or system folder leakage.
    3. Forbidden for sensitive system paths (e.g. /, /etc, ~/Library, /System).
    """
    resolved_path = Path(path_str).resolve()
    if not resolved_path.exists():
        raise ValueError(f"Target indexing path does not exist: {path_str}")

    # Explicit list of forbidden paths on macOS/Unix
    forbidden = [
        Path("/"), Path("/System"), Path("/Library"), Path("/usr"), Path("/var"), Path("/bin"), Path("/sbin"), Path("/etc"),
        Path(os.path.expanduser("~")), Path(os.path.expanduser("~/Library"))
    ]
    if resolved_path in forbidden or any(f == resolved_path for f in forbidden):
        raise ValueError(f"Security violation: indexing of system directory forbidden: {path_str}")

    # Enforce base folder containment: must reside within BASE_DIR
    try:
        resolved_path.relative_to(BASE_DIR)
    except ValueError:
        raise ValueError(f"Security boundary: path must reside strictly inside the project root: {BASE_DIR}")

    return resolved_path

def ensure_directories():
    """Utility to make sure all data, logs, and knowledge base folders exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in KNOWLEDGE_SUBDIRS.items():
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("All directories initialized successfully.")
