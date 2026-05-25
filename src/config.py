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
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:e4b")  # default to user's preloaded model

# Grounding and Chunking Configurations
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RELEVANCE_THRESHOLD = 0.65  # Target cosine similarity threshold (1 - cosine_distance >= 0.65)
TOP_K_CHUNKS = 5

def ensure_directories():
    """Utility to make sure all data and knowledge base folders exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in KNOWLEDGE_SUBDIRS.items():
        path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("All directories initialized successfully.")
