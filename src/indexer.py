import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DB_DIR, 
    KNOWLEDGE_BASE_DIR, 
    EMBEDDING_MODEL_NAME, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    ensure_directories
)
from src.parser import chunk_document

MANIFEST_FILE = CHROMA_DB_DIR / "indexing_manifest.json"

class EmbeddingHelper:
    """
    A singleton helper class to load and generate sentence embeddings locally using SentenceTransformers.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingHelper, cls).__new__(cls)
            cls._instance.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._instance

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of strings."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]


def get_chroma_client():
    """Initializes and returns a persistent Chroma client."""
    ensure_directories()
    # In newer Chroma versions, PersistentClient is the preferred API
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_or_create_collection(client):
    """Retrieves or creates the vector database collection configured with cosine space."""
    return client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a file to detect changes."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""


def load_manifest() -> Dict[str, Dict[str, Any]]:
    """Loads the file indexing manifest to check cache status."""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: Dict[str, Dict[str, Any]]):
    """Saves the file indexing manifest."""
    ensure_directories()
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def delete_document_from_db(collection, file_path_str: str):
    """Deletes all chunks associated with a specific file path from ChromaDB."""
    # ChromaDB supports filtering deletion by metadata
    try:
        collection.delete(where={"source": file_path_str})
        print(f"Cleared existing index for: {file_path_str}")
    except Exception as e:
        print(f"Error deleting old chunks for {file_path_str}: {e}")


def index_file(collection, file_path: Path, embedder: EmbeddingHelper) -> int:
    """
    Chunks a single document, generates embeddings, and indexes it into ChromaDB.
    """
    file_path_str = str(file_path)
    chunks = chunk_document(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        print(f"Skipping empty or unparseable file: {file_path.name}")
        return 0

    print(f"Indexing {file_path.name}: splitting into {len(chunks)} chunks...")
    
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    
    ids = [f"{file_path.name}_chunk_{idx}" for idx in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )
    return len(chunks)


def scan_and_index_knowledge_base() -> Dict[str, Any]:
    """
    Scans the knowledge base directory, indexes new or modified documents,
    and purges deleted documents from ChromaDB.
    """
    ensure_directories()
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    manifest = load_manifest()
    
    embedder = EmbeddingHelper()
    
    # 1. Find all active files in knowledge base folder
    active_files = {}
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        for file in files:
            if file.startswith("."): # ignore hidden files (like .DS_Store)
                continue
            path = Path(root) / file
            active_files[str(path)] = path

    stats = {
        "indexed_files": 0,
        "skipped_files": 0,
        "deleted_files": 0,
        "total_chunks_added": 0
    }
    
    updated_manifest = {}

    # 2. Add or update files
    for file_path_str, file_path in active_files.items():
        file_hash = compute_file_hash(file_path)
        mtime = file_path.stat().st_mtime
        
        # Check if unchanged
        cached_info = manifest.get(file_path_str)
        if cached_info and cached_info.get("hash") == file_hash:
            stats["skipped_files"] += 1
            updated_manifest[file_path_str] = cached_info
            continue
        
        # If it was modified or is new, delete previous index entries first
        if cached_info:
            delete_document_from_db(collection, file_path_str)
            
        try:
            chunks_count = index_file(collection, file_path, embedder)
            if chunks_count > 0:
                stats["indexed_files"] += 1
                stats["total_chunks_added"] += chunks_count
                updated_manifest[file_path_str] = {
                    "hash": file_hash,
                    "last_modified": mtime,
                    "chunks": chunks_count
                }
        except Exception as e:
            print(f"Failed to index {file_path.name}: {e}")
            if cached_info:
                # Retain old manifest cache if indexing failed so we try again next time
                updated_manifest[file_path_str] = cached_info

    # 3. Clean up deleted files
    for file_path_str in manifest.keys():
        if file_path_str not in active_files:
            print(f"Detected deletion of: {file_path_str}")
            delete_document_from_db(collection, file_path_str)
            stats["deleted_files"] += 1

    save_manifest(updated_manifest)
    return stats


if __name__ == "__main__":
    print("Starting knowledge base scanning and indexing...")
    results = scan_and_index_knowledge_base()
    print("Indexing Complete!")
    print(json.dumps(results, indent=2))
