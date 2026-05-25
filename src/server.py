import json
import sys
from pathlib import Path
from typing import Optional
from fastmcp import FastMCP

# Add project root to path for external execution hosts
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import ensure_directories, OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME, RELEVANCE_THRESHOLD
from src.indexer import scan_and_index_knowledge_base, load_manifest
from src.retriever import retrieve_relevant_context
from src.llm import generate_grounded_answer

# Initialize FastMCP Server
mcp = FastMCP("Antigravity Local RAG")

@mcp.tool()
def index_knowledge_base() -> str:
    """
    Scans the local `./data/knowledge_base/` folder, detects new, modified, or deleted files,
    creates text chunks, generates local embeddings, and updates the ChromaDB vector database.
    """
    try:
        stats = scan_and_index_knowledge_base()
        output = "✅ **Knowledge Base Indexing Completed Successfully!**\n\n"
        output += f"- 📁 Files Newly Indexed / Updated: `{stats['indexed_files']}`\n"
        output += f"- ⏭️ Files Unchanged (Skipped): `{stats['skipped_files']}`\n"
        output += f"- 🗑️ Files Cleaned Up (Removed from DB): `{stats['deleted_files']}`\n"
        output += f"- 🧩 Total Vector Chunks Ingested: `{stats['total_chunks_added']}`\n"
        return output
    except Exception as e:
        return f"❌ **Error indexing knowledge base:** `{str(e)}`"


@mcp.tool()
def search_knowledge_base(query: str, limit: int = 5, category: Optional[str] = None) -> str:
    """
    Searches the local knowledge base using hybrid retrieval (vector similarity + BM25)
    and strictly filters matching chunks below the similarity threshold to prevent hallucinations.
    
    Args:
        query: The user query or search terms.
        limit: Max number of document chunks to return (default is 5).
        category: Optional category folder to restrict search (e.g. 'docs', 'papers', 'notes').
    """
    try:
        filters = {}
        if category:
            filters["category"] = category
            
        chunks = retrieve_relevant_context(query, limit=limit, filters=filters if filters else None)
        if not chunks:
            return "🔍 **No relevant documents found in the local knowledge base** that met the grounding criteria."
            
        output = f"🔍 **Top {len(chunks)} Grounded Search Matches for: '{query}'**\n\n"
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            fname = meta.get("file_name", "Unknown File")
            path = meta.get("source", "")
            cat = meta.get("category", "General")
            score = chunk.get("rerank_score", 0.0)
            
            output += f"### Match #{idx+1} | {fname} (Category: {cat})\n"
            output += f"- **Re-rank Confidence Score**: `{score:.2f}`\n"
            output += f"- **Source Path**: `file://{path}`\n"
            output += f"- **Content Chunk**:\n```text\n{chunk['text']}\n```\n\n"
        return output
    except Exception as e:
        return f"❌ **Error executing vector search:** `{str(e)}`"


@mcp.tool()
def ask_knowledge_base(query: str, category: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """
    Executes a fully offline and strictly grounded Q&A workflow.
    Retrieves relevant text chunks from the vector database, validates relevance scores,
    and runs a local LLM prompt to generate an answer with inline source citations.
    Maintains conversational memory context across sequential turns if session_id is supplied.
    
    Args:
        query: The direct question to answer from the local documents.
        category: Optional category folder to restrict search (e.g. 'docs', 'papers', 'notes').
        session_id: Optional thread identifier to resolve coreference history memory.
    """
    try:
        filters = {}
        if category:
            filters["category"] = category
            
        # 1. Coreference Resolution / Query Condensation if history is available
        refined_query = query
        if session_id:
            from src.llm import ConversationMemory, condense_query_with_history
            history = ConversationMemory.get_history(session_id)
            if history:
                refined_query = condense_query_with_history(query, history)
                
        # 2. Retrieve candidates using refined query & filters
        chunks = retrieve_relevant_context(refined_query, filters=filters if filters else None)
        
        # 3. Synthesize grounded answer
        return generate_grounded_answer(query, chunks, session_id=session_id)
    except Exception as e:
        return f"❌ **Error running grounded inference:** `{str(e)}`"


@mcp.resource("config://status")
def get_system_status() -> str:
    """
    Provides real-time diagnostic information regarding the local RAG engine status,
    active models, relevance thresholds, and currently indexed files.
    """
    try:
        manifest = load_manifest()
        
        info = {
            "mcp_server_name": "Antigravity Local RAG",
            "status": "Online",
            "active_models": {
                "embeddings": EMBEDDING_MODEL_NAME,
                "local_llm": OLLAMA_MODEL_NAME
            },
            "grounding_parameters": {
                "cosine_relevance_threshold": RELEVANCE_THRESHOLD,
                "default_chunks_limit": 5
            },
            "indexed_files_count": len(manifest),
            "indexed_files": manifest
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return json.dumps({"status": "Error", "details": str(e)})


if __name__ == "__main__":
    ensure_directories()
    # FastMCP automatically starts stdio server when run directly
    mcp.run()
