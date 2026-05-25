import re
import math
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from src.config import (
    RELEVANCE_THRESHOLD, 
    TOP_K_CHUNKS, 
    CROSS_ENCODER_MODEL_NAME, 
    LOGS_DIR,
    ensure_directories
)
from src.indexer import get_chroma_client, get_or_create_collection, EmbeddingHelper

# Configure logging
ensure_directories()
LOG_FILE = LOGS_DIR / "retrieval.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class BM25:
    """
    Highly optimized, self-contained implementation of the Okapi BM25 ranking algorithm.
    """
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus  # List of chunks: {'id': str, 'text': str, 'metadata': dict}
        self.N = len(corpus)
        
        self.doc_lengths = []
        self.doc_term_freqs = []  # List of Dict[term, freq]
        self.df = {}  # Document frequency for terms
        
        total_len = 0
        for doc in corpus:
            tokens = self._tokenize(doc["text"])
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)
            
            for token in tf.keys():
                self.df[token] = self.df.get(token, 0) + 1
                
        self.avg_doc_len = (total_len / self.N) if self.N > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        # Basic lowercase token extraction
        return re.findall(r"\b\w{2,}\b", text.lower())

    def get_score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf_dict = self.doc_term_freqs[doc_idx]
        
        for token in query_tokens:
            if token not in tf_dict:
                continue
            
            n = self.df.get(token, 0)
            # Okapi BM25 IDF formulation with smoothing
            idf = math.log(((self.N - n + 0.5) / (n + 0.5)) + 1.0)
            
            tf = tf_dict[token]
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            
            score += idf * (numerator / denominator)
        return score

    def rank(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
            
        scored_docs = []
        for idx, doc in enumerate(self.corpus):
            score = self.get_score(query_tokens, idx)
            if score > 0.0:
                scored_docs.append({
                    **doc,
                    "bm25_score": score
                })
                
        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_docs[:limit]


class CrossEncoderHelper:
    """
    Singleton class to manage the local Cross-Encoder re-ranking model.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CrossEncoderHelper, cls).__new__(cls)
            logging.info(f"Loading local Cross-Encoder model: {CROSS_ENCODER_MODEL_NAME}")
            cls._instance.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return cls._instance

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """Calculates relevance logits (sigmoid probabilities) for query-document pairs."""
        return self.model.predict(pairs, show_progress_bar=False).tolist()


def reciprocal_rank_fusion(vector_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], c: int = 60) -> List[Dict[str, Any]]:
    """
    Merges Vector and BM25 search rankings using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    doc_mapping = {}

    # Accumulate Vector ranks
    for rank_idx, doc in enumerate(vector_results):
        doc_id = doc["id"]
        doc_mapping[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (c + rank_idx + 1))

    # Accumulate BM25 ranks
    for rank_idx, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        doc_mapping[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (c + rank_idx + 1))

    # Sort merged documents by fused RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for doc_id, rrf_score in sorted_ids:
        fused_results.append({
            **doc_mapping[doc_id],
            "rrf_score": rrf_score
        })
    return fused_results


def search_vector_db(query: str, limit: int = 50, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB vector collection with optional metadata filters.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    if collection.count() == 0:
        return []

    embedder = EmbeddingHelper()
    query_vector = embedder.encode([query])[0]

    # Convert standard where filters
    where_clause = None
    if filters:
        # If multiple filters, combine in Chroma's dict format
        if len(filters) > 1:
            where_clause = {"$and": [{k: v} for k, v in filters.items()]}
        else:
            where_clause = filters

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=limit,
        where=where_clause
    )

    if not results or not results["ids"] or not results["ids"][0]:
        return []

    processed_chunks = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for i in range(len(ids)):
        cosine_dist = distances[i]
        similarity = 1.0 - cosine_dist
        processed_chunks.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "similarity": similarity
        })
    return processed_chunks


def search_bm25(query: str, limit: int = 50, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Fetches matching metadata-filtered documents from ChromaDB and runs the local BM25 engine.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    if collection.count() == 0:
        return []

    # Retrieve matching chunks to build transient BM25 index
    where_clause = None
    if filters:
        if len(filters) > 1:
            where_clause = {"$and": [{k: v} for k, v in filters.items()]}
        else:
            where_clause = filters

    # Fetch all chunks satisfying filter constraints
    all_data = collection.get(include=["documents", "metadatas"], where=where_clause)
    if not all_data or not all_data["ids"]:
        return []

    corpus = []
    for i in range(len(all_data["ids"])):
        corpus.append({
            "id": all_data["ids"][i],
            "text": all_data["documents"][i],
            "metadata": all_data["metadatas"][i]
        })

    bm25_engine = BM25(corpus)
    return bm25_engine.rank(query, limit=limit)


def retrieve_relevant_context(
    query: str, 
    limit: int = TOP_K_CHUNKS,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    V2.0 Hybrid Search + RRF Blender + Cross-Encoder Re-ranker Pipeline.
    Retrieves candidate documents, blends vector and token indices, and applies Cross-Encoder sorting.
    """
    logging.info(f"Retrieval Request: query='{query}', filters={filters}")
    
    # 1. Parallel Candidate Retrieval (Retrieve up to 30 chunks from both systems)
    vector_candidates = search_vector_db(query, limit=30, filters=filters)
    bm25_candidates = search_bm25(query, limit=30, filters=filters)

    # 2. Reciprocal Rank Fusion (RRF)
    fused_candidates = reciprocal_rank_fusion(vector_candidates, bm25_candidates)
    if not fused_candidates:
        logging.info("Retrieval returned 0 fused candidates.")
        return []

    # 3. Cross-Encoder Re-ranking (Evaluate top 20 candidates)
    rerank_pool = fused_candidates[:20]
    reranker = CrossEncoderHelper()
    
    pairs = [[query, doc["text"]] for doc in rerank_pool]
    rerank_scores = reranker.predict(pairs)

    for idx, doc in enumerate(rerank_pool):
        doc["rerank_score"] = rerank_scores[idx]

    # Sort primarily by Cross-Encoder score descending
    rerank_pool.sort(key=lambda x: x["rerank_score"], reverse=True)

    # V2.0 Dynamic / Adaptive Relevance Thresholding
    # Cross-Encoder score (logits converted) reflects absolute semantic probability.
    # We enforce a dynamic minimum cutoff: if the best document's rerank_score is low,
    # the entire query is marked as low confidence to prevent hallucinations.
    best_score = rerank_pool[0]["rerank_score"] if rerank_pool else -99.0
    
    # Sigmoid score threshold of 0.15 represents the target relevance ceiling
    confidence_threshold = 0.15 
    
    logging.info(f"Re-ranking complete. Candidates evaluated={len(rerank_pool)}, Top Rerank Score={best_score:.4f}")

    if best_score < confidence_threshold:
        logging.warning(f"Retrieval confidence ({best_score:.4f}) below dynamic threshold ({confidence_threshold}). Rejecting context.")
        # Return empty list to trigger downstream programmatic fallback
        return []

    # Select final K chunks
    final_context = rerank_pool[:limit]
    
    # Log details of top retrieved context for Version 2.1 auditing
    log_data = {
        "query": query,
        "filters": filters,
        "selected_chunks_count": len(final_context),
        "top_rerank_score": best_score,
        "chunks": [
            {
                "id": c["id"],
                "file": c["metadata"].get("file_name"),
                "similarity": c.get("similarity", 0.0),
                "bm25_score": c.get("bm25_score", 0.0),
                "rrf_score": c.get("rrf_score", 0.0),
                "rerank_score": c.get("rerank_score", 0.0)
            }
            for c in final_context
        ]
    }
    logging.info(f"Context Selection: {json.dumps(log_data)}")
    
    return final_context


if __name__ == "__main__":
    print("Testing upgraded Hybrid Retrieval + RRF + Re-ranker pipeline...")
    res = retrieve_relevant_context("How do you calibrate the pump?")
    print(f"Retrieved {len(res)} chunks.")
    for d in res:
        print(f"File: {d['metadata']['file_name']} | Rerank Score: {d['rerank_score']:.4f}")
