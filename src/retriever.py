import re
from typing import List, Dict, Any
from src.config import RELEVANCE_THRESHOLD, TOP_K_CHUNKS
from src.indexer import get_chroma_client, get_or_create_collection, EmbeddingHelper

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", 
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", 
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", 
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", 
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", 
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", 
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", 
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", 
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}

def extract_keywords(query: str) -> List[str]:
    """Helper to clean query and extract significant keywords for term matching."""
    words = re.findall(r"\b\w{2,}\b", query.lower())
    return [w for w in words if w not in STOPWORDS]


def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """Calculates term frequency score of query keywords within a text block."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def search_vector_db(query: str, limit: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
    """
    Performs pure semantic search using vector cosine distance in ChromaDB.
    Returns filtered and formatted documents with similarity scores.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    # Check if collection is empty
    if collection.count() == 0:
        return []

    embedder = EmbeddingHelper()
    query_vector = embedder.encode([query])[0]

    # Query Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=limit
    )

    if not results or not results["ids"] or not results["ids"][0]:
        return []

    processed_chunks = []
    
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for i in range(len(ids)):
        # Chroma cosine distance ranges from 0 to 2.
        # cosine_similarity = 1.0 - cosine_distance
        cosine_dist = distances[i]
        similarity = 1.0 - cosine_dist
        
        processed_chunks.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "similarity": similarity
        })

    return processed_chunks


def retrieve_relevant_context(query: str, threshold: float = RELEVANCE_THRESHOLD, limit: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
    """
    Retrieves, scores, and filters chunks using hybrid ranking (vector similarity + keyword boost)
    and strictly enforces a minimum similarity threshold for grounding.
    """
    semantic_chunks = search_vector_db(query, limit=limit * 2) # Get slightly more chunks to re-rank
    if not semantic_chunks:
        return []

    keywords = extract_keywords(query)
    
    scored_chunks = []
    for chunk in semantic_chunks:
        kw_score = calculate_keyword_score(chunk["text"], keywords)
        
        # Hybrid score combines: 80% vector similarity + 20% exact keyword presence
        hybrid_score = (chunk["similarity"] * 0.8) + (kw_score * 0.2)
        
        scored_chunks.append({
            **chunk,
            "hybrid_score": hybrid_score
        })

    # Sort primarily by hybrid score descending, then filter by semantic threshold to ensure grounding
    scored_chunks.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Filter by absolute semantic similarity threshold to maintain grounded truth
    grounded_chunks = [c for c in scored_chunks if c["similarity"] >= threshold]
    
    # Return top K chunks requested
    return grounded_chunks[:limit]


if __name__ == "__main__":
    test_query = "What is the authentication refresh workflow?"
    print(f"Keywords extracted: {extract_keywords(test_query)}")
    print("Testing retrieve_relevant_context...")
    results = retrieve_relevant_context(test_query)
    print(f"Retrieved {len(results)} grounded chunks.")
