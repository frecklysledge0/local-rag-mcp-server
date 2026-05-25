import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.retriever import search_vector_db, retrieve_relevant_context

# Target Evaluation Dataset
# Each test case specifies a query and the ground-truth document name expected to be retrieved.
EVAL_DATASET = [
    {
        "query": "How do you calibrate the hydraulic pump H-400?",
        "expected_doc": "pump_manual.md"
    },
    {
        "query": "hexagonal locknut gauge pressure exactly 2500 PSI",
        "expected_doc": "pump_manual.md"
    },
    {
        "query": "What is the token expiration limit for access tokens?",
        "expected_doc": "auth_guide.md"
    },
    {
        "query": "POST to /api/auth/refresh to rotate JWT refresh tokens",
        "expected_doc": "auth_guide.md"
    },
    {
        "query": "How to securely store JWT refresh tokens in HttpOnly cookie?",
        "expected_doc": "auth_guide.md"
    }
]

def calculate_metrics(results: List[Dict[str, Any]], expected_doc: str, top_k: int = 5) -> Dict[str, Any]:
    """Computes MRR, Hit Rate, and Rank for a specific expected document in the retrieval results."""
    rank = 999
    hit = 0.0
    reciprocal_rank = 0.0

    for idx, doc in enumerate(results[:top_k]):
        fname = doc["metadata"].get("file_name", "")
        if fname == expected_doc:
            rank = idx + 1
            hit = 1.0
            reciprocal_rank = 1.0 / rank
            break
            
    return {
        "hit": hit,
        "mrr": reciprocal_rank,
        "rank": rank
    }


def evaluate_retrieval_system(system_name: str, retrieval_fn) -> Dict[str, Any]:
    """Runs the evaluation dataset against a specific retrieval function and averages metrics."""
    total_mrr = 0.0
    total_hit_rate_at_1 = 0.0
    total_hit_rate_at_3 = 0.0
    total_hit_rate_at_5 = 0.0
    case_metrics = []

    for case in EVAL_DATASET:
        query = case["query"]
        expected = case["expected_doc"]
        
        # Run retrieval
        results = retrieval_fn(query)
        
        # Compute metrics
        m1 = calculate_metrics(results, expected, top_k=1)
        m3 = calculate_metrics(results, expected, top_k=3)
        m5 = calculate_metrics(results, expected, top_k=5)
        
        total_hit_rate_at_1 += m1["hit"]
        total_hit_rate_at_3 += m3["hit"]
        total_hit_rate_at_5 += m5["hit"]
        total_mrr += m5["mrr"]

        case_metrics.append({
            "query": query,
            "expected_document": expected,
            "rank_obtained": m5["rank"] if m5["rank"] < 999 else "Not Found",
            "hit_at_5": m5["hit"],
            "mrr_at_5": m5["mrr"]
        })

    num_cases = len(EVAL_DATASET)
    return {
        "system_name": system_name,
        "mrr": total_mrr / num_cases,
        "recall_at_1": total_hit_rate_at_1 / num_cases,
        "recall_at_3": total_hit_rate_at_3 / num_cases,
        "recall_at_5": total_hit_rate_at_5 / num_cases,
        "details": case_metrics
    }


def run_evaluation():
    print("======================================================================")
    print("🧪 Starting Local RAG Version 2 Comparative Evaluation Suite")
    print("======================================================================")

    # 1. Evaluate Pure Vector Search (baseline)
    # Wrap standard vector DB search to conform to expected retrieval interface
    def baseline_vector_search(query: str):
        return search_vector_db(query, limit=10)

    vector_metrics = evaluate_retrieval_system("Baseline Vector Search (Version 1)", baseline_vector_search)

    # 2. Evaluate Upgraded V2 Hybrid + Cross-Encoder Reranking
    def upgraded_hybrid_search(query: str):
        return retrieve_relevant_context(query, limit=5)

    upgraded_metrics = evaluate_retrieval_system("Upgraded Hybrid + RRF + Cross-Encoder (Version 2)", upgraded_hybrid_search)

    # Render results table
    print("\n📈 SUMMARY METRICS COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<30} | {'Baseline Vector V1':<22} | {'Upgraded Hybrid V2':<22}")
    print("-" * 80)
    print(f"{'Mean Reciprocal Rank (MRR)':<30} | {vector_metrics['mrr']:<22.2%} | {upgraded_metrics['mrr']:<22.2%}")
    print(f"{'Recall @ 1 (Top Match Accuracy)':<30} | {vector_metrics['recall_at_1']:<22.2%} | {upgraded_metrics['recall_at_1']:<22.2%}")
    print(f"{'Recall @ 3':<30} | {vector_metrics['recall_at_3']:<22.2%} | {upgraded_metrics['recall_at_3']:<22.2%}")
    print(f"{'Recall @ 5':<30} | {vector_metrics['recall_at_5']:<22.2%} | {upgraded_metrics['recall_at_5']:<22.2%}")
    print("-" * 80)

    print("\n🔬 DETAILED CASE METRICS:")
    for idx, case in enumerate(upgraded_metrics["details"]):
        v_case = vector_metrics["details"][idx]
        print(f"\n👉 Query #{idx+1}: '{case['query']}'")
        print(f"   Target Doc: {case['expected_document']}")
        print(f"   [V1 Vector] Rank: {v_case['rank_obtained']} | MRR: {v_case['mrr_at_5']:.2f}")
        print(f"   [V2 Hybrid] Rank: {case['rank_obtained']} | MRR: {case['mrr_at_5']:.2f}")


if __name__ == "__main__":
    run_evaluation()
