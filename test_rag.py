import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import KNOWLEDGE_BASE_DIR, ensure_directories
from src.indexer import scan_and_index_knowledge_base
from src.retriever import retrieve_relevant_context
from src.llm import generate_grounded_answer

def setup_test_documents():
    """Writes realistic test files to the knowledge base for grounding testing."""
    ensure_directories()
    
    # 1. JWT Authentication Guide (Present category)
    auth_content = """# JWT Authentication Guide
This project uses standard JWT token-based authentication to secure all endpoints.
- Access Tokens: Valid for exactly 15 minutes.
- Refresh Tokens: Valid for exactly 7 days.
- Rotation: When a new access token is requested via `POST /api/auth/refresh`, a new refresh token is also rotated.
- Secure Storage: Refresh tokens must be stored in a secure, HttpOnly, SameSite=Strict cookie to prevent XSS attacks.
"""
    auth_file = KNOWLEDGE_BASE_DIR / "docs" / "auth_guide.md"
    with open(auth_file, "w", encoding="utf-8") as f:
        f.write(auth_content)
    print(f"Created test doc: {auth_file.name}")

    # 2. Hydraulic Pump Calibration Manual (Present category)
    pump_content = """# Hydraulic Pump Calibration Manual (Model H-400)
Follow these exact steps to calibrate the hydraulic pump:
1. Shut down all power and close the main supply valve to release residual pressure.
2. Connect the digital pressure calibrator to the test port.
3. Turn the manual relief valve clockwise until the gauge reads exactly 2500 PSI.
4. Tighten the hexagonal locknut to prevent drift.
5. Log the final PSI reading and calibration timestamp in the logs.
"""
    pump_file = KNOWLEDGE_BASE_DIR / "notes" / "pump_manual.md"
    with open(pump_file, "w", encoding="utf-8") as f:
        f.write(pump_content)
    print(f"Created test doc: {pump_file.name}")


def run_tests():
    print("\n--- 🛠️ Setup: Creating Test Documents ---")
    setup_test_documents()

    print("\n--- 📥 Phase 3: Triggering Incremental Indexing ---")
    stats = scan_and_index_knowledge_base()
    print(f"Index stats: {stats}")

    print("\n--- 🔍 Phase 4/9: Testing Search & Retrieval Quality ---")
    queries = [
        "How do you calibrate the hydraulic pump?",
        "What is the token expiration limit?",
        "Tell me about quantum key cryptography in this project" # Completely absent
    ]

    for q in queries:
        print(f"\n👉 Query: '{q}'")
        chunks = retrieve_relevant_context(q)
        print(f"   Retrieved {len(chunks)} chunks.")
        for idx, chunk in enumerate(chunks):
            print(f"   [{idx+1}] File: {chunk['metadata']['file_name']} | Re-rank Score: {chunk.get('rerank_score', 0.0):.4f}")

    print("\n--- 🤖 Phase 7/10: Testing Upgraded Grounded Q&A with Memory ---")
    
    test_cases = [
        {
            "name": "TEST 1: Present Info (JWT Auth)",
            "query": "Explain JWT token-based authentication.",
            "session_id": "test_session_jwt"
        },
        {
            "name": "TEST 2: Conversational Memory Coreference Resolution",
            "query": "How long are they valid?",
            "session_id": "test_session_jwt" # Share same session to test memory!
        },
        {
            "name": "TEST 3: Technical Manual (Pump Calibration)",
            "query": "How do you calibrate the hydraulic pump H-400?",
            "session_id": "test_session_pump"
        },
        {
            "name": "TEST 4: Absent Info (Anti-Hallucination)",
            "query": "How does this system implement quantum key encryption?",
            "session_id": "test_session_quantum"
        }
    ]

    for case in test_cases:
        print(f"\n=========================================")
        print(f"🎬 {case['name']}")
        print(f"❓ Question: {case['query']}")
        print(f"=========================================")
        
        session_id = case.get("session_id")
        refined_query = case["query"]
        
        # Resolve Conversational Memory query rewriting (exactly like the MCP server)
        if session_id:
            from src.llm import ConversationMemory, condense_query_with_history
            history = ConversationMemory.get_history(session_id)
            if history:
                refined_query = condense_query_with_history(case["query"], history)
                print(f"🔄 Memory Coreference Resolved -> Condensed Query: '{refined_query}'")

        # 1. Retrieve matching chunks using refined query
        chunks = retrieve_relevant_context(refined_query)
        
        # 2. Ask local LLM
        print("🤔 Processing via Ollama...")
        answer = generate_grounded_answer(case["query"], chunks, session_id=session_id)
        print(f"\n💬 Answer:\n{answer}\n")


if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\nTest run cancelled.")
