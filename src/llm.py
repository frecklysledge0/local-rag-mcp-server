import re
import logging
import ollama
from typing import List, Dict, Any, Optional

from src.config import OLLAMA_MODEL_NAME, LOGS_DIR, ensure_directories

# Configure logging
ensure_directories()
LOG_FILE = LOGS_DIR / "llm.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Global in-memory storage for sliding conversation history
# session_id -> list of {"query": str, "answer": str}
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MEMORY_WINDOW = 3

class ConversationMemory:
    """Manages sliding conversation history memory."""
    @staticmethod
    def add_turn(session_id: str, query: str, answer: str):
        if not session_id:
            return
        if session_id not in CONVERSATION_HISTORY:
            CONVERSATION_HISTORY[session_id] = []
        
        CONVERSATION_HISTORY[session_id].append({"query": query, "answer": answer})
        # Keep only the last N turns
        if len(CONVERSATION_HISTORY[session_id]) > MEMORY_WINDOW:
            CONVERSATION_HISTORY[session_id].pop(0)

    @staticmethod
    def get_history(session_id: str) -> List[Dict[str, str]]:
        if not session_id:
            return []
        return CONVERSATION_HISTORY.get(session_id, [])

    @staticmethod
    def clear_session(session_id: str):
        if session_id in CONVERSATION_HISTORY:
            del CONVERSATION_HISTORY[session_id]


def condense_query_with_history(query: str, history: List[Dict[str, str]]) -> str:
    """
    Coreference Resolution: Rewrites follow-up queries incorporating prior context.
    For example: 'How long are they valid?' -> 'How long are JWT refresh tokens valid?'
    """
    if not history:
        return query

    # Construct conversation context block
    history_str = ""
    for idx, turn in enumerate(history):
        history_str += f"User: {turn['query']}\n"
        # We strip citations out of the assistant text to keep the rewriting context clean
        clean_ans = re.sub(r"\[Source:[^\]]+\]", "", turn["answer"])
        history_str += f"Assistant: {clean_ans}\n"

    prompt = f"""You are a search query refinement engine.
Given the conversation history and a new follow-up question, rewrite the follow-up question to be self-contained and search-optimized by resolving pronouns (such as "it", "they", "she", "this", "that") to their original subjects mentioned in history.

Absolute Rules:
1. Output ONLY the refined, self-contained question. Do not add introductions, explanations, quotes, or markdown wrappers.
2. If the question does not contain pronouns or references that require history, output the original question exactly.

Conversation History:
{history_str}
Follow-up Question: {query}

Self-contained Search Query:"""

    try:
        logging.info(f"Query Condensation Input: raw_query='{query}', history_turns={len(history)}")
        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 0.1}
        )
        condensed_query = response.get("response", "").strip()
        # Clean any trailing/leading quotes the model might have returned
        condensed_query = re.sub(r'^["\']|["\']$', '', condensed_query)
        
        logging.info(f"Query Condensation Output: condensed_query='{condensed_query}'")
        return condensed_query if condensed_query else query
    except Exception as e:
        logging.error(f"Query condensation failed: {e}")
        return query


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Builds a highly constrained, strict citation-centric prompt."""
    context_str = ""
    for idx, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        filename = meta.get("file_name", "Unknown File")
        category = meta.get("category", "General")
        context_str += f"--- CONTEXT BLOCK #{idx+1} [Source: {filename} | Category: {category}] ---\n"
        context_str += f"{chunk['text']}\n\n"

    prompt = f"""You are a highly precise, local knowledge base assistant.
Your absolute rule is: Answer the question using ONLY the context blocks provided below.
If the provided context does not contain the answer, state: "Information not found in the local knowledge base." Do NOT attempt to fabricate, extrapolate, or guess an answer.

Follow these formatting instructions:
1. Ground your claims: Every fact in your answer MUST correspond to one of the provided context blocks.
2. Add inline citations: At the end of every sentence or statement that references a fact, write the source in square brackets, e.g., "[Source: filename.md]" using the exact file name shown in the context block header.
3. Be direct, factual, and professional.

Context:
{context_str}

User Question:
{query}

Answer:"""
    return prompt


def validate_grounding_citations(answer: str, allowed_sources: List[str]) -> bool:
    """
    V2.0 Programmatic Grounding Validator.
    Ensures:
    1. The answer contains at least one citation.
    2. All cited sources exist inside the actually retrieved chunks.
    """
    # Extract citations matching [Source: filename.ext]
    citations = re.findall(r"\[Source:\s*([^\]]+)\]", answer)
    if not citations:
        logging.warning("Grounded validation failed: LLM output contained zero citations.")
        return False

    # Check for hallucinated sources
    for source in citations:
        source_clean = source.strip()
        # Verify source matches one of the chunks provided
        if source_clean not in allowed_sources:
            logging.warning(f"Grounded validation failed: LLM cited a hallucinated source '{source_clean}' not in context list {allowed_sources}.")
            return False

    return True


def generate_grounded_answer(
    query: str, 
    chunks: List[Dict[str, Any]], 
    session_id: Optional[str] = None
) -> str:
    """
    Synthesizes and programmatically validates a grounded response using local LLM.
    Enforces dynamic thresholds, dynamic context, and hard programmatic citation guards.
    """
    # V2.0 Hard Empty Check Rule (also triggers if candidate scores fell below dynamic threshold)
    if not chunks:
        logging.info("Retrieval context was empty. Returning hard grounded rejection.")
        return "⚠️ **Information not found in the local knowledge base.**\n\nNo relevant documents met the similarity threshold to answer this query."

    prompt = build_prompt(query, chunks)
    allowed_sources = [c["metadata"].get("file_name", "").strip() for c in chunks if "metadata" in c]
    
    try:
        logging.info(f"LLM Generation Input: model='{OLLAMA_MODEL_NAME}', chunks={len(chunks)}")
        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.0,  # Strict reasoning focus
                "top_p": 0.1
            }
        )
        
        raw_answer = response.get("response", "").strip()
        logging.info(f"LLM Raw Output Length: {len(raw_answer)}")

        # Check for immediate rejection messages
        rejection_phrases = ["information not found", "not found in the local", "information is unavailable"]
        is_rejection = any(phrase in raw_answer.lower() for phrase in rejection_phrases)

        # V2.0 Programmatic Citation Verification Guard
        if not is_rejection:
            is_valid = validate_grounding_citations(raw_answer, allowed_sources)
            if not is_valid:
                # Trigger programmatic fallback to prevent ungrounded responses
                fallback_msg = "⚠️ **Programmatic Grounding Violation**\n\nThe local model generated a response but failed to provide verifiable source citations matching the retrieved documents. Fallback triggered to prevent potential hallucination."
                logging.error("Programmatic Grounding Violation! Rejection triggered.")
                return fallback_msg

        # Format sources appendix
        citations_appendix = "\n\n---\n### 📚 Sources & References\n"
        seen_sources = set()
        
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            fname = meta.get("file_name", "Unknown")
            path = meta.get("source", "")
            cat = meta.get("category", "docs")
            rerank_score = chunk.get("rerank_score", 0.0)
            
            file_link = f"[{fname}](file://{path})"
            
            source_key = (fname, path)
            if source_key not in seen_sources:
                citations_appendix += f"- **{fname}** - Confidence: `{rerank_score:.2f}` | Category: `{cat}` | File: {file_link}\n"
                seen_sources.add(source_key)

        final_response = f"{raw_answer}{citations_appendix}"

        # If it is a valid (grounded or clean rejection) answer, save to sliding conversation history
        if session_id and not is_rejection:
            ConversationMemory.add_turn(session_id, query, raw_answer)

        return final_response
        
    except Exception as e:
        logging.error(f"LLM Generation failed: {e}")
        error_msg = f"❌ **Error communicating with local LLM via Ollama.**\n\nDetails: `{str(e)}`\n\n"
        error_msg += "Please verify that the Ollama service is running (`ollama serve`):\n"
        return error_msg
