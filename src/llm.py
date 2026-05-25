import ollama
from typing import List, Dict, Any
from src.config import OLLAMA_MODEL_NAME

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Builds a strict, citation-driven prompt for the local LLM."""
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


def generate_grounded_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Interfaces with the local Ollama LLM to synthesize a grounded answer.
    Enforces immediate fallback if no context was retrieved.
    """
    # Phase 7: Immediate hard fallback if no context was retrieved
    if not chunks:
        return "⚠️ **Information not found in the local knowledge base.**\n\nNo relevant documents met the similarity threshold to answer this query."

    prompt = build_prompt(query, chunks)
    
    try:
        # Call local Ollama client
        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.0, # Zero temperature is critical to reduce hallucinations!
                "top_p": 0.1
            }
        )
        
        raw_answer = response.get("response", "").strip()
        
        # Format the final output with a gorgeous markdown references section
        citations_appendix = "\n\n---\n### 📚 Sources & References\n"
        seen_sources = set()
        
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            fname = meta.get("file_name", "Unknown")
            path = meta.get("source", "")
            cat = meta.get("category", "docs")
            sim = chunk.get("similarity", 0.0)
            
            # Format click-able link using standard path format (visible in editor/UI)
            file_link = f"[{fname}](file://{path})"
            
            source_key = (fname, path)
            if source_key not in seen_sources:
                citations_appendix += f"- **{fname}** - Match: `{sim:.1%}` | Category: `{cat}` | File: {file_link}\n"
                seen_sources.add(source_key)
                
        return f"{raw_answer}{citations_appendix}"
        
    except Exception as e:
        error_msg = f"❌ **Error communicating with local LLM via Ollama.**\n\nDetails: `{str(e)}`\n\n"
        error_msg += "Please verify that the Ollama service is running (`ollama serve`) and that the model is downloaded:\n"
        error_msg += f"```bash\nollama pull {OLLAMA_MODEL_NAME}\n```"
        return error_msg


if __name__ == "__main__":
    test_chunks = [
        {
            "text": "The project uses JWT authentication. Refresh tokens are valid for 7 days, and access tokens expire in 15 minutes. To refresh a token, POST to /api/auth/refresh with the refresh_token in the body.",
            "metadata": {
                "file_name": "auth.md",
                "source": "/Users/rohityarabati/Desktop/rAG/data/knowledge_base/docs/auth.md",
                "category": "docs"
            },
            "similarity": 0.89
        }
    ]
    print("Testing generate_grounded_answer...")
    ans = generate_grounded_answer("How does token refresh work?", test_chunks)
    print(ans)
