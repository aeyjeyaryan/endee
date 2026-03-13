from __future__ import annotations

from typing import List

from src.retrieval.vector_store import SearchResult


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based solely on the provided context.

Rules:
- Answer ONLY using information found in the context passages below.
- If the answer is not in the context, say: "I don't have enough information in the provided documents to answer this question."
- Always cite the source document(s) you used (e.g., "According to [filename]...").
- Be concise and accurate.
- Do not make up information."""


def build_rag_prompt(question: str, context_chunks: List[SearchResult]) -> List[dict]:
    """
    Build the messages list for LLM chat completion.

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    context_text = _format_context(context_chunks)

    user_content = f"""Context documents:
{context_text}

---

Question: {question}

Please provide a thorough answer based on the context above."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _format_context(chunks: List[SearchResult]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.meta.get("source", "unknown")
        page = chunk.meta.get("page_number")
        page_str = f", page {page}" if page else ""
        text = chunk.meta.get("text", "")
        lines.append(f"[{i}] Source: {source}{page_str}\n{text}\n")
    return "\n".join(lines)