from __future__ import annotations

from typing import List

from loguru import logger

from src.generation.prompt_builder import build_rag_prompt
from src.retrieval.vector_store import SearchResult
from src.utils.config import settings


class LLMClient:
    """
    Generates grounded answers using an LLM with RAG context.

    Supports:
    - Gemini (gemini-1.5-flash, etc.) — set LLM_PROVIDER=gemini
    - Ollama (llama3, mistral, etc.) — set LLM_PROVIDER=ollama
    """

    def generate_answer(
        self,
        question: str,
        context_chunks: List[SearchResult],
        temperature: float = 0.2,
    ) -> str:
        """
        Build a RAG prompt from the retrieved context and generate an answer.
        """
        messages = build_rag_prompt(question, context_chunks)
        provider = settings.llm_provider.lower()

        if provider == "gemini":
            return self._call_gemini(messages, temperature)
        elif provider == "ollama":
            return self._call_ollama(messages, temperature)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Set LLM_PROVIDER to 'gemini' or 'ollama'.")

    #  Gemini model                                                             
    

    def _call_gemini(self, messages: List[dict], temperature: float) -> str:
        import google.generativeai as genai

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")

        genai.configure(api_key=settings.gemini_api_key)

        # Extract system prompt and user content
        system_instruction = None
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]

        logger.debug(f"Calling Gemini model: {settings.gemini_model}")
        
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=system_instruction
        )
        
        response = model.generate_content(
            user_content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            )
        )
        return response.text.strip()

    # ------------------------------------------------------------------ #
    #  Ollama for initial testing
    # ------------------------------------------------------------------ #

    def _call_ollama(self, messages: List[dict], temperature: float) -> str:
        import httpx

        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        logger.debug(f"Calling Ollama model: {settings.ollama_model} at {url}")

        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()