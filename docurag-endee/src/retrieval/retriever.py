from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger

from src.ingestion.chunker import RecursiveChunker, TextChunk
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_directory, load_document
from src.retrieval.vector_store import SearchResult, VectorStore
from src.utils.config import settings

# Simple file-based registry to remember which chunk IDs belong to each source
_REGISTRY_PATH = Path(".docurag_registry.json")


def _load_registry() -> Dict[str, List[str]]:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text())
    return {}


def _save_registry(reg: Dict[str, List[str]]) -> None:
    _REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


class IngestionService:
    """
    Orchestrates the full document → chunk → embed → Endee pipeline.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.chunker = RecursiveChunker()
        self.store = VectorStore()
        self.registry = _load_registry()  # source_file → [chunk_ids]

    def ingest_file(self, path: str) -> int:
        """Load, chunk, embed, and upsert a single file. Returns number of chunks stored."""
        pages = load_document(path)
        return self._process_pages(pages, source_key=os.path.basename(path))

    def ingest_directory(self, directory: str) -> int:
        """Ingest all supported files in a directory."""
        pages = load_directory(directory)
        # Group pages by source so registry is per-file
        from collections import defaultdict
        by_source: Dict[str, list] = defaultdict(list)
        for page in pages:
            by_source[os.path.basename(page.source)].append(page)

        total = 0
        for source_key, source_pages in by_source.items():
            total += self._process_pages(source_pages, source_key)
        return total

    def _process_pages(self, pages, source_key: str) -> int:
        chunks: List[TextChunk] = self.chunker.chunk_pages(pages)
        if not chunks:
            logger.warning(f"No chunks produced for {source_key}")
            return 0

        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_batch(texts)

        items = [
            {
                "id": chunk.id,
                "vector": vec,
                "meta": {
                    "text": chunk.text,
                    "source": os.path.basename(chunk.source),
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
            }
            for chunk, vec in zip(chunks, vectors)
        ]

        self.store.upsert(items)

        # Update registry
        self.registry[source_key] = [c.id for c in chunks]
        _save_registry(self.registry)

        logger.info(f"Ingested {len(chunks)} chunks for '{source_key}'")
        return len(chunks)

    def delete_document(self, source_filename: str) -> bool:
        ids = self.registry.get(source_filename)
        if not ids:
            return False
        self.store.delete_ids(ids)
        del self.registry[source_filename]
        _save_registry(self.registry)
        logger.info(f"Deleted {len(ids)} chunks for '{source_filename}'")
        return True

    def list_documents(self) -> List[str]:
        return list(self.registry.keys())


class Retriever:
    """
    Embeds a user query and retrieves the most relevant chunks from Endee.
    """

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    def retrieve(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """
        Embed the query and perform ANN search against the Endee index.
        Returns top-K SearchResult objects sorted by similarity descending.
        """
        k = top_k or settings.top_k
        logger.debug(f"Retrieving top-{k} chunks for query: '{query[:80]}'")
        query_vec = self.embedder.embed(query)
        results = self.store.search(query_vec, top_k=k)
        logger.debug(f"Retrieved {len(results)} chunks. Best score: {results[0].similarity:.4f}" if results else "No results.")
        return results