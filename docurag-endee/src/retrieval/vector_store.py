from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from src.utils.config import settings
from endee import Precision


@dataclass
class SearchResult:
    id: str
    similarity: float
    meta: Dict[str, Any]


class VectorStore:


    INDEX_DIMENSION = 384   # must match the embedding model output
    HNSW_M = 16             # controls graph connectivity
    HNSW_EF_CONSTRUCTION = 128

    def __init__(self):
        self._client = None
        self._index = None


    def _get_client(self):
        if self._client is None:
            from endee import Endee
            token = settings.endee_auth_token or None
            self._client = Endee(token)
            self._client.set_base_url(settings.endee_base_url)
            logger.info(f"Connected to Endee at {settings.endee_base_url}")
        return self._client

    def _get_index(self):
        if self._index is None:
            client = self._get_client()
            self._ensure_index_exists(client)
            self._index = client.get_index(settings.endee_index_name)
        return self._index

    def _ensure_index_exists(self, client):
        try:
            client.create_index(
                name=settings.endee_index_name,
                dimension=self.INDEX_DIMENSION,
                space_type="cosine",
                precision=Precision.INT8,
            )
            logger.info("Index created")
        except Exception as e:
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                logger.debug(f"Index{settings.endee_index_name}already exists")
            else:
                raise


    def upsert(self, items: List[Dict[str, Any]]) -> int:
        """
        Insert or update vectors in Endee.

        Each item must have:
          - id:     str   — unique chunk identifier
          - vector: list  — float32 embedding
          - meta:   dict  — arbitrary metadata stored alongside the vector
        """
        index = self._get_index()
        index.upsert(items)
        logger.debug(f"Upserted {len(items)} vectors.")
        return len(items)

    def search(self, query_vector: List[float], top_k: int | None = None) -> List[SearchResult]:
        k = top_k or settings.top_k
        index = self._get_index()
        raw = index.query(vector=query_vector, top_k=k)
        results = []
        for r in raw:
            if isinstance(r, dict):
                results.append(SearchResult(id=r["id"], similarity=r["similarity"], meta=r.get("meta") or {}))
            else:
                results.append(SearchResult(id=r.id, similarity=r.similarity, meta=r.meta or {}))
        return results

    def delete_by_source(self, source_filename: str) -> int:
        """
        Delete all chunks that were ingested from a particular source file.
        We do this by listing all vectors and deleting matching IDs.
        """
        index = self._get_index()
        # Endee supports delete by id; we search with a broad query to find matching IDs.
        logger.info(f"Deleting chunks for source: {source_filename}")
        # We keep a local registry (see IngestionService) to track IDs per file.
        raise NotImplementedError("Use IngestionService.delete_document() which tracks IDs.")

    def delete_ids(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        index = self._get_index()
        index.delete(ids)
        logger.info(f"Deleted {len(ids)} vectors from Endee.")

    def stats(self) -> Dict[str, Any]:
        """Return index statistics from Endee."""
        client = self._get_client()
        idx_info = client.get_index(settings.endee_index_name)
        return {
            "index_name": settings.endee_index_name,
            "vector_count": getattr(idx_info, "count", "n/a"),
            "dimension": self.INDEX_DIMENSION,
            "space_type": "cosine",
            "precision": "INT8",
        }

    def reset(self) -> None:
        """Delete and recreate the index (clears all data)."""
        client = self._get_client()
        self._index = None
        try:
            client.delete_index(settings.endee_index_name)
            logger.warning(f"Deleted index '{settings.endee_index_name}'")
        except Exception as e:
            logger.warning(f"Could not delete index: {e}")
        self._ensure_index_exists(client)