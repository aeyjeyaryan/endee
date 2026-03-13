from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger
from src.utils.config import settings


class Embedder:
    """
    Wraps sentence-transformers to produce float32 embeddings.
    Model is loaded once and cached for the lifetime of the process.
    """

    _instance = None
    _model = None

    def __new__(cls):
        # Singleton — avoid loading the model multiple times
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = settings.embedding_model
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """Embed a single string and return a list of floats."""
        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> List[List[float]]:
        """
        Embed a list of strings in batches.
        Returns a list of float lists — one per input text.
        """
        self._load_model()
        logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})")
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.tolist()