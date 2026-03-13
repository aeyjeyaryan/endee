from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .loader import DocumentPage
from src.utils.config import settings


@dataclass
class TextChunk:
    """A chunk of text ready to be embedded and stored."""
    id: str              # e.g. "report.pdf_p3_c0"
    text: str
    source: str          
    page_number: int | None
    chunk_index: int


class RecursiveChunker:
    """
    Splits text using a hierarchy of separators (paragraph → sentence → word)
    to stay under the target chunk size while preserving semantic boundaries.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def chunk_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        for page in pages:
            page_chunks = self._split(page.text)
            source_name = _basename(page.source)
            page_label = f"p{page.page_number}" if page.page_number else "p1"

            for i, chunk_text in enumerate(page_chunks):
                chunk_id = f"{source_name}_{page_label}_c{i}"
                chunks.append(TextChunk(
                    id=chunk_id,
                    text=chunk_text.strip(),
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=i,
                ))
        return [c for c in chunks if len(c.text) > 20]  # skip trivially short chunks

    def _split(self, text: str) -> List[str]:
        return self._recursive_split(text, self.SEPARATORS)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # Character-level split as last resort
            return self._fixed_split(text)

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Apply overlap: carry forward tail of current chunk
                    overlap_text = current[-self.overlap:] if self.overlap else ""
                    current = (overlap_text + sep + part).strip() if overlap_text else part.strip()
                else:
                    # Single part exceeds chunk size — recurse with next separator
                    sub_chunks = self._recursive_split(part, remaining_seps)
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""

        if current:
            chunks.append(current)

        return chunks or [text]

    def _fixed_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks


def _basename(path: str) -> str:
    """Return just the filename without extension for use in IDs."""
    import os
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    # sanitize for use in Endee IDs
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:40]