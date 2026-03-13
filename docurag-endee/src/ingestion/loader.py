from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from loguru import logger


@dataclass
class DocumentPage:
    """A single page/section of a loaded document."""
    source: str          # original file path
    page_number: int | None
    text: str


def load_document(path: str | Path) -> List[DocumentPage]:
    """
    Load a file and return a list of DocumentPage objects.
    Supports: .pdf, .txt, .md, .markdown
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: pdf, txt, md")


def _load_pdf(path: Path) -> List[DocumentPage]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    pages = []
    doc = fitz.open(str(path))
    logger.info(f"Loading PDF '{path.name}' ({len(doc)} pages)")

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append(DocumentPage(
                source=str(path),
                page_number=i + 1,
                text=text,
            ))

    doc.close()
    return pages


def _load_text(path: Path) -> List[DocumentPage]:
    logger.info(f"Loading text file '{path.name}'")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    # Treat the whole file as a single "page"
    return [DocumentPage(source=str(path), page_number=None, text=text)]


def load_directory(directory: str | Path, recursive: bool = True) -> List[DocumentPage]:
    """Load all supported documents from a directory."""
    directory = Path(directory)
    patterns = ["*.pdf", "*.txt", "*.md", "*.markdown"]
    all_pages: List[DocumentPage] = []

    for pattern in patterns:
        files = list(directory.rglob(pattern) if recursive else directory.glob(pattern))
        for f in sorted(files):
            try:
                pages = load_document(f)
                all_pages.extend(pages)
                logger.info(f"Loaded {len(pages)} page(s) from {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Total pages loaded from '{directory}': {len(all_pages)}")
    return all_pages