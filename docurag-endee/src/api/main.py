from __future__ import annotations

import time
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    ContextChunk, DeleteResponse, DocumentListResponse, IngestResponse,
    QueryRequest, QueryResponse, SourceInfo, StatsResponse,
)
from src.generation.llm_client import LLMClient
from src.retrieval.retriever import IngestionService, Retriever
from src.retrieval.vector_store import VectorStore


app = FastAPI(
    title="DocuRAG — Intelligent Document Q&A",
    description=(
        "A Retrieval-Augmented Generation (RAG) API powered by **Endee** vector database. "
        "Upload documents, then ask natural language questions — answers are grounded in your documents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons (initialized lazily on first request)
_ingestion_service: IngestionService | None = None
_retriever: Retriever | None = None
_llm: LLMClient | None = None
_store: VectorStore | None = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


#  Routes 
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "DocuRAG", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(..., description="PDF, TXT, or Markdown file to ingest")):
    """
    Upload a document and index it into Endee.
    Supported formats: PDF, TXT, MD/Markdown.
    """
    allowed = {".pdf", ".txt", ".md", ".markdown"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        svc = get_ingestion_service()
        # Override source name with original filename
        from src.ingestion.loader import load_document
        from src.ingestion.chunker import RecursiveChunker
        from src.ingestion.embedder import Embedder
        import os as _os

        pages = load_document(tmp_path)
        # Patch source name
        for p in pages:
            p.source = file.filename

        chunker = RecursiveChunker()
        embedder = Embedder()
        store = get_store()

        chunks = chunker.chunk_pages(pages)
        texts = [c.text for c in chunks]
        vectors = embedder.embed_batch(texts, show_progress=False)

        items = [
            {
                "id": f"{_os.path.splitext(file.filename)[0]}_p{c.page_number or 1}_c{c.chunk_index}",
                "vector": vec,
                "meta": {
                    "text": c.text,
                    "source": file.filename,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                },
            }
            for c, vec in zip(chunks, vectors)
        ]

        store.upsert(items)

        # Update registry
        svc.registry[file.filename] = [it["id"] for it in items]
        from src.retrieval.retriever import _save_registry
        _save_registry(svc.registry)

        elapsed = round(time.time() - t0, 2)
        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_indexed=len(chunks),
            time_seconds=elapsed,
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """
    Ask a natural language question.
    Endee retrieves the most relevant document chunks, which ground the LLM answer.
    """
    retriever = get_retriever()
    llm = get_llm()

    # 1. Retrieve from Endee
    t0 = time.time()
    chunks = retriever.retrieve(request.question, top_k=request.top_k)
    retrieval_ms = int((time.time() - t0) * 1000)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found. Please ingest documents first.")

    # 2. Generate answer
    t1 = time.time()
    answer = llm.generate_answer(request.question, chunks)
    generation_ms = int((time.time() - t1) * 1000)

    # 3. Build response
    sources = [
        SourceInfo(
            source=c.meta.get("source", "unknown"),
            page_number=c.meta.get("page_number"),
            similarity=round(c.similarity, 4),
            text_preview=c.meta.get("text", "")[:150],
        )
        for c in chunks
    ]

    context = None
    if request.include_context:
        context = [
            ContextChunk(
                id=c.id,
                similarity=round(c.similarity, 4),
                source=c.meta.get("source", "unknown"),
                page_number=c.meta.get("page_number"),
                text=c.meta.get("text", ""),
            )
            for c in chunks
        ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        context=context,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Admin"])
def stats():
    """Return Endee index statistics."""
    store = get_store()
    svc = get_ingestion_service()
    raw = store.stats()
    return StatsResponse(
        **raw,
        documents_indexed=len(svc.list_documents()),
    )


@app.get("/api/v1/documents", response_model=DocumentListResponse, tags=["Admin"])
def list_documents():
    """List all documents that have been ingested."""
    svc = get_ingestion_service()
    docs = svc.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@app.delete("/api/v1/documents/{filename}", response_model=DeleteResponse, tags=["Admin"])
def delete_document(filename: str):
    """Remove a document and all its chunks from the Endee index."""
    svc = get_ingestion_service()
    deleted = svc.delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in registry.")
    return DeleteResponse(status="deleted", filename=filename)