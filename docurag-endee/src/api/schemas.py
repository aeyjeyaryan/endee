from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Ingest 

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int
    time_seconds: float


# Query 

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The natural language question to answer.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve from Endee.")
    include_context: bool = Field(False, description="If true, include retrieved chunks in the response.")


class SourceInfo(BaseModel):
    source: str
    page_number: Optional[int]
    similarity: float
    text_preview: str = Field(..., description="First 150 characters of the chunk.")


class ContextChunk(BaseModel):
    id: str
    similarity: float
    source: str
    page_number: Optional[int]
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    context: Optional[List[ContextChunk]] = None
    retrieval_ms: int
    generation_ms: int


#  Stats 

class StatsResponse(BaseModel):
    index_name: str
    vector_count: Any
    dimension: int
    space_type: str
    precision: str
    documents_indexed: int


# Documents 

class DocumentListResponse(BaseModel):
    documents: List[str]
    total: int


class DeleteResponse(BaseModel):
    status: str
    filename: str