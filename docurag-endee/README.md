# DocuRAG — Intelligent Document Q&A with Endee Vector Database

## Project Overview & Problem Statement

### The Problem

Organizations accumulate thousands of documents — PDFs, reports, wikis, manuals — but finding precise answers buried inside them remains painfully slow. Traditional keyword search misses semantically related content, and LLMs hallucinate when they lack grounding context.

### The Solution

**DocuRAG** solves this by combining:
1. **Semantic chunking** — documents are split into meaningful passages and converted to dense vector embeddings
2. **High-performance vector search** — Endee stores and retrieves the most relevant chunks at sub-millisecond speed
3. **LLM-grounded generation** — retrieved context is injected into an LLM prompt, producing accurate, cited answers

This is the classic RAG (Retrieval-Augmented Generation) pattern, but built with Endee as the backbone for vector storage and ANN search.

## Features

| Feature | Description |
|---|---|
| 📄 **Multi-format ingestion** | Supports PDF, TXT, Markdown, and plain text files |
| 🔍 **Semantic search** | Finds relevant passages by meaning, not just keywords |
| 🤖 **Grounded Q&A** | Answers backed by real document passages with source citations |
| ⚡ **High-speed retrieval** | Endee HNSW index provides sub-millisecond ANN search |
| 🌐 **REST API** | Full FastAPI service with interactive Swagger docs |
| 🖥️ **CLI interface** | Command-line tool for ingestion and querying |
| 🐳 **Docker ready** | One-command deployment for the full stack |
| 🧪 **Test suite** | Unit and integration tests included |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DocuRAG System                            │
│                                                                  │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │  Documents  │────▶│   Chunker    │────▶│ Embedding Model  │  │
│  │ PDF/TXT/MD  │     │ (Recursive   │     │ (sentence-       │  │
│  └─────────────┘     │  text split) │     │  transformers)   │  │
│                       └──────────────┘     └────────┬─────────┘  │
│                                                     │ float32[]   │
│                                            ┌────────▼─────────┐  │
│                                            │   Endee Vector   │  │
│                                            │   Database       │  │
│                                            │  ┌────────────┐  │  │
│                                            │  │ HNSW Index │  │  │
│                                            │  │ (cosine)   │  │  │
│                                            │  └────────────┘  │  │
│                                            └────────┬─────────┘  │
│                                                     │             │
│  │  Final       │◀───│  LLM         │◀────│  Top-K Chunks    │  │
│  │  Answer      │    │  (Gemini /    │     │  Retriever       │  │
│  │  + Citations │    │   Ollama)    │     └──────────────────┘  │
│  └──────────────┘    └──────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```

### Technical Approach

1. **Document Ingestion Pipeline**
   - Documents are loaded and parsed (PyMuPDF for PDFs, native for text)
   - Text is recursively chunked with a configurable size (default: 512 tokens) and overlap (64 tokens)
   - Each chunk is embedded using `all-MiniLM-L6-v2` (384-dim, fast, high quality)
   - Vectors + metadata (source file, page number, chunk index) are upserted into Endee

2. **Query Pipeline**
   - User query is embedded with the same model
   - Endee performs ANN search (HNSW, cosine similarity) returning top-K chunks
   - Retrieved chunks are formatted into a prompt context window
   - An LLM generates a final, cited answer

3. **Endee Configuration**
   - **Index**: `docurag_index`
   - **Dimension**: 384 (matching `all-MiniLM-L6-v2`)
   - **Space type**: `cosine`
   - **Precision**: `INT8` (for memory efficiency)
   - **HNSW M**: 16, **ef_construction**: 128

## Project Structure

```
docurag-endee/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # Document loading (PDF, TXT, MD)
│   │   ├── chunker.py         # Recursive text chunking
│   │   └── embedder.py        # Sentence-transformer embeddings
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # Endee client wrapper
│   │   └── retriever.py       # Top-K retrieval logic
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py  # RAG prompt construction
│   │   └── llm_client.py      # Gemini / Ollama integration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app
│   │   └── schemas.py         # Pydantic request/response models
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration management
├── data/
│   └── sample_docs/           # Sample documents for demo
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_api.py
├── scripts/
│   ├── ingest.py              # CLI ingestion script
│   └── query.py               # CLI query script
├── docs/
│   └── architecture.md
├── docker-compose.yml         # Full stack: Endee + DocuRAG API
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- A Google Gemini API key **OR** [Ollama](https://ollama.ai) running locally 

---



###   Development Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/docurag-endee.git
cd docurag-endee

# 2. Start only Endee via Docker
docker compose up endee -d

# 3. Create Python virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env .env
# Edit .env with your settings
## ENV VARS:
# Endee Settings
ENDEE_BASE_URL=http://localhost:8080/api/v1
ENDEE_AUTH_TOKEN=
ENDEE_INDEX_NAME=docurag_index

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSyA_CbIDn1Irv1lEpm5WaMz8vdSP-SwJTYA
GEMINI_MODEL=gemini-2.5-flash

# (Ollama alternative — no API key needed)
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3

# Retrieval
TOP_K=5

# 6. Ingest documents
(ingests priorly)
python scripts/ingest.py --dir data/sample_docs

# 7. Start the API server
uvicorn src.api.main:app --reload --port 8000

# 8. Or use the CLI directly
python scripts/query.py "What is the PDF about?"
```


##  Usage

## Access Using Frontend

# 1. Run the backend
uvicorn src.api.main:app --reload --port 8000

# 2. Run the index.html file



### CLI — Query

```bash
# Simple question
python scripts/query.py "What is a the PDF about?"

# Show top-K retrieved chunks
python scripts/query.py "Explain the heading" --show-context --top-k 3
```

### REST API

```bash
# Ingest a document via API
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@myfile.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key components of a RAG system?",
    "top_k": 5
  }'

# Get index stats
curl http://localhost:8000/api/v1/stats

# List ingested documents
curl http://localhost:8000/api/v1/documents

# Delete a document from the index
curl -X DELETE http://localhost:8000/api/v1/documents/myfile.pdf
```


## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

The Endee vector database is Apache-2.0 licensed. See [endee-io/endee](https://github.com/endee-io/endee).