# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

```bash
uv sync                    # Install dependencies
uv run main.py query       # Start interactive query REPL
uv run main.py load        # Index PDFs from PDF_DIRECTORY into Pinecone
uv run main.py wipe        # Delete all vectors from the Pinecone index (prompts for confirmation)
```

## Environment

Requires a `.env` file (see `env.example`):
- `PINECONE_API_KEY` — Pinecone serverless API key
- `OPENAI_API_KEY` — OpenAI API key (used for embeddings)
- `PDF_DIRECTORY` — Path to directory containing PDFs to index

## Architecture

This is a PDF semantic search system (RAG pipeline) with two runtime modes:

**Load mode** (`--load`): PDFs → chunked text → OpenAI embeddings → Pinecone index

**Query mode** (default): user query → embedding → Pinecone vector search → ranked results

### Component responsibilities

| File | Role |
|------|------|
| `main.py` | Entry point; OpenAI client setup, `embed_texts()` for batching, CLI dispatch |
| `pdf_loader.py` | `PDFLoader` — LangChain PyPDFLoader with a 2-page sliding window for cross-page semantic chunking |
| `data_loader.py` | `DataLoader` — thin wrapper; configures chunk_size/overlap/subject, yields `ChunkRecord`s |
| `pinecone_db.py` | `PineconeDB` — auto-creates serverless index, batch upserts, skip-if-exists dedup, vector queries |
| `retriever.py` | `Retriever` — embeds a query string, calls Pinecone, returns ranked results with metadata |

### Key design points

- **Cross-page chunking**: `PDFLoader` processes PDFs in 2-page windows with carry-over, so chunks can span page boundaries without loading the whole document.
- **Iterator-first**: `data_loader.iter_records()` is a generator; nothing is fully materialized unless explicitly requested.
- **Embedding model**: `text-embedding-3-small` (1536 dims). Dimension is validated before upsert.
- **Batch size**: Pinecone upserts default to 50 records per batch (`pinecone_db.py`).
- **Deduplication**: `index_records(skip_if_exists=True)` checks existing vector IDs before embedding to avoid redundant API calls.
