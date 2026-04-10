# ATS Resume Generator

RAG pipeline that generates ATS-optimized resumes based on job descriptions using your existing resumes as reference. Built with LangChain, Ollama, and OpenSearch — runs entirely locally at zero cost.

## Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Homebrew (macOS)

## Setup

```bash
./setup.sh /path/to/your/resumes/folder
```

This will:
1. Install Ollama (if not present)
2. Pull the mistral:7b and nomic-embed-text models
3. Start OpenSearch 3.5 via Docker
4. Install Python dependencies
5. Extract skills and experience from your resumes
6. Ingest your resumes into the vector store with hybrid search support

## Usage

```bash
# Web UI
streamlit run app.py

# CLI — from a file
python main.py test/sample_jd.txt

# CLI — interactive (paste JD, press Enter twice)
python main.py
```

## How It Works

1. **Search** — Finds the best matching existing resume using hybrid search (semantic kNN + BM25 keyword matching)
2. **Score** — Scores the best match against the job description using ATS scoring
3. **Reuse or Generate** — If the existing resume scores ≥ 90, it's used as-is. Otherwise, a new resume is generated
4. **Refine** — The generated resume is scored and refined up to 2 times using ATS feedback (missing keywords + improvement suggestions)
5. **Export** — The highest-scoring resume is saved as a formatted `.docx` file

### Anti-Hallucination

The pipeline extracts structured skills and experience during ingestion and passes them explicitly to the LLM. The prompts strictly constrain the model to only use data from your actual resumes — no invented degrees, certifications, or metrics.

### Document Formatting

The generated `.docx` uses professional formatting extracted from your existing resumes:
- Times New Roman font throughout
- Centered name and contact info
- Section headers with bottom borders
- Experience lines with right-aligned dates (Company | Title → dates)
- Skills as bullet points with bold categories
- Education with bold degree | university and right-aligned graduation date
- Compact layout with professional formatting

## Project Structure

```
├── setup.sh               # One-command setup (Ollama, OpenSearch, deps, ingestion)
├── docker-compose.yml     # OpenSearch 3.5 container
├── requirements.txt       # Python dependencies
├── ingest.py              # Extract skills/experience and index resumes
├── app.py                 # Streamlit web UI
├── main.py                # CLI entry point
├── test/
│   └── sample_jd.txt     # Sample job description for testing
└── src/
    ├── config.py          # Configuration (models, OpenSearch, chunking)
    ├── document_loader.py # Load and chunk PDF/DOCX files
    ├── vector_store.py    # OpenSearch hybrid search with nested chunks
    ├── rag_chain.py       # LLM prompts, ATS scoring, resume generation
    └── docx_writer.py     # Word document export with professional formatting
```

## Configuration

Edit `src/config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `mistral:7b` | LLM for generation and scoring |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for semantic search |
| `OPENSEARCH_INDEX` | `resumes` | OpenSearch index name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `INGEST_BATCH_SIZE` | `10` | Resumes processed per batch |

## Tech Stack

- **LLM**: Ollama (mistral:7b) — free, local
- **Embeddings**: Ollama (nomic-embed-text) — free, local
- **Vector Store**: OpenSearch 3.5 (Docker) — hybrid search with nested kNN + BM25
- **Framework**: LangChain
- **Output**: python-docx
- **UI**: Streamlit
