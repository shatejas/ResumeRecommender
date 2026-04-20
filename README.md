# ATS Resume Generator

RAG pipeline that generates ATS-optimized resumes based on job descriptions using your existing resumes as reference. Built with LangChain, Ollama, and OpenSearch — runs entirely locally at zero cost.

## Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Homebrew (macOS)

## Setup

```bash
./setup.sh
```

This will:
1. Install Ollama and Docker (if not present)
2. Pull the mistral:7b and nomic-embed-text models
3. Start OpenSearch 3.5 via Docker
4. Install Python dependencies
5. Launch the Web UI

## Usage

```bash
# Web UI
python app.py

# CLI — ingest resumes first, then generate
python ingest.py /path/to/your/resumes/folder
python main.py test/sample_jd.txt

# CLI — interactive (paste JD, press Enter twice)
python main.py
```

In the Web UI, use the sidebar to upload and ingest resumes, then paste a job description to generate.

### Job Scanner

Automatically scan job portals, find matching listings, and generate tailored resumes:

```bash
# Full pipeline: scan portals → generate resumes
python scan_jobs.py

# Scan only — saves matching jobs to data/scanned_jobs.json
python scan_jobs.py scan

# Generate from previously scanned jobs
python scan_jobs.py generate
```

The scanner:
1. Fetches jobs from tracked companies via ATS APIs (Greenhouse, Lever, Ashby, etc.) and career page scraping
2. Runs web search queries for broader discovery
3. Filters by title keywords, location, and sponsorship requirements
4. Deduplicates against scan history
5. Fetches full job descriptions and validates with LLM (location + sponsorship check)
6. Generates tailored resumes for qualifying jobs, saved to `resumes/<CompanyName>/`

Configure in `portals.yml`:
- **tracked_companies** — direct ATS API or career page scraping per company
- **search_queries** — Google search with site filters (lever, greenhouse, ashby) or direct career site searches
- **title_filter** — positive/negative keyword matching
- **location_filter** — country allowlist + remote toggle
- **needs_sponsorship** — skip jobs that explicitly deny sponsorship

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

## Candidate Profile

Edit `candidate.yml` to set your contact info, education, certifications, and resume section order. These fields are always included exactly as written — the LLM never invents them.

```yaml
sections:
  - summary
  - skills
  - experience
  - projects
  - education and certifications

contact:
  name: Jane Doe
  email: jane@example.com
  phone: (555) 123-4567
  location: San Francisco, CA
  linkedin: linkedin.com/in/janedoe
  github: github.com/janedoe

education:
  - degree: M.S. Computer Science
    university: Stanford University
    year: May 2022

certifications:
  - name: AWS Certified Solutions Architect (2024)
```

## LLM Providers

Switch between providers in the Web UI sidebar or configure in `api_keys.yml`:

| Provider | Model | Cost | Setup |
|----------|-------|------|-------|
| **Ollama** (default) | mistral:7b | Free | Included in setup |
| **Gemini** | gemini-2.5-flash | Free tier | [Get API key](https://aistudio.google.com/apikey) |
| **ChatGPT** | gpt-4o-mini | Pay per use | [Get API key](https://platform.openai.com/api-keys) |
| **Claude** | claude-sonnet-4-20250514 | Pay per use | [Get API key](https://console.anthropic.com/settings/keys) |

For the job scanner, copy `api_keys.template.yml` to `api_keys.yml` and fill in your keys. The scanner auto-selects the best available provider (Gemini > ChatGPT > Claude > Ollama).

## Project Structure

```
├── setup.sh               # One-command setup (Ollama, OpenSearch, deps, ingestion)
├── docker-compose.yml     # OpenSearch 3.5 container
├── requirements.txt       # Python dependencies
├── ingest.py              # Extract skills/experience and index resumes
├── app.py                 # Gradio web UI
├── main.py                # CLI entry point
├── scan_jobs.py           # Job scanner + auto resume generation
├── candidate.yml          # Candidate profile (contact, education, certs, section order)
├── portals.yml            # Job scanner configuration (companies, filters, queries)
├── api_keys.template.yml  # API key template
├── data/
│   ├── scanned_jobs.json  # Last scan results
│   └── scan_history.tsv   # All previously seen jobs (dedup)
├── test/
│   └── sample_jd.txt      # Sample job description for testing
└── src/
    ├── config.py          # Configuration (models, OpenSearch, chunking)
    ├── candidate.py       # Load candidate profile from YAML
    ├── document_loader.py # Load and chunk PDF/DOCX files
    ├── vector_store.py    # OpenSearch hybrid search with nested chunks
    ├── rag_chain.py       # LLM prompts, ATS scoring, resume generation
    ├── resume_model.py    # Structured resume data model
    ├── structured_writer.py # Word doc from structured data (section-ordered)
    ├── docx_writer.py     # Word document export with professional formatting
    └── scanner/
        ├── scanner.py     # Orchestrates scan: fetch → filter → deduplicate
        ├── models.py      # JobListing data model
        ├── ats_clients.py # ATS API clients (Greenhouse, Lever, Ashby, etc.)
        ├── careers_scraper.py # Playwright-based career page scraper
        ├── web_search.py  # Google search + career site search
        ├── filters.py     # Title and location filtering
        └── validator.py   # Job validation utilities
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

- **LLM**: Ollama (mistral:7b) / Gemini / ChatGPT / Claude — switchable
- **Embeddings**: Ollama (nomic-embed-text) — free, local
- **Vector Store**: OpenSearch 3.5 (Docker) — hybrid search with nested kNN + BM25
- **Framework**: LangChain
- **Output**: python-docx
- **UI**: Gradio
- **Job Scanner**: Playwright + ATS APIs (Greenhouse, Lever, Ashby) + Google search
