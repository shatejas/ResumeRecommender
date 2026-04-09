# ATS Resume Generator

RAG pipeline that generates ATS-optimized resumes based on job descriptions using your existing resumes as reference.

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Homebrew (macOS)

## Setup

```bash
./setup.sh /path/to/your/resumes/folder
```

This will:
1. Install Ollama and pango (if not present)
2. Pull the llama3.2 model
3. Start OpenSearch via Docker
4. Install Python dependencies
5. Ingest your resumes into the vector store

## Usage

```bash
# From a file
python main.py test/sample_jd.txt

# Interactive (paste JD, press Enter twice)
python main.py
```

The tool will:
- Generate an ATS-optimized resume
- Score it against the job description
- Automatically refine up to 3 times if the ATS score is below 90
- Save the final resume as `generated_resume.pdf`

## Project Structure

```
├── setup.sh             # One-command setup
├── docker-compose.yml   # OpenSearch container
├── requirements.txt     # Python dependencies
├── ingest.py            # Index resumes into vector store
├── main.py              # Generate resumes
├── test/
│   └── sample_jd.txt   # Sample job description for testing
└── src/
    ├── config.py        # Configuration settings
    ├── document_loader.py   # Load PDF/DOCX files
    ├── vector_store.py      # OpenSearch hybrid search
    ├── rag_chain.py         # LangChain RAG pipeline + ATS scoring
    └── pdf_writer.py        # PDF export
```

## Configuration

Edit `src/config.py` to change:
- Ollama model (default: llama3.2)
- OpenSearch settings
- Chunk size for document splitting
