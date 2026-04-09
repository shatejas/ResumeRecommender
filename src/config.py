"""Configuration settings for the RAG pipeline."""

from pathlib import Path

# OpenSearch settings
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_INDEX = "resumes"

# Ollama settings
OLLAMA_MODEL = "llama3.2"  # Change to your preferred model
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ingestion settings
INGEST_BATCH_SIZE = 10  # Number of resume files per batch

# Resume folder (set during ingestion)
def _read_resume_folder():
    path = Path(__file__).parent.parent / ".resume_folder"
    return path.read_text().strip() if path.exists() else "."

RESUME_FOLDER = _read_resume_folder()
