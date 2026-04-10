"""Configuration settings for the RAG pipeline."""

from pathlib import Path

# OpenSearch
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_INDEX = "resumes"

# Ollama
OLLAMA_MODEL = "mistral:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ingestion
INGEST_BATCH_SIZE = 10

# Resume folder (saved during ingestion, read at generation time)
_FOLDER_FILE = Path(__file__).parent.parent / ".resume_folder"
RESUME_FOLDER = _FOLDER_FILE.read_text().strip() if _FOLDER_FILE.exists() else "."
