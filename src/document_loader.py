"""Load and chunk resume documents from a folder."""

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, INGEST_BATCH_SIZE


def get_resume_files(folder_path: str) -> list[Path]:
    """Get all supported resume file paths from a folder."""
    folder = Path(folder_path)
    return [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".pdf", ".docx", ".doc"]
    ]


def load_single_resume(file_path: Path) -> list:
    """Load a single resume file, keeping only the source in metadata."""
    if file_path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(file_path)).load()
    else:
        docs = Docx2txtLoader(str(file_path)).load()

    for doc in docs:
        doc.metadata = {"source": str(file_path)}
    return docs


def chunk_documents(documents: list) -> list:
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(documents)
