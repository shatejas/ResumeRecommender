"""Ingest resumes into the vector store."""

import sys
from pathlib import Path
from src.config import INGEST_BATCH_SIZE
from src.document_loader import get_resume_files, load_single_resume, chunk_documents
from src.vector_store import ensure_index, index_resume

RESUME_FOLDER_FILE = Path(__file__).parent / ".resume_folder"


def main(resume_folder: str):
    resume_folder = str(Path(resume_folder).resolve())
    RESUME_FOLDER_FILE.write_text(resume_folder)
    files = get_resume_files(resume_folder)
    total = len(files)
    print(f"Found {total} resumes in: {resume_folder}")

    ensure_index()

    for i in range(0, total, INGEST_BATCH_SIZE):
        batch_files = files[i:i + INGEST_BATCH_SIZE]
        batch_num = i // INGEST_BATCH_SIZE + 1
        print(f"\nBatch {batch_num}: processing {len(batch_files)} files...")

        for f in batch_files:
            docs = load_single_resume(f)
            chunks = chunk_documents(docs)
            index_resume(source=str(f), chunks=chunks)
            print(f"  Indexed {f.name} ({len(chunks)} chunks)")

    print("\nDone! All resumes indexed in OpenSearch.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <resume_folder_path>")
        sys.exit(1)
    main(sys.argv[1])
