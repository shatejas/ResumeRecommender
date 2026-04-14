"""Ingest resumes into the vector store."""

import re
import sys
from pathlib import Path
from src.config import INGEST_BATCH_SIZE
from src.document_loader import get_resume_files, load_single_resume, chunk_documents
from src.vector_store import ensure_index, index_resume

RESUME_FOLDER_FILE = Path(__file__).parent / ".resume_folder"

SECTION_RE = re.compile(
    r"^\s*(SUMMARY|SKILLS|TECHNICAL\s+SKILLS|EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|"
    r"WORK\s+EXPERIENCE|EDUCATION|CERTIFICATIONS|PROJECTS|ACADEMIC\s+PROJECTS|"
    r"KEY\s+PROJECTS|ACHIEVEMENTS|PUBLICATIONS|AWARDS|VOLUNTEER)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Map header variations to canonical names
_HEADER_MAP = {
    "skills": "skills", "technical skills": "skills",
    "experience": "experience", "professional experience": "experience", "work experience": "experience",
    "education": "education",
    "certifications": "certifications",
    "projects": "projects", "academic projects": "projects", "key projects": "projects",
    "summary": "summary",
    "achievements": "achievements",
    "publications": "publications",
    "awards": "awards",
    "volunteer": "volunteer",
}


def _parse_sections(resume_text: str) -> dict[str, str]:
    """Parse resume text into canonical named sections."""
    lines = [line.strip() for line in resume_text.splitlines()]
    normalized = "\n".join(lines)
    sections = {}
    matches = list(SECTION_RE.finditer(normalized))
    for i, match in enumerate(matches):
        raw_header = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        canonical = _HEADER_MAP.get(raw_header, raw_header)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        content = normalized[start:end].strip()
        if content:
            # Append if same canonical section appears multiple times
            if canonical in sections:
                sections[canonical] += "\n" + content
            else:
                sections[canonical] = content
    return sections


def extract_skills_experience(resume_text: str) -> tuple[str, str]:
    """Extract skills and experience sections."""
    sections = _parse_sections(resume_text)
    return sections.get("skills", ""), sections.get("experience", "")


def extract_all_sections(resume_text: str) -> tuple[str, str, str, str, str]:
    """Extract skills, experience, education, certifications, and projects."""
    sections = _parse_sections(resume_text)
    return (
        sections.get("skills", ""),
        sections.get("experience", ""),
        sections.get("education", ""),
        sections.get("certifications", ""),
        sections.get("projects", ""),
    )


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
            if not docs:
                continue
            chunks = chunk_documents(docs)
            full_text = "\n".join(d.page_content for d in docs)
            skills, experience, education, certifications, projects = extract_all_sections(full_text)

            index_resume(source=str(f), skills=skills, experience=experience,
                         education=education, certifications=certifications,
                         projects=projects, chunks=chunks)
            print(f"  Indexed {f.name} ({len(chunks)} chunks)")

    print("\nDone! All resumes indexed in OpenSearch.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <resume_folder_path>")
        sys.exit(1)
    main(sys.argv[1])
