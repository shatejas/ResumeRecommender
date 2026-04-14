"""Load candidate profile from YAML."""

import yaml
from pathlib import Path

CANDIDATE_FILE = Path(__file__).parent.parent / "candidate.yml"


def load_candidate(yaml_text: str = "") -> dict:
    """Load candidate profile from YAML text or the default file."""
    try:
        if yaml_text and yaml_text.strip():
            data = yaml.safe_load(yaml_text) or {}
        elif CANDIDATE_FILE.exists():
            data = yaml.safe_load(CANDIDATE_FILE.read_text()) or {}
        else:
            return {}
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid candidate.yml: {e}")


def format_contact(data: dict) -> str:
    """Format contact info as a single header line."""
    contact = data.get("contact", {})
    if not contact:
        return ""
    name = contact.get("name", "")
    parts = [v for k, v in contact.items() if k != "name" and v]
    line = " | ".join(parts)
    return f"{name}\n{line}" if name else line


def format_education(data: dict) -> str:
    """Format education entries."""
    entries = data.get("education", [])
    if not entries:
        return ""
    lines = []
    for e in entries:
        degree = e.get("degree", "")
        uni = e.get("university", "")
        year = e.get("year", "")
        lines.append(f"{degree} — {uni}, {year}")
    return "\n".join(lines)


def format_certifications(data: dict) -> str:
    """Format certification entries."""
    entries = data.get("certifications", [])
    if not entries:
        return ""
    return "\n".join(f"{c.get('name', '')} ({c.get('year', '')})" for c in entries)


def get_sections(data: dict) -> list[str]:
    """Get the resume section order from candidate config."""
    from src.resume_model import DEFAULT_SECTIONS
    sections = data.get("sections", None)
    if sections and isinstance(sections, list):
        return [s.lower().strip() for s in sections]
    return DEFAULT_SECTIONS
