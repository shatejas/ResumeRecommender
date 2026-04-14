"""Prompt templates loaded from text files with shared rules."""

from pathlib import Path
from langchain.prompts import PromptTemplate

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_RULES = (_PROMPTS_DIR / "rules.txt").read_text()


def _load(filename: str, input_variables: list[str]) -> PromptTemplate:
    template = (_PROMPTS_DIR / filename).read_text().replace("{rules}", _RULES)
    return PromptTemplate(input_variables=input_variables, template=template)


RESUME_PROMPT = _load("resume.txt", ["contact", "skills", "experience", "education", "certifications", "context", "question"])
RESUME_STRUCTURED_PROMPT = _load("resume_structured.txt", [
    "contact", "skills", "experience", "education", "certifications",
    "context", "question", "section_list", "json_schema",
])
ATS_SCORE_PROMPT = _load("ats_score.txt", ["resume", "job_description"])
REFINE_PROMPT = _load("refine.txt", ["resume", "job_description", "missing_keywords", "improvements", "skills", "experience"])


# JSON schema fragments for each section type
_SECTION_SCHEMAS = {
    "summary": '  "summary": "3-5 sentence professional summary tailored to the job description"',
    "skills": '  "skills": {{\n    "Category1": "skill1, skill2, skill3",\n    "Category2": "skill4, skill5"\n  }}',
    "experience": '  "experience": [\n    {{\n      "company": "Company Name",\n      "title": "Job Title",\n      "dates": "Mon YYYY \u2013 Mon YYYY",\n      "bullets": [\n        "Achievement with quantifiable metrics",\n        "Achievement with quantifiable metrics"\n      ]\n    }}\n  ]',
    "education": '  "education": [\n    {{\n      "degree": "Degree and Major",\n      "university": "University Name",\n      "date": "Mon YYYY",\n      "gpa": "GPA if available or empty string"\n    }}\n  ]',
    "certifications": '  "certifications": ["Certification Name (Year)"]',
    "projects": '  "projects": [\n    {{\n      "name": "Project Name",\n      "description": "Brief description",\n      "bullets": ["Key achievement"]\n    }}\n  ]',
}


def build_json_schema(sections: list[str]) -> str:
    """Build a dynamic JSON schema example based on the requested sections."""
    parts = [
        '  "name": "Candidate Full Name"',
        '  "contact": "email | phone | location | linkedin"',
    ]
    for s in sections:
        if s in _SECTION_SCHEMAS:
            parts.append(_SECTION_SCHEMAS[s])
        else:
            # Custom section
            parts.append(f'  "custom_sections": {{{{ "{s}": ["Item 1", "Item 2"] }}}}')
    return "{{\n" + ",\n".join(parts) + "\n}}"
