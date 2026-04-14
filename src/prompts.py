"""Prompt templates loaded from text files with shared rules."""

from pathlib import Path
from langchain.prompts import PromptTemplate

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_RULES = (_PROMPTS_DIR / "rules.txt").read_text()


def _load(filename: str, input_variables: list[str]) -> PromptTemplate:
    template = (_PROMPTS_DIR / filename).read_text().replace("{rules}", _RULES)
    return PromptTemplate(input_variables=input_variables, template=template)


RESUME_PROMPT = _load("resume.txt", ["contact", "skills", "experience", "education", "certifications", "context", "question"])
ATS_SCORE_PROMPT = _load("ats_score.txt", ["resume", "job_description"])
REFINE_PROMPT = _load("refine.txt", ["resume", "job_description", "missing_keywords", "improvements", "skills", "experience"])
