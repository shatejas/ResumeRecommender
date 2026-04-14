"""RAG chain for ATS-optimized resume generation."""

import re
from typing import Optional
from langchain_ollama import ChatOllama
from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.prompts import RESUME_PROMPT, ATS_SCORE_PROMPT, REFINE_PROMPT
from src.candidate import load_candidate, format_contact, format_education, format_certifications


# LLM provider state — set by UI or defaults to Ollama
_llm_provider = "ollama"
_api_key = ""


def set_llm_provider(provider: str, api_key: str = ""):
    """Set the LLM provider. Called from the UI."""
    global _llm_provider, _api_key
    _llm_provider = provider
    _api_key = api_key


def _parse_feedback(feedback: str) -> dict:
    """Parse ATS feedback into structured fields."""
    result = {"score": 0, "missing_keywords": "", "improvements": ""}
    current_section = None
    improvements = []

    for line in feedback.splitlines():
        upper = line.upper().strip()
        if "ATS SCORE:" in upper:
            match = re.search(r"(\d+)", line)
            if match:
                result["score"] = int(match.group(1))
        elif "MISSING KEYWORDS:" in upper:
            result["missing_keywords"] = line.split(":", 1)[1].strip()
            current_section = None
        elif "IMPROVEMENTS:" in upper or "IMPROVEMENT:" in upper:
            current_section = "improvements"
        elif current_section == "improvements" and line.strip().startswith("-"):
            improvements.append(line.strip())

    result["improvements"] = "\n".join(improvements)
    return result


def _get_llm(temperature: Optional[float] = None):
    if _llm_provider == "gemini" and _api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": "gemini-2.5-flash", "google_api_key": _api_key}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatGoogleGenerativeAI(**kwargs)

    if _llm_provider == "openai" and _api_key:
        from langchain_openai import ChatOpenAI
        kwargs = {"model": "gpt-4o-mini", "api_key": _api_key}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)

    kwargs = {"model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOllama(**kwargs)


def _collect_structured_data(results: list[dict]) -> tuple[str, str, str, str]:
    """Aggregate skills, experience, education, and certifications from search results."""
    all_skills = set()
    all_experience = []
    all_education = []
    all_certifications = []

    for r in results:
        if r["skills"]:
            all_skills.update(s.strip() for s in r["skills"].splitlines() if s.strip())
        if r["experience"]:
            all_experience.append(r["experience"])
        if r.get("education"):
            all_education.append(r["education"])
        if r.get("certifications"):
            all_certifications.append(r["certifications"])

    return (
        "\n".join(sorted(all_skills)),
        "\n\n".join(all_experience),
        "\n".join(all_education),
        "\n".join(all_certifications),
    )


def generate_resume(job_description: str, results: list = None, candidate_yaml: str = "") -> tuple[str, str, str]:
    """Generate an ATS-optimized resume. Returns (resume, skills, experience)."""
    if results is None:
        from src.vector_store import search_resumes
        results = search_resumes(job_description)

    skills, experience, education, certifications = _collect_structured_data(results)
    context = "\n\n---\n\n".join(r["content"] for r in results)

    # Candidate YAML overrides for contact, education, certifications
    candidate = load_candidate(candidate_yaml)
    contact = format_contact(candidate)
    yml_education = format_education(candidate)
    yml_certifications = format_certifications(candidate)
    if yml_education:
        education = yml_education
    if yml_certifications:
        certifications = yml_certifications

    response = _get_llm().invoke(RESUME_PROMPT.format(
        contact=contact, skills=skills, experience=experience, education=education,
        certifications=certifications, context=context, question=job_description,
    )).content

    return response, skills, experience


def score_resume(resume: str, job_description: str) -> tuple[int, str, dict]:
    """Score a resume against a job description. Returns (score, feedback, parsed)."""
    feedback = _get_llm(temperature=0).invoke(
        ATS_SCORE_PROMPT.format(resume=resume, job_description=job_description)
    ).content
    parsed = _parse_feedback(feedback)
    return parsed["score"], feedback, parsed


def refine_resume(resume: str, job_description: str, parsed_feedback: dict,
                  skills: str, experience: str) -> str:
    """Refine a resume based on parsed ATS feedback and structured candidate data."""
    return _get_llm().invoke(REFINE_PROMPT.format(
        resume=resume,
        job_description=job_description,
        missing_keywords=parsed_feedback["missing_keywords"],
        improvements=parsed_feedback["improvements"],
        skills=skills,
        experience=experience,
    )).content


def extract_job_info(job_description: str) -> tuple[str, str]:
    """Extract company name and job title from a job description using regex."""
    company = ""
    title = ""

    # Try "Company: X" or "Company Name: X" or "Organization: X"
    co_match = re.search(
        r"(?:company|organization|employer)\s*(?:name)?\s*:\s*(.+)",
        job_description, re.IGNORECASE,
    )
    if co_match:
        company = co_match.group(1).strip().split("\n")[0]

    # Try "Title: X" or "Role: X" or "Position: X"
    title_match = re.search(
        r"(?:job\s*title|role|position|title)\s*:\s*(.+)",
        job_description, re.IGNORECASE,
    )
    if title_match:
        title = title_match.group(1).strip().split("\n")[0]

    # Fallback: first line is often the title (e.g. "Senior Data Scientist – Personalization")
    if not title:
        first_line = job_description.strip().splitlines()[0].strip()
        if len(first_line) < 80 and not first_line.endswith("."):
            title = first_line

    # Clean: remove location/extra info after common separators
    title = re.split(r"\s*[–\-—|]\s*(?:Remote|Hybrid|On.?site|Location)", title, flags=re.IGNORECASE)[0].strip()
    company = re.sub(r"\s*\(.*?\)", "", company).strip()  # Remove parentheticals like (Hybrid)

    return company or "Company", title or "Role"
