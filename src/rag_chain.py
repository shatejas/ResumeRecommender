"""RAG chain for ATS-optimized resume generation."""

import re
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.vector_store import search_resumes

RESUME_PROMPT = PromptTemplate(
    input_variables=["skills", "experience", "context", "question"],
    template="""You are an expert resume writer. Generate an ATS-optimized resume using ONLY the candidate data provided below. Do NOT invent any information.

CANDIDATE SKILLS (use ONLY these):
{skills}

CANDIDATE EXPERIENCE (use ONLY these roles and achievements):
{experience}

Full resume context for reference:
{context}

Job Description:
{question}

You MUST follow this EXACT format. Here is an example:

Jane Doe
jane.doe@email.com | 555-123-4567 | Seattle, WA | linkedin.com/in/janedoe

SUMMARY
Senior Data Scientist with 6+ years of experience in machine learning and NLP. Proven track record of deploying scalable ML pipelines serving 10M+ users.

SKILLS
Programming: Python, SQL, R, Java
ML Frameworks: PyTorch, TensorFlow, scikit-learn
Cloud: AWS SageMaker, EMR, S3, Redshift
Tools: Spark, Airflow, MLflow, Docker

EXPERIENCE
Acme Corp | Senior Data Scientist | Jan 2022 – Present
- Built and deployed recommendation engine using collaborative filtering and deep learning, increasing CTR by 25% and driving $2M incremental revenue
- Designed and maintained feature engineering pipeline processing 5TB daily event data on Spark, reducing data prep time by 40%
- Led A/B experiments across 3 product lines to measure model impact on conversion, revenue, and user engagement metrics
- Mentored 3 junior data scientists, established team ML best practices, and led weekly knowledge-sharing sessions
- Optimized model inference latency from 200ms to 50ms for real-time serving using model quantization and caching

Beta Inc | Data Scientist | Jun 2019 – Dec 2021
- Developed NLP models for text classification and sentiment analysis achieving 92% accuracy on production data
- Optimized model inference latency by 40% for real-time serving, supporting 1M+ daily predictions
- Built automated feature engineering pipeline reducing data prep time by 60% and improving model iteration speed
- Conducted exploratory data analysis on 50+ datasets to identify trends, anomalies, and business opportunities

EDUCATION
MS Computer Science — University of Washington, 2019
BS Information Technology — State University, 2017

CERTIFICATIONS
AWS Certified Cloud Practitioner (2024)

CRITICAL RULES:
- ONLY use skills, experience, education, and certifications from the CANDIDATE DATA above
- Do NOT invent degrees, certifications, companies, job titles, or metrics not in the candidate data
- Do NOT use the company name or job title from the Job Description as experience. The Job Description is the TARGET role, not past experience.
- EXPERIENCE section must ONLY contain companies and roles listed in CANDIDATE EXPERIENCE above
- You MAY rephrase existing bullet points to include job description keywords, but do NOT change company names, job titles, or dates
- You MAY reorder skills to prioritize those matching the job description
- SUMMARY must be ONLY 2-3 sentences. Do NOT put skills, education, or experience in SUMMARY.
- SKILLS must be ONLY a grouped list of technical skills from the candidate data. No sentences.
- EXPERIENCE must have role lines as: Company | Title | Mon YYYY – Mon YYYY
- EXPERIENCE must list the MOST RECENT job FIRST and OLDEST job LAST. Example: if someone worked at Company A (2024-Present) and Company B (2020-2023), Company A MUST come before Company B.
- Every role MUST have 3-5 bullet points starting with "-" describing achievements with quantifiable metrics. Each bullet should be 1-2 lines long.
- Each section must appear exactly ONCE
- Do NOT add explanations or commentary before or after the resume

Output the resume now:""",
)

ATS_SCORE_PROMPT = PromptTemplate(
    input_variables=["resume", "job_description"],
    template="""You are an ATS (Applicant Tracking System) scoring engine.

Resume:
{resume}

Job Description:
{job_description}

Score this resume against the job description. Respond in EXACTLY this format:

ATS SCORE: <score>/100

KEYWORD MATCHES: <comma-separated list of matched keywords>

MISSING KEYWORDS: <comma-separated list of important missing keywords>

IMPROVEMENTS:
- <suggestion 1>
- <suggestion 2>
- <suggestion 3>""",
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["resume", "job_description", "missing_keywords", "improvements", "skills", "experience"],
    template="""You are an expert resume writer. Improve this resume using ONLY the candidate data provided.

Current resume:
{resume}

Job Description:
{job_description}

CANDIDATE SKILLS (use ONLY these):
{skills}

CANDIDATE EXPERIENCE (use ONLY these roles and achievements):
{experience}

MISSING KEYWORDS that MUST be added (only if they exist in candidate data):
{missing_keywords}

IMPROVEMENTS that MUST be applied:
{improvements}

Rewrite the resume with these fixes applied. Follow this EXACT format:

[Full Name]
[Contact info on one line]

SUMMARY
[2-3 sentences ONLY. No skills lists, no education, no experience details here.]

SKILLS
[Grouped technical skills — only from candidate data, prioritize missing keywords that match]

EXPERIENCE
[Company | Title | Mon YYYY – Mon YYYY]
- [Achievement bullet 1 with quantifiable metrics]
- [Achievement bullet 2 with quantifiable metrics]
- [Achievement bullet 3 with quantifiable metrics]

[Repeat for each role with 3-5 bullets each]

EDUCATION
[Degree — University, Year]

CERTIFICATIONS
[Certification (Year)]

CRITICAL RULES:
- ONLY use data from the CANDIDATE SKILLS and CANDIDATE EXPERIENCE sections above
- Do NOT invent degrees, certifications, companies, job titles, or metrics
- Do NOT use the company name or job title from the Job Description as experience. The Job Description is the TARGET role, not past experience.
- EXPERIENCE section must ONLY contain companies and roles listed in CANDIDATE EXPERIENCE above
- EXPERIENCE must list the MOST RECENT job FIRST and OLDEST job LAST. Example: if someone worked at Company A (2024-Present) and Company B (2020-2023), Company A MUST come before Company B.
- You MAY rephrase existing bullets to incorporate missing keywords, but do NOT change company names, job titles, or dates
- SUMMARY is ONLY 2-3 sentences. Nothing else.
- Each section appears exactly ONCE
- Every role MUST have 3-5 bullet points starting with "-". Each bullet should be 1-2 lines long.
- The resume MUST fill exactly one full page. If it is too short, add more bullet points to experience roles using details from the CANDIDATE EXPERIENCE data. Each bullet should be 1-2 lines long.
- Do NOT add commentary before or after the resume

Output the improved resume now:""",
)


from typing import Optional


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


def _get_llm(temperature: Optional[float] = None) -> ChatOllama:
    kwargs = {"model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOllama(**kwargs)


def _collect_structured_data(results: list[dict]) -> tuple[str, str]:
    """Aggregate skills and experience from search results."""
    all_skills = set()
    all_experience = []

    for r in results:
        if r["skills"]:
            all_skills.update(s.strip() for s in r["skills"].splitlines() if s.strip())
        if r["experience"]:
            all_experience.append(r["experience"])

    return "\n".join(sorted(all_skills)), "\n\n".join(all_experience)


def generate_resume(job_description: str) -> tuple[str, str, str]:
    """Generate an ATS-optimized resume. Returns (resume, skills, experience)."""
    results = search_resumes(job_description)

    print("--- Retrieved Sources ---")
    for r in results:
        print(f"  • {r['source']}")
    print("--- End Sources ---\n")

    skills, experience = _collect_structured_data(results)
    context = "\n\n---\n\n".join(r["content"] for r in results)

    response = _get_llm().invoke(RESUME_PROMPT.format(
        skills=skills, experience=experience, context=context, question=job_description,
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
