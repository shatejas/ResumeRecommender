"""RAG chain for ATS-optimized resume generation."""

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.vector_store import ResumeRetriever

RESUME_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert resume writer specializing in ATS-optimized resumes.

Using the following resume examples as reference:
{context}

Job Description:
{question}

Generate an ATS-optimized resume that:
1. Matches keywords from the job description
2. Uses clear section headers (Summary, Experience, Skills, Education, Certifications)
3. Lists experience in reverse chronological order (most recent job first), with company name, title, and dates
4. Includes quantifiable achievements
5. Uses standard formatting (no tables/graphics)
6. Incorporates relevant skills and experiences from the reference resumes

Output the resume in clean, ATS-friendly format:"""
)


def create_rag_chain():
    """Create the RAG chain for resume generation."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    retriever = ResumeRetriever(k=3)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RESUME_PROMPT},
    )


def generate_resume(job_description: str) -> str:
    """Generate an ATS-optimized resume for a job description."""
    chain = create_rag_chain()
    result = chain.invoke(job_description)

    print("--- Retrieved Sources ---")
    for doc in result["source_documents"]:
        print(f"  • {doc.metadata['source']}")
        print(f"    Preview: {doc.page_content[:100]}...\n")
    print("--- End Sources ---\n")

    return result["result"]


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
- <suggestion 3>"""
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["resume", "job_description", "feedback"],
    template="""You are an expert resume writer specializing in ATS-optimized resumes.

Here is a resume that needs improvement:
{resume}

Job Description:
{job_description}

ATS Feedback:
{feedback}

Rewrite the resume to address the feedback. Focus on:
1. Adding the missing keywords naturally into the experience section
2. Expanding experience bullets with more relevant details
3. Keeping all existing strong content
4. Maintaining reverse chronological order for experience (most recent job first)

Output the improved resume in clean, ATS-friendly format:"""
)


def score_resume(resume: str, job_description: str) -> tuple[int, str]:
    """Score a generated resume. Returns (score, full feedback text)."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    feedback = llm.invoke(ATS_SCORE_PROMPT.format(
        resume=resume, job_description=job_description
    )).content

    score = 0
    for line in feedback.splitlines():
        if "ATS SCORE:" in line.upper():
            import re
            match = re.search(r"(\d+)", line)
            if match:
                score = int(match.group(1))
            break
    return score, feedback


def refine_resume(resume: str, job_description: str, feedback: str) -> str:
    """Refine a resume based on ATS feedback."""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return llm.invoke(REFINE_PROMPT.format(
        resume=resume, job_description=job_description, feedback=feedback
    )).content
