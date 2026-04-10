"""Generate ATS-optimized resumes from job descriptions."""

import sys
import re
from datetime import datetime
from pathlib import Path
from src.rag_chain import generate_resume, score_resume, refine_resume, extract_job_info
from src.vector_store import search_resumes
from src.docx_writer import save_resume_docx
from src.config import RESUME_FOLDER

MAX_ITERATIONS = 2
MIN_ATS_SCORE = 90


def read_job_description() -> str:
    """Read JD from file arg, stdin pipe, or interactive input."""
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            return f.read()

    if not sys.stdin.isatty():
        return sys.stdin.read()

    print("Paste the job description (press Enter twice to submit):\n")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def main():
    print("=== ATS Resume Generator ===\n")
    job_description = read_job_description()

    if not job_description.strip():
        print("No job description provided.")
        return

    # Step 1: Score only the top matching resume
    print("\nSearching for matching existing resumes...\n")
    existing = search_resumes(job_description, k=1)

    if existing:
        r = existing[0]
        print(f"Scoring best match: {r['source']}")
        score, _, _ = score_resume(r["content"], job_description)
        print(f"  ATS Score: {score}/100\n")

        if score >= MIN_ATS_SCORE:
            print(f"✅ Existing resume scores {score}/100. No generation needed!")
            print(f"📄 Use: {r['source']}")
            return

    # Step 2: Generate and refine
    print(f"⚠️  No existing resume meets {MIN_ATS_SCORE}. Generating new one...\n")
    resume, skills, experience = generate_resume(job_description)
    best_resume, best_score = resume, 0

    for iteration in range(MAX_ITERATIONS):
        print(f"=== Iteration {iteration + 1} ===")
        print(f"\n{resume}\n")

        print("Scoring resume...\n")
        score, feedback, parsed = score_resume(resume, job_description)
        print(feedback)

        if score > best_score:
            best_score = score
            best_resume = resume

        if score >= MIN_ATS_SCORE:
            print(f"\n✅ ATS score {score}/100 meets threshold. Done!")
            break

        print(f"\n⚠️  ATS score {score}/100 is below {MIN_ATS_SCORE}. Refining...")
        print(f"   Missing keywords: {parsed['missing_keywords']}")
        print(f"   Improvements:\n{parsed['improvements']}\n")
        resume = refine_resume(resume, job_description, parsed, skills, experience)
    else:
        print("\n⚠️  Max iterations reached.")

    print(f"\n=== Final Resume (Best ATS Score: {best_score}/100) ===\n")
    print(best_resume)

    company, title = extract_job_info(job_description)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w-]", "_", f"{company}_{title}")[:60]
    filename = f"{safe_name}_{timestamp}.docx"

    output_path = Path(RESUME_FOLDER) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_resume_docx(best_resume, str(output_path))
    print(f"\n📄 Resume saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
