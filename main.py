"""Generate ATS-optimized resumes from job descriptions."""

import sys
from pathlib import Path
from src.rag_chain import generate_resume, score_resume, refine_resume
from src.vector_store import search_resumes
from src.pdf_writer import save_resume_pdf
from src.config import RESUME_FOLDER

MAX_ITERATIONS = 3
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

    # Step 1: Check if any existing resume already meets the threshold
    print("\nSearching for matching existing resumes...\n")
    existing = search_resumes(job_description)
    best_resume, best_score, best_source = None, 0, None

    for doc in existing:
        source = doc.metadata["source"]
        print(f"Scoring existing resume: {source}")
        score, feedback = score_resume(doc.page_content, job_description)
        print(f"  ATS Score: {score}/100\n")

        if score > best_score:
            best_score = score
            best_resume = doc.page_content
            best_source = source

        if score >= MIN_ATS_SCORE:
            print(f"✅ Existing resume scores {score}/100. No generation needed!")
            print(f"📄 Use: {source}")
            return

    # Step 2: Generate and refine if no existing resume meets threshold
    if best_score < MIN_ATS_SCORE:
        print(f"\n⚠️  No existing resume meets {MIN_ATS_SCORE}. Generating new one...\n")
        resume = generate_resume(job_description)

        for iteration in range(MAX_ITERATIONS):
            print(f"=== Iteration {iteration + 1} ===")
            print(f"\n{resume}\n")

            print("Scoring resume...\n")
            score, feedback = score_resume(resume, job_description)
            print(feedback)

            if score > best_score:
                best_score = score
                best_resume = resume

            if score >= MIN_ATS_SCORE:
                print(f"\n✅ ATS score {score}/100 meets threshold. Done!")
                break

            print(f"\n⚠️  ATS score {score}/100 is below {MIN_ATS_SCORE}. Refining...\n")
            resume = refine_resume(resume, job_description, feedback)
        else:
            print(f"\n⚠️  Max iterations reached.")

    print(f"\n=== Final Resume (Best ATS Score: {best_score}/100) ===\n")
    print(best_resume)

    output_path = Path(RESUME_FOLDER) / "generated_resume.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_resume_pdf(best_resume, str(output_path))
    print(f"\n📄 Resume saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
