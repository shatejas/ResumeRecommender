"""Scan job portals and generate resumes — decoupled into scan + generate steps.

Usage:
    python3 scan_jobs.py              # scan then generate
    python3 scan_jobs.py scan         # scan only, saves jobs to data/scanned_jobs.json
    python3 scan_jobs.py generate     # generate from previously scanned jobs
"""

import json
import re
import sys
import time
import yaml
from pathlib import Path
from playwright.sync_api import sync_playwright
from src.scanner.scanner import scan

JOBS_FILE = Path(__file__).parent / "data" / "scanned_jobs.json"
API_KEYS_FILE = Path(__file__).parent / "api_keys.yml"
MAX_RESUMES = 2


# --- LLM setup ---

def _setup_llm_provider():
    """Auto-detect best LLM from api_keys.yml. Gemini > ChatGPT > Claude > Ollama."""
    from src.rag_chain import set_llm_provider

    keys = {}
    if API_KEYS_FILE.exists():
        try:
            keys = yaml.safe_load(API_KEYS_FILE.read_text()) or {}
        except Exception:
            pass

    if keys.get("gemini"):
        set_llm_provider("gemini", keys["gemini"])
        return "Gemini"
    if keys.get("openai"):
        set_llm_provider("openai", keys["openai"])
        return "ChatGPT"
    if keys.get("anthropic"):
        set_llm_provider("claude", keys["anthropic"])
        return "Claude"

    set_llm_provider("ollama")
    return "Ollama"


# --- Scan ---

def _scan_jobs():
    """Scan portals, save results to JSON, return (jobs, config)."""
    start = time.time()
    jobs, config = scan()
    elapsed = time.time() - start

    if not jobs:
        print(f"\nNo new matching jobs found. ({elapsed:.1f}s)")
        return [], config

    sorted_jobs = sorted(jobs, key=lambda x: x.match_score, reverse=True)

    # Display
    print(f"\n{'='*90}")
    print(f"{'Title':<40} {'Company':<20} {'Location':<20} {'Score'}")
    print(f"{'='*90}")
    for j in sorted_jobs:
        score_str = f"{j.match_score:+.0f}" if j.match_score else ""
        print(f"{j.title[:39]:<40} {j.company[:19]:<20} {j.location[:19]:<20} {score_str}")
        print(f"  🔗 {j.url}")
    print(f"\n📊 {len(jobs)} new matching jobs. ({elapsed:.1f}s)")

    # Save to JSON
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config": {
            "needs_sponsorship": config.get("needs_sponsorship", False),
            "location_countries": config.get("location_filter", {}).get("countries", []),
        },
        "jobs": [
            {"title": j.title, "company": j.company, "url": j.url,
             "location": j.location, "match_score": j.match_score}
            for j in sorted_jobs
        ],
    }
    JOBS_FILE.write_text(json.dumps(data, indent=2))
    print(f"\n💾 Saved {len(sorted_jobs)} jobs to {JOBS_FILE}")

    return sorted_jobs, config


def _load_scanned_jobs():
    """Load previously scanned jobs from JSON."""
    if not JOBS_FILE.exists():
        print(f"❌ No scanned jobs found. Run: python3 scan_jobs.py scan")
        return None, None

    data = json.loads(JOBS_FILE.read_text())
    config = data["config"]

    from src.scanner.models import JobListing
    jobs = [JobListing(**j) for j in data["jobs"]]

    print(f"📂 Loaded {len(jobs)} jobs from {JOBS_FILE}")
    return jobs, config


# --- Generate ---

def _fetch_jd(url: str, page) -> str:
    """Fetch JD text from a job URL."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(2000)
        for sel in ["nav", "header", "footer", "[class*='cookie']"]:
            for el in page.query_selector_all(sel):
                try:
                    el.evaluate("el => el.remove()")
                except Exception:
                    pass
        return page.inner_text("body")
    except Exception as e:
        print(f"      ⚠️ Failed to fetch JD: {e}")
        return ""


def _llm_check(jd_text: str) -> dict:
    """Ask LLM to check location and sponsorship."""
    from src.rag_chain import _get_llm

    prompt = f"""Analyze this job description and answer these two questions.
Respond in EXACTLY this format and nothing else:

LOCATION: <city, state/country or "Remote" or "Not specified">
SPONSORSHIP: <"yes" ONLY if the company explicitly says they sponsor visas, "no" ONLY if they explicitly say they do NOT sponsor, "unknown" if sponsorship is not mentioned at all>

IMPORTANT: If the job description does not mention sponsorship, visa, or work authorization at all, you MUST answer "unknown". Only answer "no" if the JD explicitly states they will not sponsor.

Job Description:
{jd_text[:3000]}"""

    try:
        response = _get_llm(temperature=0).invoke(prompt).content
        result = {"location": "", "sponsorship": "unknown"}
        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("LOCATION:"):
                result["location"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SPONSORSHIP:"):
                val = line.split(":", 1)[1].strip().lower()
                if "unknown" in val or "not mentioned" in val or "not specified" in val:
                    result["sponsorship"] = "unknown"
                elif val.startswith("yes") or ("sponsor" in val and "not" not in val and "no" not in val):
                    result["sponsorship"] = "sponsors"
                elif val.startswith("no") and "not mentioned" not in val and "unknown" not in val:
                    result["sponsorship"] = "no_sponsorship"
                else:
                    result["sponsorship"] = "unknown"
        return result
    except Exception as e:
        print(f"      ⚠️ LLM check failed: {e}")
        return {"location": "", "sponsorship": "unknown"}


def _generate_resume(jd_text: str, job):
    """Generate a resume for a job using the existing pipeline."""
    from src.rag_chain import generate_resume_structured, generate_resume, score_resume, refine_resume
    from src.vector_store import search_resumes, ensure_search_pipeline
    from src.structured_writer import save_structured_docx
    from src.docx_writer import save_resume_docx
    from src.candidate import load_candidate, get_sections
    from src.config import RESUME_FOLDER

    ensure_search_pipeline()
    results = search_resumes(jd_text, k=5)

    # Generate
    structured_data, skills, experience = generate_resume_structured(jd_text, results=results)
    if structured_data:
        resume_text = structured_data.to_text() if hasattr(structured_data, 'to_text') else str(structured_data)
    else:
        resume_text, skills, experience = generate_resume(jd_text, results=results)
        structured_data = None

    # Score
    score, _, parsed = score_resume(resume_text, jd_text)
    print(f"      ATS Score: {score}/100")

    # Refine once if below 90
    if score < 90:
        resume_text = refine_resume(resume_text, jd_text, parsed, skills, experience)
        score2, _, _ = score_resume(resume_text, jd_text)
        print(f"      Refined: {score2}/100")
        if score2 > score:
            structured_data = None

    # Save to CompanyName/ folder
    safe_company = re.sub(r"[^\w\s-]", "", job.company).strip().replace(" ", "_")
    safe_title = re.sub(r"[^\w\s-]", "", job.title)[:40].strip().replace(" ", "_")
    output_dir = Path(RESUME_FOLDER) / safe_company
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_title}.docx"

    if structured_data:
        candidate = load_candidate()
        sections = get_sections(candidate)
        save_structured_docx(structured_data, str(output_path), sections=sections)
    else:
        save_resume_docx(resume_text, str(output_path))

    print(f"      💾 {output_path}")


def _generate_resumes(jobs, config):
    """Fetch JDs, validate, and generate resumes for scanned jobs."""
    provider = _setup_llm_provider()
    print(f"🤖 Using LLM: {provider}")

    needs_sponsorship = config.get("needs_sponsorship", False) if isinstance(config, dict) else config.get("needs_sponsorship", False)
    generated = 0

    print(f"\n{'='*90}")
    print(f"📝 Fetching JDs and generating resumes (max {MAX_RESUMES})...")
    print(f"{'='*90}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for job in jobs:
            if generated >= MAX_RESUMES:
                break

            print(f"\n  🔍 {job.company} | {job.title}")

            jd_text = _fetch_jd(job.url, page)
            if not jd_text:
                print(f"    ⚠️ Could not fetch JD, skipping")
                continue

            print(f"    🤖 Checking location & sponsorship...")
            info = _llm_check(jd_text)
            print(f"    📍 Location: {info['location']}")
            print(f"    🛂 Sponsorship: {info['sponsorship']}")

            if needs_sponsorship and info["sponsorship"] == "no_sponsorship":
                print(f"    ❌ No sponsorship — skipping")
                continue

            print(f"    ✅ Passed! Generating resume...")
            _generate_resume(jd_text, job)
            generated += 1

        page.close()
        browser.close()

    print(f"\n{'='*90}")
    print(f"✅ Generated {generated} resumes.")


# --- Main ---

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd == "scan":
        print("=== Job Portal Scanner ===")
        _scan_jobs()

    elif cmd == "generate":
        print("=== Resume Generator ===")
        jobs, config = _load_scanned_jobs()
        if jobs:
            _generate_resumes(jobs, config)

    else:
        print("=== Job Portal Scanner ===")
        jobs, config = _scan_jobs()
        if jobs:
            _generate_resumes(jobs, config)


if __name__ == "__main__":
    main()
