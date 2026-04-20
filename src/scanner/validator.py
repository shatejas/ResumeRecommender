"""Validate job listings by fetching full JD and using LLM to check location + sponsorship."""

from playwright.sync_api import sync_playwright
from src.scanner.models import JobListing


def _fetch_jd_text(url: str, page) -> str:
    """Fetch and extract text from a job posting URL."""
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
        print(f"      ⚠️ Failed to fetch: {e}")
        return ""


def _llm_check_job(jd_text: str) -> dict:
    """Ask the LLM to extract location and sponsorship info from a JD."""
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
                # Only mark as no_sponsorship if the LLM clearly says "no"
                # and doesn't also say "unknown" or "not mentioned"
                if "unknown" in val or "not mentioned" in val or "not specified" in val:
                    result["sponsorship"] = "unknown"
                elif val.startswith("yes") or "sponsor" in val and "not" not in val and "no" not in val:
                    result["sponsorship"] = "sponsors"
                elif val.startswith("no") and "not mentioned" not in val and "unknown" not in val:
                    result["sponsorship"] = "no_sponsorship"
                else:
                    result["sponsorship"] = "unknown"

        return result
    except Exception as e:
        print(f"      ⚠️ LLM check failed: {e}")
        return {"location": "", "sponsorship": "unknown"}


def validate_jobs(jobs: list[JobListing]) -> list[dict]:
    """Validate jobs by fetching JDs and using LLM to check location + sponsorship.
    Returns list of dicts: {job, sponsorship, location, jd_text}
    """
    if not jobs:
        return []

    results = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for i, job in enumerate(jobs):
                print(f"    [{i+1}/{len(jobs)}] {job.company} | {job.title[:40]}...", end=" ")
                text = _fetch_jd_text(job.url, page)

                if text:
                    info = _llm_check_job(text)
                    # Update job location if LLM found one and job didn't have it
                    if info["location"] and not job.location:
                        job.location = info["location"]
                    print(f"→ {info['sponsorship']} | {info['location'][:30]}")
                else:
                    info = {"location": "", "sponsorship": "unknown"}
                    print(f"→ no JD text")

                results.append({
                    "job": job,
                    "sponsorship": info["sponsorship"],
                    "location": info["location"],
                    "jd_text": text,
                })

            page.close()
            browser.close()

    except Exception as e:
        print(f"  ⚠️ Validation error: {e}")
        for job in jobs[len(results):]:
            results.append({"job": job, "sponsorship": "unknown", "location": "", "jd_text": ""})

    return results
