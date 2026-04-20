"""Gradio UI for ATS Resume Generator."""

import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
import gradio as gr
import requests
from bs4 import BeautifulSoup
from src.config import RESUME_FOLDER
from src.candidate import CANDIDATE_FILE

API_KEYS_FILE = Path(__file__).parent / "api_keys.yml"
SCAN_HISTORY_FILE = Path(__file__).parent / "data" / "scan_history.tsv"
JOB_STATUS_FILE = Path(__file__).parent / "data" / "job_status.json"


def _load_api_keys() -> dict:
    """Load API keys from api_keys.yml if it exists."""
    if API_KEYS_FILE.exists():
        import yaml
        try:
            data = yaml.safe_load(API_KEYS_FILE.read_text()) or {}
            return {k: (v or "") for k, v in data.items()}
        except Exception:
            pass
    return {}

MAX_ITERATIONS = 2
MIN_ATS_SCORE = 90


# --- Job status tracking ---

def _load_job_statuses() -> dict:
    """Load job statuses from JSON. Returns {url: status}."""
    if not JOB_STATUS_FILE.exists():
        return {}
    try:
        return json.loads(JOB_STATUS_FILE.read_text())
    except Exception:
        return {}


def _save_job_statuses(statuses: dict):
    JOB_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOB_STATUS_FILE.write_text(json.dumps(statuses, indent=2))


_job_statuses: dict = {}


def _refresh_statuses():
    global _job_statuses
    _job_statuses = _load_job_statuses()


def _set_job_status(urls_text: str, status: str) -> str:
    """Set status for selected job URLs."""
    if not urls_text.strip():
        return "❌ No jobs selected."
    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    for url in urls:
        _job_statuses[url] = status
    _save_job_statuses(_job_statuses)
    return f"✅ Marked {len(urls)} job(s) as {status}"


# --- Scan history cache ---

def _load_scan_history() -> list[dict]:
    """Load all jobs from scan_history.tsv, deduplicated by URL."""
    if not SCAN_HISTORY_FILE.exists():
        return []
    jobs = []
    seen_urls = set()
    for line in SCAN_HISTORY_FILE.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and not line.startswith("company\t"):
            url = parts[2]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            jobs.append({
                "company": parts[0],
                "title": parts[1],
                "url": url,
                "location": parts[3] if len(parts) > 3 else "",
            })
    return jobs


_scan_cache: list[dict] = []


def _refresh_scan_cache():
    global _scan_cache
    _scan_cache = _load_scan_history()
    _refresh_statuses()
    return _scan_cache


def _get_companies(status_filter="new") -> list[str]:
    if not _scan_cache:
        _refresh_scan_cache()
    jobs = _scan_cache
    if status_filter != "all":
        jobs = [j for j in jobs if _job_statuses.get(j["url"], "new") == status_filter]
    return sorted(set(j["company"] for j in jobs))


def _get_job_choices_for_company(company: str, status_filter: str = "new") -> gr.update:
    if not company:
        return gr.update(choices=[], value=[])
    jobs = [j for j in _scan_cache if j["company"] == company]
    if status_filter != "all":
        jobs = [j for j in jobs if _job_statuses.get(j["url"], "new") == status_filter]
    choices = [f"{j['title']} | {j['location'] or 'Remote'}" for j in jobs]
    return gr.update(choices=choices, value=[])


def _get_urls_for_jobs(company: str, job_choices: list[str]) -> str:
    if not company or not job_choices:
        return ""
    urls = []
    for choice in job_choices:
        title = choice.rsplit(" | ", 1)[0]
        for j in _scan_cache:
            if j["company"] == company and j["title"] == title:
                urls.append(j["url"])
                break
    return "\n".join(urls)


def _on_status_or_company_change(status_filter, company):
    """When status filter or company changes, update company list and job list."""
    companies = _get_companies(status_filter)
    # If current company not in filtered list, reset
    if company not in companies:
        company = None
    jobs_update = _get_job_choices_for_company(company, status_filter) if company else gr.update(choices=[], value=[])
    return gr.update(choices=companies, value=company), jobs_update, ""


def _on_jobs_change(company, job_choices):
    urls = _get_urls_for_jobs(company, job_choices)
    multi = len(job_choices) > 1
    return urls, gr.update(interactive=not multi), gr.update(interactive=not multi)


def _generate_for_scanned_jobs(urls_text, candidate_yaml, min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key):
    """Fetch JDs and generate resumes for multiple scanned job URLs."""
    if not urls_text.strip():
        return "❌ No jobs selected. Pick a company and select jobs first.", ""
    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
    all_logs, last_feedback = [], ""
    for i, url in enumerate(urls, 1):
        all_logs.append(f"{'='*60}\n📌 Job {i}/{len(urls)}: {url}\n{'='*60}")
        jd = fetch_jd_from_url(url)
        if not jd or jd.startswith("❌"):
            all_logs.append(f"❌ Could not fetch JD from {url}\n")
            continue
        log, feedback = generate(jd, candidate_yaml, min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key)
        all_logs.append(log + "\n")
        if feedback:
            last_feedback = feedback
    return "\n".join(all_logs), last_feedback


def _unified_generate(jd_text, selected_urls, candidate_yaml, min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key):
    urls = [u.strip() for u in selected_urls.strip().splitlines() if u.strip()]
    if len(urls) > 1:
        return _generate_for_scanned_jobs(selected_urls, candidate_yaml, min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key)
    if jd_text.strip():
        return generate(jd_text, candidate_yaml, min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key)
    if len(urls) == 1:
        jd = fetch_jd_from_url(urls[0])
        if jd and not jd.startswith("❌"):
            return generate(jd, candidate_yaml, min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key)
        return f"❌ Could not fetch JD from {urls[0]}", ""
    return "❌ No job description provided.", ""


def _run_scan():
    """Run the job scanner and return status message."""
    import subprocess
    print("🔍 Starting job scan...")
    result = subprocess.run(
        ["python3", "scan_jobs.py", "scan"],
        capture_output=False, text=True, cwd=str(Path(__file__).parent),
    )
    _refresh_scan_cache()
    return f"✅ Scan complete. {len(_scan_cache)} jobs in history."


def _generate_single_job(url: str, company: str, title: str, min_score: int, max_iter: int,
                         llm_choice: str, gemini_key: str, openai_key: str, claude_key: str) -> str:
    """Generate resume for a single job. Saves to per-job folder. Returns status line."""
    from src.rag_chain import set_llm_provider, generate_resume_structured, generate_resume, score_resume, refine_resume
    from src.vector_store import ensure_search_pipeline, search_resumes
    from src.structured_writer import save_structured_docx
    from src.docx_writer import save_resume_docx
    from src.candidate import load_candidate, get_sections
    import time as _time

    # Set LLM
    if llm_choice == "Gemini (API)":
        set_llm_provider("gemini", gemini_key)
    elif llm_choice == "ChatGPT (API)":
        set_llm_provider("openai", openai_key)
    elif llm_choice == "Claude (API)":
        set_llm_provider("claude", claude_key)
    else:
        set_llm_provider("ollama")

    ensure_search_pipeline()
    t0 = _time.time()

    # Fetch JD
    jd = fetch_jd_from_url(url)
    if not jd or jd.startswith("❌"):
        return f"❌ {title}: could not fetch JD"

    candidate_yaml = _load_candidate_yaml()

    # Search for reference resumes (fast, no LLM)
    results = search_resumes(jd, k=3)

    # Generate directly (skip scoring existing — these are new jobs)
    structured_data, skills, experience = generate_resume_structured(jd, results=results, candidate_yaml=candidate_yaml)
    if structured_data:
        resume = _structured_to_text(structured_data)
    else:
        resume, skills, experience = generate_resume(jd, results=results, candidate_yaml=candidate_yaml)
        structured_data = None

    # Score once
    score, feedback, parsed = score_resume(resume, jd)

    # Refine once if below threshold
    if score < min_score and max_iter > 0:
        resume = refine_resume(resume, jd, parsed, skills, experience)
        score2, feedback2, _ = score_resume(resume, jd)
        if score2 > score:
            score, feedback = score2, feedback2
            structured_data = None

    # Save to per-job folder
    safe_company = re.sub(r"[^\w\s-]", "", company).strip().replace(" ", "_")
    safe_title = re.sub(r"[^\w\s-]", "", title)[:50].strip().replace(" ", "_")
    output_dir = Path(RESUME_FOLDER) / safe_company / safe_title
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "resume.docx"
    if structured_data:
        candidate = load_candidate(candidate_yaml)
        sections = get_sections(candidate)
        save_structured_docx(structured_data, str(output_path), sections=sections)
    else:
        save_resume_docx(resume, str(output_path))

    # Write details.txt
    details = f"ATS Score: {score}/100\nJob URL: {url}\n\nATS Recommendation:\n{feedback}\n\n{'='*60}\nJob Description:\n{'='*60}\n{jd}"
    (output_dir / "details.txt").write_text(details)

    # Update status
    _job_statuses[url] = "generated"
    _save_job_statuses(_job_statuses)

    elapsed = _time.time() - t0
    return f"✅ {title}: {score}/100 ({elapsed:.0f}s) → {output_dir}"

    return f"✅ {title}: {score}/100 → {output_dir}"


def _generate_parallel(urls_text: str, company: str, min_score: int, max_iter: int,
                       llm_choice: str, gemini_key: str, openai_key: str, claude_key: str):
    """Generate resumes for selected jobs in parallel (5 threads)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not urls_text.strip():
        return "❌ No jobs selected."

    urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]

    # Map URLs to titles
    url_to_title = {}
    for j in _scan_cache:
        if j["url"] in urls:
            url_to_title[j["url"]] = j["title"]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                _generate_single_job, url, company, url_to_title.get(url, "Unknown"),
                min_score, max_iter, llm_choice, gemini_key, openai_key, claude_key
            ): url for url in urls
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                url = futures[future]
                results.append(f"❌ {url_to_title.get(url, url)}: {e}")

    return "\n".join(results)


from typing import Optional


def _extract_jsonld_job(soup: BeautifulSoup) -> Optional[str]:
    """Extract job description from JSON-LD structured data if present."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if data.get("@type") == "JobPosting" and data.get("description"):
                desc = data["description"]
                # Handle double-encoded HTML (e.g. &lt;p&gt; -> <p> -> text)
                decoded = BeautifulSoup(desc, "html.parser").get_text()
                desc_text = BeautifulSoup(decoded, "html.parser").get_text(separator="\n", strip=True)
                parts = []
                if data.get("title"):
                    parts.append(data["title"])
                org = data.get("hiringOrganization")
                if isinstance(org, dict) and org.get("name"):
                    parts.append(f"Company: {org['name']}")
                loc = data.get("jobLocation")
                if isinstance(loc, dict):
                    addr = loc.get("address", {})
                    city = addr.get("addressLocality", "")
                    state = addr.get("addressRegion", "")
                    if city or state:
                        parts.append(f"Location: {', '.join(filter(None, [city, state]))}")
                if parts:
                    parts.append("")
                parts.append(desc_text)
                return "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _clean_jd_text(text: str, max_chars: int = 5000) -> str:
    """Clean up extracted JD text: strip noise, collapse whitespace, truncate."""
    # Remove leftover HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Remove lines that are just symbols, very short nav items, or empty
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        # Skip lines that are just punctuation/symbols
        if re.match(r'^[\W\d\s]{1,10}$', line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated]"
    return text.strip()


def fetch_jd_from_url(url: str) -> str:
    """Scrape job description text from a URL. Tries JSON-LD first, then plain HTML."""
    if not url or not url.strip():
        return ""
    try:
        resp = requests.get(url.strip(), headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try JSON-LD structured data first (works for most job sites)
        jd = _extract_jsonld_job(soup)
        if jd:
            return _clean_jd_text(jd)

        # Fallback: plain HTML text extraction
        for tag in soup(["script", "style", "nav", "header", "footer", "iframe",
                         "noscript", "svg", "form", "button", "input"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return _clean_jd_text(text)
    except Exception as e:
        return f"❌ Failed to fetch: {e}"


def ingest_resumes(files):
    """Ingest uploaded resume files into the vector store."""
    if not files:
        return "No files uploaded. Please select files or a folder first."

    from src.vector_store import index_resume, ensure_index_exists, resume_exists
    from src.document_loader import load_single_resume, chunk_documents
    from src.vector_store import ensure_search_pipeline
    from ingest import extract_all_sections

    ensure_search_pipeline()
    ensure_index_exists()

    # Normalize: Gradio can pass strings, NamedString, or UploadFile objects
    file_paths = []
    for f in files:
        if isinstance(f, str):
            file_paths.append(f)
        elif hasattr(f, "name"):
            file_paths.append(f.name)
        else:
            continue

    if not file_paths:
        return "No valid files found."

    ingested, skipped = 0, 0
    log = []

    for fpath in file_paths:
        name = Path(fpath).name
        # Gradio folder uploads preserve original path in parent dirs
        # e.g. /tmp/gradio/.../Shweta Gupta/file.pdf
        original = str(Path(fpath).resolve())

        if not name.lower().endswith((".pdf", ".docx")):
            continue

        if resume_exists(name):
            skipped += 1
            log.append(f"⏭️ Skipped {name} (already indexed)")
            continue

        docs = load_single_resume(Path(fpath))
        if not docs:
            log.append(f"⚠️ Could not load {name}")
            continue

        for doc in docs:
            doc.metadata = {"source": name}

        chunks = chunk_documents(docs)
        full_text = "\n".join(d.page_content for d in docs)
        skills, experience, education, certifications, projects = extract_all_sections(full_text)
        index_resume(source=name, skills=skills, experience=experience,
                     education=education, certifications=certifications,
                     projects=projects, chunks=chunks, original_path=original)
        ingested += 1
        log.append(f"✅ Indexed {name} ({len(chunks)} chunks)")

    log.append(f"\n📊 {ingested} new, {skipped} skipped.")
    return "\n".join(log)


def _load_candidate_yaml() -> str:
    if CANDIDATE_FILE.exists():
        return CANDIDATE_FILE.read_text()
    return ""


def _save_candidate_yaml(yaml_text: str) -> str:
    import yaml
    try:
        yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return f"❌ Invalid YAML — not saved.\n{e}"
    CANDIDATE_FILE.write_text(yaml_text)
    return "✅ Saved candidate.yml"


def generate(job_description, candidate_yaml, min_score, max_iter, llm_choice, gemini_api_key, openai_api_key, claude_api_key):
    """Run the full pipeline: search, generate, score, refine, export."""
    if not job_description.strip():
        return "No job description provided.", ""

    # Always read candidate profile from saved file, not the textbox
    candidate_yaml = _load_candidate_yaml()

    from src.rag_chain import set_llm_provider
    from src.vector_store import ensure_search_pipeline

    # Set LLM provider
    if llm_choice == "Gemini (API)":
        if not gemini_api_key:
            return "❌ Please enter your Gemini API key.", ""
        set_llm_provider("gemini", gemini_api_key)
    elif llm_choice == "ChatGPT (API)":
        if not openai_api_key:
            return "❌ Please enter your OpenAI API key.", ""
        set_llm_provider("openai", openai_api_key)
    elif llm_choice == "Claude (API)":
        if not claude_api_key:
            return "❌ Please enter your Anthropic API key.", ""
        set_llm_provider("claude", claude_api_key)
    else:
        set_llm_provider("ollama")

    ensure_search_pipeline()
    log = []

    try:
        return _run_pipeline(job_description, candidate_yaml, min_score, max_iter, log)
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key" in error_msg:
            return "❌ Invalid API key. Please check and try again.", ""
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return "❌ API rate limit reached. Please wait and try again.", ""
        return f"❌ Error: {error_msg}", ""


def _structured_to_text(data) -> str:
    """Convert ResumeData to plain text for ATS scoring."""
    lines = [data.name, data.contact, "", "SUMMARY", data.summary, "", "SKILLS"]
    for cat, items in data.skills.items():
        lines.append(f"{cat}: {items}")
    lines.append("\nEXPERIENCE")
    for role in data.experience:
        lines.append(f"{role.company} | {role.title} | {role.dates}")
        for b in role.bullets:
            lines.append(f"- {b}")
        lines.append("")
    lines.append("EDUCATION")
    for edu in data.education:
        gpa = f", GPA: {edu.gpa}" if edu.gpa else ""
        lines.append(f"{edu.degree} | {edu.university}{gpa}, {edu.date}")
    if data.certifications:
        lines.append("\nCERTIFICATIONS")
        for cert in data.certifications:
            lines.append(cert)
    return "\n".join(lines)


def _run_pipeline(job_description, candidate_yaml, min_score, max_iter, log):
    from src.rag_chain import generate_resume, generate_resume_structured, score_resume, refine_resume, extract_job_info
    from src.vector_store import search_resumes
    from src.docx_writer import save_resume_docx
    from src.structured_writer import save_structured_docx
    from src.config import RESUME_FOLDER

    pipeline_start = time.time()

    # Step 1: Search once, reuse results
    t0 = time.time()
    log.append("🔍 Searching existing resumes...")
    results = search_resumes(job_description, k=5)
    for r in results:
        log.append(f"  • {Path(r['source']).name}")
    print(f"  ⏱ Search: {time.time() - t0:.1f}s")

    # Step 2: Score best match
    if results:
        r = results[0]
        t0 = time.time()
        log.append(f"\n📊 Scoring best match: {Path(r['source']).name}")
        score, feedback, _ = score_resume(r["content"], job_description)
        log.append(f"  ATS Score: {score}/100")
        print(f"  ⏱ Score existing: {time.time() - t0:.1f}s")

        if score >= min_score:
            source = r["source"]
            candidates = [
                Path(source),
                Path(RESUME_FOLDER) / Path(source).name,
            ]
            full_path = next((p for p in candidates if p.exists()), None)
            display_path = str(full_path) if full_path else source
            log.append(f"\n✅ Existing resume scores {score}/100. No generation needed!")
            log.append(f"📄 Use: {display_path}")

            return "\n".join(log), feedback

    # Step 3: Generate using search results directly (no second search)
    t0 = time.time()
    log.append("\n⚙️ Generating resume...")

    # Try structured generation first (tool-calling for API, JSON for Ollama)
    structured_data, skills, experience = generate_resume_structured(
        job_description, results=results, candidate_yaml=candidate_yaml
    )

    if structured_data:
        log.append("  ✅ Structured output generated")
        # Convert structured data to text for scoring/refinement
        resume = _structured_to_text(structured_data)
        use_structured = True
    else:
        log.append("  ⚠️ Structured output failed, falling back to text generation")
        resume, skills, experience = generate_resume(
            job_description, results=results, candidate_yaml=candidate_yaml
        )
        structured_data = None
        use_structured = False

    print(f"  ⏱ Generate: {time.time() - t0:.1f}s")
    best_resume, best_score, best_feedback = resume, 0, ""
    best_structured = structured_data

    # Step 4: Score and refine
    for iteration in range(int(max_iter)):
        t0 = time.time()
        log.append(f"\n--- Iteration {iteration + 1} ---")
        score, feedback, parsed = score_resume(resume, job_description)
        log.append(f"ATS Score: {score}/100")
        print(f"  ⏱ Score iter {iteration + 1}: {time.time() - t0:.1f}s")

        if score > best_score:
            best_score = score
            best_resume = resume
            best_structured = None  # Refined text loses structured format
            best_feedback = feedback

        if score >= min_score:
            log.append("✅ Score meets threshold!")
            break

        log.append(f"Missing: {parsed['missing_keywords']}")
        log.append("Refining...")
        t0 = time.time()
        resume = refine_resume(resume, job_description, parsed, skills, experience)
        print(f"  ⏱ Refine iter {iteration + 1}: {time.time() - t0:.1f}s")
    else:
        log.append("\n⚠️ Max iterations reached.")

    log.append(f"\n📊 Best ATS Score: {best_score}/100")
    total = time.time() - pipeline_start
    print(f"\n⏱ Total pipeline: {total:.1f}s")

    # Step 4: Export
    company, title = extract_job_info(job_description)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w-]", "_", f"{company}_{title}")[:60]
    filename = f"{safe_name}_{timestamp}.docx"

    output_path = Path(RESUME_FOLDER) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_structured:
        from src.candidate import get_sections, load_candidate
        candidate = load_candidate(candidate_yaml)
        sections = get_sections(candidate)
        save_structured_docx(best_structured, str(output_path), sections=sections)
    else:
        save_resume_docx(best_resume, str(output_path))

    log.append(f"💾 Saved to: {output_path}")

    return "\n".join(log), best_feedback


def clear_index():
    """Wipe the entire resume index."""
    from src.vector_store import ensure_index
    ensure_index()
    return "🗑️ Index cleared. All resumes removed."


with gr.Blocks(title="ATS Resume Generator") as app:
    gr.Markdown("# 📄 ATS Resume Generator")
    _keys = _load_api_keys()

    # Shared state for LLM settings across pages
    llm_state = gr.State("Ollama (Local)")
    gemini_state = gr.State(_keys.get("gemini", ""))
    openai_state = gr.State(_keys.get("openai", ""))
    claude_state = gr.State(_keys.get("anthropic", ""))
    score_state = gr.State(MIN_ATS_SCORE)
    iter_state = gr.State(MAX_ITERATIONS)

    with gr.Tabs():

        # ===== PAGE 1: Settings =====
        with gr.Tab("👤 Candidate Settings & Profile"):
            with gr.Row():
                # Left: LLM provider + candidate profile
                with gr.Column():
                    gr.Markdown("### LLM Provider")
                    gemini_key = gr.Textbox(label="Gemini API Key", type="password", value=_keys.get("gemini", ""))
                    openai_key = gr.Textbox(label="OpenAI API Key", type="password", value=_keys.get("openai", ""))
                    claude_key = gr.Textbox(label="Anthropic API Key", type="password", value=_keys.get("anthropic", ""))
                    save_keys_btn = gr.Button("💾 Save API Keys")

                    def _save_api_keys(gemini, openai_k, claude):
                        import yaml
                        data = {}
                        if gemini:
                            data["gemini"] = gemini
                        if openai_k:
                            data["openai"] = openai_k
                        if claude:
                            data["anthropic"] = claude
                        API_KEYS_FILE.write_text(yaml.dump(data, default_flow_style=False))
                        return "✅ API keys saved"

                    gr.Markdown("---")
                    gr.Markdown("### 👤 Candidate Profile")
                    candidate_editor = gr.Textbox(label="candidate.yml", value=_load_candidate_yaml(), lines=12)
                    save_candidate_btn = gr.Button("💾 Save Profile")

                # Right: reference resumes + clear index + log
                with gr.Column():
                    gr.Markdown("### 📁 Reference Resumes")
                    upload_files = gr.File(label="Select PDF/DOCX files", file_count="multiple", file_types=[".pdf", ".docx"])
                    ingest_files_btn = gr.Button("📤 Ingest Files")
                    upload_folder = gr.File(label="Select a folder", file_count="directory")
                    ingest_folder_btn = gr.Button("📤 Ingest Folder")
                    clear_btn = gr.Button("🗑️ Clear Index", variant="stop")
                    gr.Markdown("---")
                    settings_log = gr.Textbox(label="Log", lines=5, interactive=False)

            # Event wiring (outside Row so components are all defined)
            save_keys_btn.click(fn=_save_api_keys, inputs=[gemini_key, openai_key, claude_key], outputs=settings_log, queue=False)
            save_candidate_btn.click(fn=_save_candidate_yaml, inputs=candidate_editor, outputs=settings_log, queue=False)
            ingest_files_btn.click(fn=ingest_resumes, inputs=upload_files, outputs=settings_log)
            ingest_folder_btn.click(fn=ingest_resumes, inputs=upload_folder, outputs=settings_log)
            clear_btn.click(fn=clear_index, outputs=settings_log, queue=False)

        # ===== PAGE 2: Scan =====
        with gr.Tab("🔍 Scan"):
            with gr.Row():
                # Left: portals.yml editor
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 portals.yml")
                    portals_editor = gr.Textbox(
                        label="portals.yml",
                        value=Path(__file__).parent.joinpath("portals.yml").read_text() if Path(__file__).parent.joinpath("portals.yml").exists() else "",
                        lines=25,
                    )
                    save_portals_btn = gr.Button("💾 Save portals.yml")
                    portals_log = gr.Textbox(label="Log", lines=2, interactive=False)

                    def _save_portals(text):
                        import yaml
                        try:
                            yaml.safe_load(text)
                        except yaml.YAMLError as e:
                            return f"❌ Invalid YAML — not saved.\n{e}"
                        Path(__file__).parent.joinpath("portals.yml").write_text(text)
                        return "✅ Saved portals.yml"

                    save_portals_btn.click(fn=_save_portals, inputs=portals_editor, outputs=portals_log, queue=False)

                # Right: scan controls + results
                with gr.Column(scale=4):
                    gr.Markdown("### Scan Job Portals")
                    gr.Markdown("Scan all configured portals for new listings. Detailed logs appear in the console.")
                    scan_btn = gr.Button("🔄 Run Scan", variant="primary")
                    scan_output = gr.Textbox(label="Results", lines=25, max_lines=50, interactive=False)

                    def _run_scan_ui():
                        import subprocess
                        t0 = time.time()
                        result = subprocess.run(
                            ["python3", "scan_jobs.py", "scan"],
                            capture_output=True, text=True, cwd=str(Path(__file__).parent),
                        )
                        elapsed = time.time() - t0
                        _refresh_scan_cache()
                        output = result.stdout
                        lines = output.strip().splitlines()
                        summary = [f"✅ Scan complete ({elapsed:.1f}s)", ""]
                        for line in lines:
                            if "Total fetched:" in line or "After filtering:" in line or "New jobs:" in line:
                                summary.append(line.strip())
                        new_jobs = [j for j in _scan_cache if _job_statuses.get(j["url"], "new") == "new"]
                        if new_jobs:
                            from collections import Counter
                            counts = Counter(j["company"] for j in new_jobs)
                            summary.append(f"\n📊 {len(new_jobs)} new jobs across {len(counts)} companies:")
                            for company, count in counts.most_common(10):
                                summary.append(f"  • {company} ({count})")
                            if len(counts) > 10:
                                summary.append(f"  • ... and {len(counts) - 10} more")
                        return "\n".join(summary)

                    scan_btn.click(fn=_run_scan_ui, outputs=scan_output)

        # ===== PAGE 3: Job Board =====
        with gr.Tab("📋 Job Board"):
            with gr.Row():
                # Left sidebar: generation settings
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("### ⚙️ Generation")
                    jb_llm = gr.Radio(
                        choices=["Ollama (Local)", "Gemini (API)", "ChatGPT (API)", "Claude (API)"],
                        value="Ollama (Local)", label="LLM Provider",
                    )
                    min_score = gr.Slider(50, 100, value=MIN_ATS_SCORE, step=5, label="Min ATS Score")
                    max_iter = gr.Slider(1, 5, value=MAX_ITERATIONS, step=1, label="Max Iterations")

                # Right: job board
                with gr.Column(scale=5):
                    with gr.Row():
                        jb_company = gr.Dropdown(choices=_get_companies("new"), label="Company", interactive=True, scale=4)
                        jb_refresh = gr.Button("🔄", size="sm", scale=0, min_width=40)

                    jb_jobs = gr.CheckboxGroup(choices=[], label="Jobs", interactive=True)
                    jb_urls = gr.Textbox(label="Selected Job URLs", interactive=False, lines=2)

                    with gr.Row():
                        jb_generate = gr.Button("🚀 Generate Resumes", variant="primary")
                        jb_discard = gr.Button("🗑 Discard Selected")
                        jb_mark_applied = gr.Button("✅ Mark Applied")
                        jb_view_jd = gr.Button("👁 View JD", size="sm")

                    jb_jd_preview = gr.Textbox(label="Job Description", lines=10, visible=False, interactive=False)
                    jb_progress = gr.Textbox(label="Progress", lines=8, interactive=False)

            # --- Job Board wiring ---
            def _jb_refresh_all():
                _refresh_scan_cache()
                companies = _get_companies("new")
                return gr.update(choices=companies, value=None), gr.update(choices=[], value=[]), ""

            def _jb_company_change(company):
                return _get_job_choices_for_company(company, "new")

            def _jb_jobs_change(company, job_choices):
                return _get_urls_for_jobs(company, job_choices)

            def _jb_view(urls):
                if not urls or not urls.strip():
                    return gr.update(visible=False, value="")
                url = urls.splitlines()[0]
                jd = fetch_jd_from_url(url)
                return gr.update(visible=True, value=jd)

            def _jb_discard(urls_text):
                msg = _set_job_status(urls_text, "discarded")
                companies = _get_companies("new")
                return msg, gr.update(choices=companies), gr.update(choices=[], value=[]), ""

            def _jb_apply(urls_text):
                msg = _set_job_status(urls_text, "applied")
                companies = _get_companies("new")
                return msg, gr.update(choices=companies), gr.update(choices=[], value=[]), ""

            def _jb_gen(urls_text, company, llm, gemini, openai_k, claude, score, iters):
                return _generate_parallel(urls_text, company, score, iters, llm, gemini, openai_k, claude)

            jb_refresh.click(fn=_jb_refresh_all, outputs=[jb_company, jb_jobs, jb_urls], queue=False)
            jb_company.change(fn=_jb_company_change, inputs=jb_company, outputs=jb_jobs, queue=False)
            jb_jobs.change(fn=_jb_jobs_change, inputs=[jb_company, jb_jobs], outputs=jb_urls, queue=False)
            jb_view_jd.click(fn=_jb_view, inputs=jb_urls, outputs=jb_jd_preview)
            jb_discard.click(fn=_jb_discard, inputs=jb_urls, outputs=[jb_progress, jb_company, jb_jobs, jb_urls], queue=False)
            jb_mark_applied.click(fn=_jb_apply, inputs=jb_urls, outputs=[jb_progress, jb_company, jb_jobs, jb_urls], queue=False)
            jb_generate.click(
                fn=_jb_gen,
                inputs=[jb_urls, jb_company, jb_llm, gemini_key, openai_key, claude_key, min_score, max_iter],
                outputs=jb_progress,
            )

        # ===== PAGE 4: Manual Resume =====
        with gr.Tab("📄 Manual Resume"):
            gr.Markdown("### Paste a job description to generate a one-off resume")
            with gr.Row():
                jd_url = gr.Textbox(label="Job URL", placeholder="https://...", lines=1, scale=4)
                fetch_btn = gr.Button("🔗 Fetch", scale=1)
            jd_input = gr.Textbox(label="Job Description", lines=12, placeholder="Paste the full job description here or fetch from URL above...")
            fetch_btn.click(fn=fetch_jd_from_url, inputs=jd_url, outputs=jd_input)
            manual_generate_btn = gr.Button("🚀 Generate Resume", variant="primary")

            gen_log = gr.Textbox(label="Pipeline Log", lines=8, interactive=False)
            ats_feedback = gr.Textbox(label="ATS Recommendation", lines=8, interactive=False)

            manual_generate_btn.click(
                fn=generate,
                inputs=[jd_input, candidate_editor, min_score, max_iter, jb_llm, gemini_key, openai_key, claude_key],
                outputs=[gen_log, ats_feedback],
            )

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
