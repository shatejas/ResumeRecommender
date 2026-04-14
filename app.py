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
            return jd

        # Fallback: plain HTML text extraction
        for tag in soup(["script", "style", "nav", "header", "footer", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
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


with gr.Blocks(title="ATS Resume Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📄 ATS Resume Generator")
    gr.Markdown("Generate ATS-optimized resumes from job descriptions using your existing resumes.")

    with gr.Row():
        # --- Column 1: Settings ---
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### ⚙️ Settings")
            _keys = _load_api_keys()
            llm_provider = gr.Radio(
                choices=["Ollama (Local)", "Gemini (API)", "ChatGPT (API)", "Claude (API)"],
                value="Ollama (Local)",
                label="LLM Provider",
            )
            gemini_key = gr.Textbox(
                label="Gemini API Key",
                type="password",
                value=_keys.get("gemini", ""),
                placeholder="Paste your API key or set in api_keys.yml",
                visible=False,
            )
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                value=_keys.get("openai", ""),
                placeholder="Paste your API key or set in api_keys.yml",
                visible=False,
            )
            claude_key = gr.Textbox(
                label="Anthropic API Key",
                type="password",
                value=_keys.get("anthropic", ""),
                placeholder="Paste your API key or set in api_keys.yml",
                visible=False,
            )
            llm_provider.change(
                fn=lambda p: (
                    gr.update(visible=p == "Gemini (API)"),
                    gr.update(visible=p == "ChatGPT (API)"),
                    gr.update(visible=p == "Claude (API)"),
                ),
                inputs=llm_provider,
                outputs=[gemini_key, openai_key, claude_key],
            )
            min_score = gr.Slider(50, 100, value=MIN_ATS_SCORE, step=5, label="Minimum ATS Score")
            max_iter = gr.Slider(1, 5, value=MAX_ITERATIONS, step=1, label="Max Refinement Iterations")
            gr.Markdown("---")
            gr.Markdown("### 👤 Candidate Profile")
            candidate_editor = gr.Textbox(
                label="candidate.yml",
                value=_load_candidate_yaml,
                lines=15,
                placeholder="Edit your contact, education, and certifications here...",
            )
            save_candidate_btn = gr.Button("💾 Save")
            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Clear Index", variant="stop")
            ingest_log_box = gr.Textbox(label="Log", lines=8, interactive=False)
            save_candidate_btn.click(fn=_save_candidate_yaml, inputs=candidate_editor, outputs=ingest_log_box)
            clear_btn.click(fn=clear_index, outputs=ingest_log_box)

        # --- Column 2: Upload + JD ---
        with gr.Column(scale=2):
            gr.Markdown("### 📁 Upload Resumes")
            with gr.Tab("Files"):
                upload_files = gr.File(
                    label="Select PDF/DOCX files",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                )
                ingest_files_btn = gr.Button("📤 Ingest Files")
            with gr.Tab("Folder"):
                upload_folder = gr.File(
                    label="Select a folder",
                    file_count="directory",
                )
                ingest_folder_btn = gr.Button("📤 Ingest Folder")
            ingest_files_btn.click(fn=ingest_resumes, inputs=upload_files, outputs=ingest_log_box)
            ingest_folder_btn.click(fn=ingest_resumes, inputs=upload_folder, outputs=ingest_log_box)

            gr.Markdown("---")
            gr.Markdown("### 📝 Job Description")
            with gr.Row():
                jd_url = gr.Textbox(
                    label="Job URL",
                    placeholder="https://...",
                    lines=1,
                    scale=4,
                )
                fetch_btn = gr.Button("🔗 Fetch", scale=1)
            jd_input = gr.Textbox(
                label="Job Description",
                lines=12,
                placeholder="Paste the full job description here or fetch from URL above...",
            )
            fetch_btn.click(fn=fetch_jd_from_url, inputs=jd_url, outputs=jd_input)
            generate_btn = gr.Button("🚀 Generate Resume", variant="primary")

        # --- Column 3: Generated Resume ---
        with gr.Column(scale=3):
            gr.Markdown("### 📄 Generated Resume")
            ats_feedback = gr.Textbox(label="📝 ATS Recommendation (for manual refinement)", lines=10, interactive=False)
            gen_log = gr.Textbox(label="Pipeline Log", lines=8, interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[jd_input, candidate_editor, min_score, max_iter, llm_provider, gemini_key, openai_key, claude_key],
        outputs=[gen_log, ats_feedback],
    )

if __name__ == "__main__":
    app.launch()
