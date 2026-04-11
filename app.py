"""Gradio UI for ATS Resume Generator."""

import re
import tempfile
from datetime import datetime
from pathlib import Path
import gradio as gr
from src.config import RESUME_FOLDER

MAX_ITERATIONS = 2
MIN_ATS_SCORE = 90


def ingest_resumes(files):
    """Ingest uploaded resume files into the vector store."""
    if not files:
        return "No files uploaded. Please select files or a folder first."

    from src.vector_store import index_resume, ensure_index_exists, resume_exists
    from src.document_loader import load_single_resume, chunk_documents
    from src.vector_store import ensure_search_pipeline
    from ingest import extract_skills_experience

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
        skills, experience = extract_skills_experience(full_text)
        index_resume(source=name, skills=skills, experience=experience, chunks=chunks, original_path=original)
        ingested += 1
        log.append(f"✅ Indexed {name} ({len(chunks)} chunks)")

    log.append(f"\n📊 {ingested} new, {skipped} skipped.")
    return "\n".join(log)


def generate(job_description, min_score, max_iter, llm_choice, gemini_api_key, openai_api_key):
    """Run the full pipeline: search, generate, score, refine, export."""
    if not job_description.strip():
        return "No job description provided.", None, "", ""

    from src.rag_chain import set_llm_provider
    from src.vector_store import ensure_search_pipeline

    # Set LLM provider
    if llm_choice == "Gemini (API)":
        if not gemini_api_key:
            return "❌ Please enter your Gemini API key.", None, "", ""
        set_llm_provider("gemini", gemini_api_key)
    elif llm_choice == "ChatGPT (API)":
        if not openai_api_key:
            return "❌ Please enter your OpenAI API key.", None, "", ""
        set_llm_provider("openai", openai_api_key)
    else:
        set_llm_provider("ollama")

    ensure_search_pipeline()
    log = []

    try:
        return _run_pipeline(job_description, min_score, max_iter, log)
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key" in error_msg:
            return "❌ Invalid API key. Please check and try again.", None, "", ""
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return "❌ API rate limit reached. Please wait and try again.", None, "", ""
        return f"❌ Error: {error_msg}", None, "", ""


def _run_pipeline(job_description, min_score, max_iter, log):
    from src.rag_chain import generate_resume, score_resume, refine_resume, extract_job_info
    from src.vector_store import search_resumes
    from src.docx_writer import save_resume_docx, resume_to_html
    from src.config import RESUME_FOLDER

    # Step 1: Search once, reuse results
    log.append("🔍 Searching existing resumes...")
    results = search_resumes(job_description, k=3)
    for r in results:
        log.append(f"  • {Path(r['source']).name}")

    # Step 2: Score best match
    if results:
        r = results[0]
        log.append(f"\n📊 Scoring best match: {Path(r['source']).name}")
        score, feedback, _ = score_resume(r["content"], job_description)
        log.append(f"  ATS Score: {score}/100")

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

            import shutil, tempfile as tf
            download_path = None
            if full_path:
                tmp = tf.NamedTemporaryFile(suffix=full_path.suffix, delete=False)
                shutil.copy2(str(full_path), tmp.name)
                download_path = tmp.name

            return "\n".join(log), download_path, "", feedback

    # Step 3: Generate using search results directly (no second search)
    log.append("\n⚙️ Generating resume...")
    resume, skills, experience = generate_resume(job_description, results=results)
    best_resume, best_score, best_feedback = resume, 0, ""

    # Step 4: Score and refine
    for iteration in range(int(max_iter)):
        log.append(f"\n--- Iteration {iteration + 1} ---")
        score, feedback, parsed = score_resume(resume, job_description)
        log.append(f"ATS Score: {score}/100")

        if score > best_score:
            best_score = score
            best_resume = resume
            best_feedback = feedback

        if score >= min_score:
            log.append("✅ Score meets threshold!")
            break

        log.append(f"Missing: {parsed['missing_keywords']}")
        log.append("Refining...")
        resume = refine_resume(resume, job_description, parsed, skills, experience)
    else:
        log.append("\n⚠️ Max iterations reached.")

    log.append(f"\n📊 Best ATS Score: {best_score}/100")

    # Step 4: Export
    company, title = extract_job_info(job_description)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w-]", "_", f"{company}_{title}")[:60]
    filename = f"{safe_name}_{timestamp}.docx"

    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    save_resume_docx(best_resume, tmp.name)

    output_path = Path(RESUME_FOLDER) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_resume_docx(best_resume, str(output_path))

    log.append(f"💾 Saved to: {output_path}")

    return "\n".join(log), tmp.name, resume_to_html(best_resume), best_feedback


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
            llm_provider = gr.Radio(
                choices=["Ollama (Local)", "Gemini (API)", "ChatGPT (API)"],
                value="Ollama (Local)",
                label="LLM Provider",
            )
            gemini_key = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="Paste your API key...",
                visible=False,
            )
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Paste your API key...",
                visible=False,
            )
            llm_provider.change(
                fn=lambda p: (gr.update(visible=p == "Gemini (API)"), gr.update(visible=p == "ChatGPT (API)")),
                inputs=llm_provider,
                outputs=[gemini_key, openai_key],
            )
            min_score = gr.Slider(50, 100, value=MIN_ATS_SCORE, step=5, label="Minimum ATS Score")
            max_iter = gr.Slider(1, 5, value=MAX_ITERATIONS, step=1, label="Max Refinement Iterations")
            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Clear Index", variant="stop")
            ingest_log_box = gr.Textbox(label="Log", lines=8, interactive=False)
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
            jd_input = gr.Textbox(
                label="📝 Paste Job Description",
                lines=12,
                placeholder="Paste the full job description here...",
            )
            generate_btn = gr.Button("🚀 Generate Resume", variant="primary")

        # --- Column 3: Generated Resume ---
        with gr.Column(scale=3):
            gr.Markdown("### 📄 Generated Resume")
            resume_output = gr.HTML(label="Resume Preview")
            download_file = gr.File(label="📥 Download", height=50)
            ats_feedback = gr.Textbox(label="📝 ATS Recommendation (for manual refinement)", lines=10, interactive=False)
            gen_log = gr.Textbox(label="Pipeline Log", lines=8, interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[jd_input, min_score, max_iter, llm_provider, gemini_key, openai_key],
        outputs=[gen_log, download_file, resume_output, ats_feedback],
    )

if __name__ == "__main__":
    app.launch()
