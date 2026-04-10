"""Streamlit UI for ATS Resume Generator."""

import re
import tempfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from src.rag_chain import generate_resume, score_resume, refine_resume, extract_job_info
from src.vector_store import search_resumes
from src.docx_writer import save_resume_docx
from src.config import RESUME_FOLDER

MAX_ITERATIONS = 2
MIN_ATS_SCORE = 90

st.set_page_config(page_title="ATS Resume Generator", page_icon="📄", layout="wide")
st.title("📄 ATS Resume Generator")
st.caption("Generate ATS-optimized resumes from job descriptions using your existing resumes as reference.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    min_score = st.slider("Minimum ATS Score", 50, 100, MIN_ATS_SCORE, step=5)
    max_iter = st.slider("Max Refinement Iterations", 1, 5, MAX_ITERATIONS)

# Input
job_description = st.text_area(
    "Paste the Job Description",
    height=300,
    placeholder="Paste the full job description here...",
)

if st.button("🚀 Generate Resume", type="primary", disabled=not job_description.strip()):
    progress = st.empty()
    status_log = st.container()

    # Step 1: Search existing resumes
    with status_log:
        st.subheader("Step 1: Searching existing resumes")
        with st.spinner("Running hybrid search..."):
            existing = search_resumes(job_description, k=1)

    if existing:
        r = existing[0]
        with status_log:
            st.write(f"Best match: `{Path(r['source']).name}`")
            with st.spinner("Scoring existing resume..."):
                score, _, _ = score_resume(r["content"], job_description)
            st.metric("Existing Resume ATS Score", f"{score}/100")

            if score >= min_score:
                st.success(f"✅ Existing resume scores {score}/100. No generation needed!")
                st.info(f"📄 Use: {r['source']}")
                st.stop()

    # Step 2: Generate
    with status_log:
        st.subheader("Step 2: Generating new resume")
        with st.spinner("Generating resume from matched resumes..."):
            resume, skills, experience = generate_resume(job_description)

    best_resume, best_score = resume, 0

    # Step 3: Score and refine loop
    for iteration in range(max_iter):
        with status_log:
            st.subheader(f"Step 3: Scoring & Refining (Iteration {iteration + 1})")
            with st.spinner("Scoring resume..."):
                score, feedback, parsed = score_resume(resume, job_description)

            col1, col2 = st.columns(2)
            col1.metric("ATS Score", f"{score}/100")
            col2.metric("Best Score So Far", f"{max(score, best_score)}/100")

            if score > best_score:
                best_score = score
                best_resume = resume

            if score >= min_score:
                st.success(f"✅ ATS score {score}/100 meets threshold!")
                break

            with st.expander("ATS Feedback", expanded=False):
                st.text(feedback)

            st.warning(f"⚠️ Score {score}/100 is below {min_score}. Refining...")
            if parsed["missing_keywords"]:
                st.write(f"**Missing keywords:** {parsed['missing_keywords']}")

            with st.spinner("Refining resume..."):
                resume = refine_resume(resume, job_description, parsed, skills, experience)
    else:
        with status_log:
            st.warning("⚠️ Max iterations reached.")

    # Step 4: Display and export
    st.divider()
    st.subheader(f"Final Resume (Best ATS Score: {best_score}/100)")
    st.text(best_resume)

    # Generate docx
    company, title = extract_job_info(job_description)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w-]", "_", f"{company}_{title}")[:60]
    filename = f"{safe_name}_{timestamp}.docx"

    # Save to temp file for download
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        save_resume_docx(best_resume, tmp.name)
        tmp_path = tmp.name

    # Also save to resume folder
    output_path = Path(RESUME_FOLDER) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_resume_docx(best_resume, str(output_path))

    with open(tmp_path, "rb") as f:
        st.download_button(
            label="📥 Download Resume (.docx)",
            data=f.read(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )

    st.success(f"📄 Also saved to: {output_path}")
