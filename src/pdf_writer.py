"""Convert resume text to PDF."""

import markdown
from weasyprint import HTML

CSS = """
body {
    font-family: 'Helvetica', 'Arial', sans-serif;
    font-size: 11pt;
    line-height: 1.4;
    margin: 40px 50px;
    color: #222;
}
h1 { font-size: 18pt; margin-bottom: 2px; }
h2 { font-size: 13pt; border-bottom: 1px solid #999; padding-bottom: 2px; margin-top: 14px; }
h3 { font-size: 11pt; margin-bottom: 2px; }
ul { margin: 4px 0; padding-left: 20px; }
li { margin-bottom: 2px; }
p { margin: 4px 0; }
"""


def save_resume_pdf(resume_text: str, output_path: str):
    """Convert markdown-formatted resume to PDF."""
    html_body = markdown.markdown(resume_text)
    full_html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"
    HTML(string=full_html).write_pdf(output_path)
