"""
Resume parsing utilities.

Responsibilities (per assignment scope):
  - Accept a PDF resume
  - Extract raw text
  - Extract structured data: skills, experience, education

We keep extraction dependency-light (no spaCy/NLTK model downloads needed)
by combining:
  1. A curated skills keyword list (easy to extend) for skill extraction.
  2. Section-header regex matching for experience/education blocks.

This keeps the project runnable offline with just `pip install -r requirements.txt`.
"""

import io
import re
import pdfplumber

# A reasonably broad keyword list covering common tech-role skills.
# Extend this list for your target job domains.
SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "sqlite", "redis",
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring", "spring boot", ".net",
    "html", "css", "tailwind", "bootstrap",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "git", "github", "gitlab", "jenkins",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "rest api", "graphql", "microservices", "system design",
    "agile", "scrum", "jira",
    "linux", "bash", "shell scripting",
    "data structures", "algorithms", "oop",
    "llm", "langchain", "openai", "anthropic", "prompt engineering",
]

EDUCATION_HEADERS = r"(education|academic background|qualifications)"
EXPERIENCE_HEADERS = r"(experience|work history|employment|professional experience)"
SECTION_HEADER_RE = re.compile(
    r"^\s*[A-Z][A-Za-z ]{2,40}\s*$", re.MULTILINE
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF resume using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_skills(text: str) -> list[str]:
    """Match resume text against the skills keyword list (case-insensitive)."""
    lower_text = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        # word-boundary-ish match so "go" doesn't match inside "google"
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, lower_text):
            found.add(skill)
    return sorted(found)


def _extract_section(text: str, header_pattern: str) -> str:
    """
    Heuristic section extractor: finds a line matching the header pattern,
    then returns text until the next ALL-CAPS-ish / titled section header
    or end of document.
    """
    lines = text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        if re.search(header_pattern, line, re.IGNORECASE) and len(line.strip()) < 60:
            start_idx = i + 1
            break
    if start_idx is None:
        return ""

    collected = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        # Stop if we hit what looks like the next section header
        if stripped and SECTION_HEADER_RE.match(stripped) and len(stripped.split()) <= 4:
            if not re.search(header_pattern, stripped, re.IGNORECASE):
                break
        collected.append(line)
        if len(collected) > 25:  # safety cap
            break
    return "\n".join(collected).strip()


def extract_experience(text: str) -> str:
    return _extract_section(text, EXPERIENCE_HEADERS)


def extract_education(text: str) -> str:
    return _extract_section(text, EDUCATION_HEADERS)


def guess_candidate_name(text: str, filename: str) -> str:
    """Very light heuristic: assume the first non-empty line that looks like
    a name (2-4 capitalized words, no digits) is the candidate's name.
    Falls back to the filename."""
    for line in text.split("\n")[:5]:
        stripped = line.strip()
        words = stripped.split()
        if 1 < len(words) <= 4 and all(w[0:1].isupper() for w in words) \
                and not any(ch.isdigit() for ch in stripped):
            return stripped
    return filename.rsplit(".", 1)[0]


def parse_resume(file_bytes: bytes, filename: str) -> dict:
    text = extract_text_from_pdf(file_bytes)
    return {
        "filename": filename,
        "name": guess_candidate_name(text, filename),
        "raw_text": text,
        "skills": ", ".join(extract_skills(text)),
        "experience": extract_experience(text),
        "education": extract_education(text),
    }
