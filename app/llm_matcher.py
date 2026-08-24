"""
LLM-based semantic matching between a candidate resume and a job description.

Uses Google Gemini API with a deterministic fallback so the application
still produces a useful result if Gemini is temporarily unavailable.
"""

import os
import re

from google import genai
from google.genai import types


MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-flash-latest")


MATCH_PROMPT_TEMPLATE = """You are an expert technical recruiter.

Compare this candidate's resume against the job description.

Give a fit score from 1 to 10.

Your response MUST be exactly:

SCORE: <number>
JUSTIFICATION: <2-3 sentences>

Do not use JSON.
Do not use markdown.
Do not add any other sections.

Candidate Resume

Skills:
<<SKILLS>>

Experience:
<<EXPERIENCE>>

Education:
<<EDUCATION>>

Full Resume Text:
<<RAW_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>
"""


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=15000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            ),
        ),
    )


def build_prompt(candidate: dict, job_description: str) -> str:
    return (
        MATCH_PROMPT_TEMPLATE
        .replace(
            "<<SKILLS>>",
            str(candidate.get("skills", "")),
        )
        .replace(
            "<<EXPERIENCE>>",
            str(candidate.get("experience", "")),
        )
        .replace(
            "<<EDUCATION>>",
            str(candidate.get("education", "")),
        )
        .replace(
            "<<RAW_TEXT>>",
            str(candidate.get("raw_text", ""))[:3500],
        )
        .replace(
            "<<JOB_DESCRIPTION>>",
            str(job_description)[:3500],
        )
    )


def _fallback_match(candidate: dict, job_description: str) -> dict:
    """
    Deterministic fallback used when Gemini is unavailable or
    returns an unusable response.
    """

    resume_text = " ".join(
        [
            str(candidate.get("skills", "")),
            str(candidate.get("experience", "")),
            str(candidate.get("education", "")),
            str(candidate.get("raw_text", "")),
        ]
    ).lower()

    jd_text = str(job_description).lower()

    important_skills = [
        "cybersecurity",
        "network security",
        "tcp/ip",
        "linux",
        "python",
        "scripting",
        "vulnerability",
        "siem",
        "security monitoring",
        "incident response",
        "cloud security",
        "aws",
        "azure",
        "splunk",
        "firewall",
        "yara",
        "virustotal",
        "encryption",
    ]

    resume_matches = [
        skill for skill in important_skills
        if skill in resume_text
    ]

    jd_requirements = [
        skill for skill in important_skills
        if skill in jd_text
    ]

    if jd_requirements:
        matched = [
            skill
            for skill in jd_requirements
            if skill in resume_text
        ]
        ratio = len(matched) / len(jd_requirements)
    else:
        matched = resume_matches
        ratio = min(len(matched) / 8, 1.0)

    # Produce a sensible 1-10 score.
    score = round(3 + (ratio * 6))
    score = max(1, min(10, score))

    missing = [
        skill
        for skill in jd_requirements
        if skill not in resume_text
    ]

    matched_text = ", ".join(matched[:6]) if matched else "general technical skills"
    missing_text = ", ".join(missing[:5]) if missing else "no major listed requirements"

    justification = (
        f"The candidate matches several relevant areas including "
        f"{matched_text}. "
        f"However, the resume shows limited or missing evidence for "
        f"{missing_text}. "
        f"Overall, the profile shows a {score}/10 fit for this role."
    )

    return {
        "score": float(score),
        "justification": justification,
    }


def _parse_gemini_response(text: str):
    """
    Parse a Gemini response in several possible formats.
    Returns (score, justification) or (None, None).
    """

    if not text:
        return None, None

    cleaned = text.strip()

    # Remove accidental code fences.
    cleaned = re.sub(
        r"```(?:text|json)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("```", "").strip()

    score = None
    justification = None

    # Preferred format:
    # SCORE: 7
    score_match = re.search(
        r"\bSCORE\s*:\s*(10|[1-9](?:\.\d+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if score_match:
        score = float(score_match.group(1))

    # JSON fallback:
    if score is None:
        score_match = re.search(
            r'"score"\s*:\s*(10|[1-9](?:\.\d+)?)',
            cleaned,
            flags=re.IGNORECASE,
        )

        if score_match:
            score = float(score_match.group(1))

    # Justification:
    justification_match = re.search(
        r"\bJUSTIFICATION\s*:\s*(.+)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if justification_match:
        justification = justification_match.group(1).strip()

    # JSON fallback:
    if not justification:
        justification_match = re.search(
            r'"justification"\s*:\s*"(.+?)"',
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if justification_match:
            justification = (
                justification_match.group(1)
                .replace("\\n", " ")
                .replace('\\"', '"')
                .strip()
            )

    # If Gemini only gives an explanation, keep it if useful.
    if score is not None and not justification:
        remaining = re.sub(
            r"\bSCORE\s*:\s*(10|[1-9](?:\.\d+)?)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        remaining = re.sub(
            r"\bJUSTIFICATION\s*:\s*",
            "",
            remaining,
            flags=re.IGNORECASE,
        )
        remaining = remaining.strip(" \n:-{}\"")

        if len(remaining) > 20:
            justification = remaining

    if score is not None and justification:
        score = max(1.0, min(10.0, score))
        return score, justification

    return None, None


def compute_match(candidate: dict, job_description: str) -> dict:
    """
    Try Gemini first.

    If Gemini fails, times out, or returns malformed output,
    use the deterministic fallback.
    """

    client = _get_client()

    if client is not None:
        try:
            prompt = build_prompt(candidate, job_description)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                ),
            )

            text = (response.text or "").strip()

            print("\n===== GEMINI RESPONSE =====")
            print(repr(text))
            print("===========================\n")

            score, justification = _parse_gemini_response(text)

            if score is not None and justification:
                return {
                    "score": score,
                    "justification": justification,
                }

        except Exception as exc:
            print("\n===== GEMINI ERROR =====")
            print(str(exc))
            print("========================\n")

    # Guaranteed fallback.
    return _fallback_match(candidate, job_description)