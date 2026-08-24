"""
LLM-based semantic matching between a candidate resume and a job description.

Uses Google Gemini API.
"""

import os
import re

from google import genai
from google.genai import types


MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-flash-latest")


MATCH_PROMPT_TEMPLATE = """You are an expert technical recruiter.

Compare the candidate resume with the job description.

Give a fit score from 1 to 10.

Return your response EXACTLY in this format:

SCORE: 7
JUSTIFICATION: The candidate has strong Python and cybersecurity knowledge. They also have relevant AWS and security-tool experience, but they have limited hands-on SIEM and incident-response experience.

Keep the justification to 2 or 3 sentences.

Candidate Resume

Skills:
<<SKILLS>>

Experience:
<<EXPERIENCE>>

Education:
<<EDUCATION>>

Resume Text:
<<RAW_TEXT>>

Job Description:
<<JOB_DESCRIPTION>>
"""


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return genai.Client(api_key=api_key)


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
            str(candidate.get("raw_text", ""))[:4000],
        )
        .replace(
            "<<JOB_DESCRIPTION>>",
            str(job_description)[:4000],
        )
    )


def compute_match(candidate: dict, job_description: str) -> dict:
    client = _get_client()

    prompt = build_prompt(candidate, job_description)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=300,
        ),
    )

    text = (response.text or "").strip()

    print("\n===== GEMINI RESPONSE =====")
    print(repr(text))
    print("===========================\n")

    # Extract score.
    score_match = re.search(
        r"SCORE\s*:\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )

    # Extract justification.
    justification_match = re.search(
        r"JUSTIFICATION\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if score_match:
        score = float(score_match.group(1))
        score = max(1.0, min(10.0, score))
    else:
        # Fallback: find any standalone score from 1-10.
        fallback_score = re.search(
            r"\b([1-9]|10)(?:\s*/\s*10)?\b",
            text,
        )

        if fallback_score:
            score = float(fallback_score.group(1))
        else:
            score = 0.0

    if justification_match:
        justification = justification_match.group(1).strip()
    else:
        # Fallback if the model didn't use the requested label.
        justification = text.strip()

    if not justification:
        justification = (
            "No justification was returned by the language model."
        )

    return {
        "score": score,
        "justification": justification,
    }