"""
LLM-based semantic matching between a candidate resume and a job description.

Uses Google Gemini API.

The function compute_match() returns:
    {
        "score": float,
        "justification": str
    }
"""

import os
import json
import re

from google import genai
from google.genai import types


# Gemini model used for matching.
# You can override this with LLM_MODEL in .env.
MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-flash-latest")


MATCH_PROMPT_TEMPLATE = """You are an expert technical recruiter.

Compare the candidate resume against the job description.

Give the candidate a fit score from 1 to 10.

Requirements:
- Return ONLY valid JSON.
- Do not use markdown.
- Do not use code fences.
- Keep the justification between 2 and 4 sentences.
- Mention important matched skills and missing skills.

Return exactly this structure:

{
  "score": 7,
  "justification": "The candidate matches several important requirements."
}

Candidate resume:

Skills:
<<SKILLS>>

Experience:
<<EXPERIENCE>>

Education:
<<EDUCATION>>

Full resume text:
<<RAW_TEXT>>

Job description:
<<JOB_DESCRIPTION>>
"""


def _get_client():
    """
    Create a Gemini client using GEMINI_API_KEY from the environment.
    """

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    return genai.Client(api_key=api_key)


def build_prompt(candidate: dict, job_description: str) -> str:
    """
    Build the recruiter prompt without using str.format(),
    so the JSON braces in the prompt cannot cause KeyError.
    """

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


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from Gemini's response.

    Handles:
    - normal JSON
    - JSON inside markdown code fences
    - JSON surrounded by extra text
    """

    text = (text or "").strip()

    if not text:
        raise ValueError("Gemini returned an empty response.")

    # Remove markdown code fences.
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace("```", "").strip()

    # First attempt: entire response is JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Second attempt: find the JSON object inside surrounding text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        possible_json = cleaned[start:end + 1]

        try:
            return json.loads(possible_json)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from Gemini response: {text[:500]}"
    )


def compute_match(candidate: dict, job_description: str) -> dict:
    """
    Call Gemini and calculate the candidate's fit score.

    Returns:
        {
            "score": float,
            "justification": str
        }
    """

    client = _get_client()
    prompt = build_prompt(candidate, job_description)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=500,
            response_mime_type="application/json",
        ),
    )

    text = (response.text or "").strip()
    print("\n===== GEMINI RAW RESPONSE =====")
    print(repr(text))
    print("================================\n")

    try:
        parsed = _extract_json(text)

        # Extract score.
        score = float(parsed.get("score", 0))

        # Keep score within the required 1-10 range.
        score = max(1.0, min(10.0, score))

        # Extract explanation.
        justification = str(
            parsed.get(
                "justification",
                "No justification was returned by Gemini.",
            )
        ).strip()

        return {
            "score": score,
            "justification": justification,
        }

    except (ValueError, json.JSONDecodeError, TypeError):

        # Fallback 1:
        # Try to recover a score even if Gemini returned incomplete JSON.
        score_match = re.search(
            r'"score"\s*:\s*(\d+(?:\.\d+)?)',
            text,
        )

        if score_match:

            score = float(score_match.group(1))
            score = max(1.0, min(10.0, score))

            # Try to recover the justification.
            justification_match = re.search(
                r'"justification"\s*:\s*"(.+)',
                text,
                flags=re.DOTALL,
            )

            if justification_match:
                justification = (
                    justification_match.group(1)
                    .rstrip("}")
                    .rstrip('"')
                    .replace("\\n", " ")
                    .replace('\\"', '"')
                    .strip()
                )
            else:
                justification = (
                    "Gemini returned a fit score, but the "
                    "justification could not be parsed."
                )

            return {
                "score": score,
                "justification": justification,
            }

        # Fallback 2:
        # Nothing useful could be recovered.
        return {
            "score": 0.0,
            "justification": (
                "Gemini response could not be parsed. "
                f"Raw response: {text[:500]}"
            ),
        }