# Smart Resume Screener

An LLM-powered tool that parses resumes, extracts structured candidate data,
and scores candidates against a job description using semantic matching.

## Objective

Intelligently parse resumes, extract skills, and match them with job
descriptions — surfacing a ranked, justified shortlist of candidates.

## Features

- Upload PDF resumes and automatically extract structured data: **skills**,
  **experience**, and **education**.
- Store a job description and compare it against any uploaded candidate.
- Use an LLM (Claude) to compute a **1–10 fit score** with a **written
  justification**, not just a keyword-match number.
- View a ranked shortlist of candidates per job, sorted by score.
- Simple HTML dashboard to drive the whole workflow without touching the API directly.

## Architecture

```
                ┌────────────────────┐
                │   Browser (HTML)   │
                │  frontend/index.html│
                └─────────┬──────────┘
                          │ fetch() calls
                          ▼
                ┌────────────────────┐
                │   FastAPI backend  │
                │     app/main.py    │
                └───┬─────────┬──────┘
                    │         │
      ┌─────────────┘         └─────────────┐
      ▼                                      ▼
┌───────────────────┐              ┌───────────────────┐
│ resume_parser.py   │              │  llm_matcher.py    │
│ - pdfplumber text   │              │ - builds prompt     │
│   extraction        │              │ - calls Claude API  │
│ - skill keyword      │             │ - parses JSON score │
│   matching           │             │   + justification   │
│ - section heuristics │             └───────────────────┘
│   for exp/education  │
└───────────────────┘
      │
      ▼
┌───────────────────┐
│   SQLite database   │
│ (SQLAlchemy models)  │
│ Candidate / Job /     │
│ Match tables          │
└───────────────────┘
```

**Flow:**
1. User pastes a job description → stored in `job_descriptions` table.
2. User uploads a PDF resume → `resume_parser.py` extracts raw text with
   `pdfplumber`, then derives skills (keyword matching), experience and
   education (section-header heuristics) → stored in `candidates` table.
3. User triggers a match → `llm_matcher.py` builds a prompt containing the
   candidate's extracted data + full resume text and the job description,
   sends it to the Claude API, and parses the returned JSON `{score,
   justification}` → stored in `matches` table.
4. `/shortlist/{job_id}` returns all matches for a job, ranked by score
   descending, for the dashboard to render.

## Tech Stack

- **Backend:** Python, FastAPI, SQLAlchemy, SQLite
- **Resume parsing:** pdfplumber
- **LLM:** Anthropic Claude API (`anthropic` Python SDK)
- **Frontend:** Plain HTML/CSS/JS (no build step required)

## LLM Usage & Prompt

The matching step sends the LLM the candidate's extracted skills/experience/
education plus the raw resume text, alongside the job description, and asks
for a structured JSON response. This is the exact prompt template used
(see `app/llm_matcher.py::MATCH_PROMPT_TEMPLATE`):

```
You are an expert technical recruiter.

Compare the following resume with this job description and rate the fit on
a scale of 1-10 with justification.

Resume (extracted skills, experience, education, and raw text):
---
Skills: {skills}
Experience: {experience}
Education: {education}

Full resume text:
{raw_text}
---

Job Description:
---
{job_description}
---

Respond ONLY with a JSON object in this exact format, no extra text:
{
  "score": <integer 1-10>,
  "justification": "<2-4 sentence explanation of the score, referencing
    specific matched or missing skills/experience>"
}
```

Design choices:
- **Structured output (JSON)** is enforced in the prompt so the score and
  justification can be parsed and stored reliably rather than scraped from
  free text.
- **Both extracted fields and raw text** are sent, so the LLM can reason
  over nuance the keyword extractor misses (e.g. a project description that
  implies a skill without naming it), while still benefiting from the
  structured extraction for a quick summary.
- The matcher has a **fallback path**: if the LLM ever returns malformed
  JSON, the raw text is preserved as the justification and the match isn't
  lost, rather than the request crashing.

## Project Structure

```
smart-resume-screener/
├── app/
│   ├── main.py            # FastAPI app & routes
│   ├── database.py        # SQLAlchemy engine/session setup
│   ├── models.py          # Candidate / JobDescription / Match tables
│   ├── schemas.py         # Pydantic request/response models
│   ├── resume_parser.py   # PDF text extraction + skill/section extraction
│   └── llm_matcher.py     # LLM prompt + Claude API call + JSON parsing
├── frontend/
│   └── index.html         # Minimal dashboard (upload, match, shortlist)
├── sample_data/
│   └── sample_job_description.txt
├── requirements.txt
├── .env.example
└── README.md
```

## Setup & Running Locally

1. **Clone the repo and enter it:**
   ```bash
   git clone https://github.com/<your-username>/smart-resume-screener.git
   cd smart-resume-screener
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your ANTHROPIC_API_KEY
   ```

4. **Run the backend:**
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be live at `http://localhost:8000`. Interactive API docs
   (Swagger UI) are auto-generated at `http://localhost:8000/docs`.

5. **Open the dashboard:**
   Just open `frontend/index.html` directly in your browser (double-click
   it, or use a simple static server). It talks to the API at
   `http://localhost:8000`.

## API Endpoints

| Method | Endpoint                  | Description                                  |
|--------|----------------------------|-----------------------------------------------|
| GET    | `/`                        | Health check                                  |
| POST   | `/job-descriptions`        | Create a job description                      |
| GET    | `/job-descriptions`        | List job descriptions                         |
| POST   | `/candidates/upload`       | Upload + parse a resume PDF                   |
| GET    | `/candidates`              | List parsed candidates                        |
| POST   | `/match`                   | Run LLM match for `{candidate_id, job_id}`    |
| GET    | `/shortlist/{job_id}`      | Ranked, justified shortlist for a job         |

## Demo Video

https://github.com/user-attachments/assets/90591007-3bb1-47d2-a4b2-31d57e974c14

## Notes / Possible Extensions

- Skill extraction currently uses a curated keyword list for
  reliability without extra ML dependencies; this could be swapped for an
  NLP model (spaCy NER) for more flexible extraction.
- Matching is done one candidate at a time; a batch endpoint could score
  every uploaded candidate against a job in one call.
- Authentication/multi-user support is out of scope for this assignment.
