"""
Smart Resume Screener - FastAPI backend

Endpoints:
  POST /job-descriptions          -> create a job description
  GET  /job-descriptions          -> list job descriptions
  POST /candidates/upload         -> upload + parse a resume PDF
  GET  /candidates                -> list parsed candidates
  POST /match                     -> run LLM match for a candidate + job
  GET  /shortlist/{job_id}        -> shortlisted candidates for a job,
                                      ranked by score, with justification
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from dotenv import load_dotenv
from typing import List

from app.database import Base, engine, get_db
from app import models, schemas
from app.resume_parser import parse_resume
from app.llm_matcher import compute_match

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Resume Screener")

# Allow the simple static frontend (opened via file:// or a dev server) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "smart-resume-screener"}


# ---------- Job Descriptions ----------

@app.post("/job-descriptions", response_model=schemas.JobDescriptionOut)
def create_job_description(job: schemas.JobDescriptionCreate, db: Session = Depends(get_db)):
    jd = models.JobDescription(title=job.title, raw_text=job.raw_text)
    db.add(jd)
    db.commit()
    db.refresh(jd)
    return jd


@app.get("/job-descriptions", response_model=List[schemas.JobDescriptionOut])
def list_job_descriptions(db: Session = Depends(get_db)):
    return db.query(models.JobDescription).order_by(desc(models.JobDescription.id)).all()


# ---------- Candidates ----------

@app.post("/candidates/upload", response_model=schemas.CandidateOut)
async def upload_candidate(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    file_bytes = await file.read()
    parsed = parse_resume(file_bytes, file.filename)

    candidate = models.Candidate(
        filename=parsed["filename"],
        name=parsed["name"],
        raw_text=parsed["raw_text"],
        skills=parsed["skills"],
        experience=parsed["experience"],
        education=parsed["education"],
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)
    return candidate


@app.get("/candidates", response_model=List[schemas.CandidateOut])
def list_candidates(db: Session = Depends(get_db)):
    return db.query(models.Candidate).order_by(desc(models.Candidate.id)).all()


# ---------- Matching ----------

@app.post("/match", response_model=schemas.MatchOut)
def run_match(req: schemas.MatchRequest, db: Session = Depends(get_db)):
    candidate = db.query(models.Candidate).get(req.candidate_id)
    job = db.query(models.JobDescription).get(req.job_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found.")
    if not job:
        raise HTTPException(status_code=404, detail="Job description not found.")

    result = compute_match(
        candidate={
            "skills": candidate.skills,
            "experience": candidate.experience,
            "education": candidate.education,
            "raw_text": candidate.raw_text,
        },
        job_description=job.raw_text,
    )

    match = models.Match(
        candidate_id=candidate.id,
        job_id=job.id,
        score=result["score"],
        justification=result["justification"],
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@app.get("/shortlist/{job_id}", response_model=List[schemas.ShortlistEntry])
def get_shortlist(job_id: int, db: Session = Depends(get_db)):
    """Return candidates matched against this job, ranked by score descending."""
    results = (
        db.query(models.Match, models.Candidate)
        .join(models.Candidate, models.Match.candidate_id == models.Candidate.id)
        .filter(models.Match.job_id == job_id)
        .order_by(desc(models.Match.score))
        .all()
    )
    return [
        schemas.ShortlistEntry(
            candidate_id=candidate.id,
            candidate_name=candidate.name,
            filename=candidate.filename,
            score=match.score,
            justification=match.justification,
        )
        for match, candidate in results
    ]
