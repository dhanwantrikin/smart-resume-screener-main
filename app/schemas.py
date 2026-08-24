from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class JobDescriptionCreate(BaseModel):
    title: str
    raw_text: str


class JobDescriptionOut(BaseModel):
    id: int
    title: str
    raw_text: str
    created_at: datetime

    class Config:
        from_attributes = True


class CandidateOut(BaseModel):
    id: int
    filename: str
    name: Optional[str]
    skills: Optional[str]
    experience: Optional[str]
    education: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class MatchRequest(BaseModel):
    candidate_id: int
    job_id: int


class MatchOut(BaseModel):
    id: int
    candidate_id: int
    job_id: int
    score: float
    justification: str
    created_at: datetime

    class Config:
        from_attributes = True


class ShortlistEntry(BaseModel):
    candidate_id: int
    candidate_name: Optional[str]
    filename: str
    score: float
    justification: str
