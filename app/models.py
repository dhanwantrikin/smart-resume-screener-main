from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database import Base


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    raw_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    matches = relationship("Match", back_populates="job")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    name = Column(String, nullable=True)
    raw_text = Column(Text)
    skills = Column(Text)        # comma-separated extracted skills
    experience = Column(Text)    # extracted experience snippets
    education = Column(Text)     # extracted education snippets
    created_at = Column(DateTime, default=datetime.utcnow)

    matches = relationship("Match", back_populates="candidate")


class Match(Base):
    """Stores the LLM-computed match score + justification for a
    (candidate, job_description) pair, so results don't need to be
    recomputed every time and can be listed/sorted later."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    job_id = Column(Integer, ForeignKey("job_descriptions.id"))
    score = Column(Float)
    justification = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="matches")
    job = relationship("JobDescription", back_populates="matches")
