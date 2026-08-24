"""
Database setup using SQLAlchemy + SQLite.
SQLite is used for simplicity (no external DB server needed to run/demo the project).
Swap SQLALCHEMY_DATABASE_URL for Postgres/MySQL in production.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = "sqlite:///./resume_screener.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
