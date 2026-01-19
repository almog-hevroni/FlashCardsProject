"""
Database Engine and Session Management for FlashCards Project

This module provides database connection and session management that works
seamlessly across different database backends:
- SQLite (local development)
- PostgreSQL (cloud deployment)
- MySQL (alternative cloud option)

Configuration is driven by the DATABASE_URL environment variable.

Usage:
    # Context manager (recommended for scripts/CLI)
    from app.data.db_engine import get_db
    with get_db() as db:
        user = db.query(User).filter(User.user_id == "123").first()

    # FastAPI dependency injection
    from app.data.db_engine import get_db_session
    @app.get("/users/{user_id}")
    async def get_user(user_id: str, db: Session = Depends(get_db_session)):
        return db.query(User).filter(User.user_id == user_id).first()
"""

import os
import logging
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.data.models import Base

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE URL CONFIGURATION
# =============================================================================

def get_database_url() -> str:
    """
    Get the database URL from environment or use default SQLite.

    """
    url = os.getenv("DATABASE_URL", "")
    
    if not url:
        # Default to SQLite in the store directory
        store_path = os.getenv("VECTOR_STORE_PATH", "./store")
        Path(store_path).mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{store_path}/meta.sqlite"
    
    # Handle postgres:// → postgresql:// conversion for cloud providers
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return url


DATABASE_URL = get_database_url()


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================

def create_db_engine():
    """
    Create SQLAlchemy engine with appropriate settings for the database type.
    """
    url = DATABASE_URL
    
    if url.startswith("sqlite"):
        # SQLite-specific configuration
        engine = create_engine(
            url,
            connect_args={
                "check_same_thread": False,  # Allow multi-threaded access
                "timeout": 30,  # Wait up to 30s for locks
            },
            # Use StaticPool for SQLite to maintain single connection
            # This avoids "database is locked" errors in multi-threaded apps
            poolclass=StaticPool if "memory" in url else None,
            pool_pre_ping=True,
            echo=os.getenv("SQL_ECHO", "").lower() in ("true", "1", "yes"),
        )
        
        # Enable WAL mode for better concurrent read/write performance
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()
        
        logger.info("Created SQLite engine: %s", url.split("///")[-1])
    
    else:
        # PostgreSQL/MySQL configuration for cloud deployment
        engine = create_engine(
            url,
            pool_size=5,  # Maintain 5 connections in pool
            max_overflow=10,  # Allow up to 10 additional connections
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=300,  # Recycle connections every 5 minutes
            echo=os.getenv("SQL_ECHO", "").lower() in ("true", "1", "yes"),
        )
        
        logger.info("Created database engine: %s", url.split("@")[-1] if "@" in url else url)
    
    return engine


# Global engine instance
engine = create_db_engine()

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db() -> None:
    """
    Create all tables defined in models.py.

    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")


def drop_all_tables() -> None:
    """
    Drop all tables. Only for testing/development.
    """
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped!")


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Handles commit on success, rollback on exception, and cleanup.
    
    Usage:
        with get_db() as db:
            db.add(User(user_id="123", name="Test"))
            # Auto-commits on exit, rolls back on exception
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Does NOT auto-commit - leaves transaction control to the endpoint.
    
    Usage:
        @app.post("/users")
        async def create_user(user: UserCreate, db: Session = Depends(get_db_session)):
            db_user = User(**user.dict())
            db.add(db_user)
            db.commit()
            return db_user
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# HEALTH CHECK
# =============================================================================

def check_db_connection() -> dict:
    """
    Check database connectivity. Returns status dict.
    
    Usage:
        status = check_db_connection()
        # {"status": "healthy", "database": "postgresql", "latency_ms": 5.2}
    """
    import time
    
    try:
        start = time.time()
        with get_db() as db:
            db.execute(text("SELECT 1"))
        latency_ms = (time.time() - start) * 1000
        
        db_type = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql"
        
        return {
            "status": "healthy",
            "database": db_type,
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_table_counts() -> dict:
    """
    Get row counts for all tables. Useful for debugging/monitoring.
    """
    from app.data.models import (
        User, Document, Chunk, Exam, Topic, Card, 
        CardReview, Event, QuestionIndexEntry
    )
    
    counts = {}
    with get_db() as db:
        for model in [User, Document, Chunk, Exam, Topic, Card, CardReview, Event, QuestionIndexEntry]:
            try:
                counts[model.__tablename__] = db.query(model).count()
            except Exception:
                counts[model.__tablename__] = "error"
    
    return counts
