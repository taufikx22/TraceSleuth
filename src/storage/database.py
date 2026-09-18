"""TraceSleuth Storage — Database Configuration.

SQLAlchemy engine, session factory, and table initialization.
Uses SQLite for local development; PostgreSQL-compatible via standard SQLAlchemy Core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.storage.models import Base

# Default SQLite location — stored alongside the project data
_DEFAULT_DB_DIR = Path("./data")
_DEFAULT_DB_URL = f"sqlite:///{_DEFAULT_DB_DIR / 'tracesleuth.db'}"


class DatabaseManager:
    """Manages SQLAlchemy engine and session lifecycle."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or _DEFAULT_DB_URL

        # Ensure the data directory exists for SQLite
        if self.database_url.startswith("sqlite"):
            db_path = self.database_url.replace("sqlite:///", "")
            if db_path not in (":memory:", "") and not db_path.startswith(":memory:"):
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        engine_kwargs = {}
        if "sqlite" in self.database_url:
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in self.database_url:
                from sqlalchemy.pool import StaticPool
                engine_kwargs["poolclass"] = StaticPool

        self.engine = create_engine(
            self.database_url,
            echo=False,
            **engine_kwargs,
        )

        # Enable foreign key enforcement for SQLite
        if "sqlite" in self.database_url:

            @event.listens_for(self.engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init_db(self) -> None:
        """Create all tables defined in the ORM models."""
        Base.metadata.create_all(bind=self.engine)

    def drop_db(self) -> None:
        """Drop all tables — use for testing only."""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Create a new database session."""
        return self.SessionLocal()


# Lazy-initialized default singleton
_default_db: Optional[DatabaseManager] = None


def get_database(database_url: Optional[str] = None) -> DatabaseManager:
    """Get or create the default DatabaseManager singleton."""
    global _default_db
    if _default_db is None:
        _default_db = DatabaseManager(database_url=database_url)
        _default_db.init_db()
    return _default_db


def reset_database() -> None:
    """Reset the singleton — used in testing."""
    global _default_db
    _default_db = None
