"""TraceSleuth API — FastAPI Application.

Main application entrypoint with CORS, lifespan management,
and route mounting for trace explorer and incident management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_incidents import router as incidents_router
from src.api.routes_traces import router as traces_router
from src.storage.database import get_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    db = get_database()
    db.init_db()
    yield


app = FastAPI(
    title="TraceSleuth",
    description="Failure Forensics and Trace Explorer for AI Pipelines",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(traces_router)
app.include_router(incidents_router)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "tracesleuth"}
