"""TraceSleuth Ingestion — Service Layer.

Orchestrates the pipeline: save trace to DB, run taxonomy classifier,
store resulting incidents with evidence.
"""

from __future__ import annotations

from typing import List, Optional

from src.analysis.taxonomy import FailureTaxonomyClassifier
from src.domain.incident import Incident
from src.domain.trace import TraceData
from src.storage.database import DatabaseManager
from src.storage.repository import TraceRepository


class IngestionService:
    """Processes incoming traces: persists to storage and classifies failures."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self.classifier = FailureTaxonomyClassifier()

    def ingest_trace(self, trace: TraceData, auto_classify: bool = True) -> List[Incident]:
        """Ingest a trace into the database and optionally classify failures.

        Args:
            trace: The TraceData to persist.
            auto_classify: If True, run the failure taxonomy classifier and store incidents.

        Returns:
            List of Incident objects created during classification (empty if not classified).
        """
        session = self.db_manager.get_session()
        try:
            repo = TraceRepository(session)
            repo.save_trace(trace)

            incidents: List[Incident] = []
            if auto_classify:
                incidents = self.classifier.classify(trace)
                for incident in incidents:
                    repo.save_incident(incident)

            return incidents
        finally:
            session.close()

    def classify_trace(self, trace_id: str) -> List[Incident]:
        """Run classification on an already-stored trace.

        Useful for re-analyzing traces with updated rules.
        """
        session = self.db_manager.get_session()
        try:
            repo = TraceRepository(session)
            trace = repo.get_trace(trace_id)
            if not trace:
                return []

            incidents = self.classifier.classify(trace)
            for incident in incidents:
                repo.save_incident(incident)

            return incidents
        finally:
            session.close()
