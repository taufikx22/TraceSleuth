"""Integration Tests for Deployment & Containerization (Milestone 11).

Verifies Dockerfile configuration, Docker Compose service topology,
CI/CD GitHub Actions workflow, API healthcheck endpoints, and
fresh environment execution (spec Sections 23, 24, 38).
"""

from __future__ import annotations

from pathlib import Path
import yaml
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from scripts.verify_deployment import verify_fresh_deployment


@pytest.fixture
def client():
    return TestClient(app)


class TestDeploymentArtifacts:
    """Verifies existence, syntax, and contents of Milestone 11 deployment artifacts."""

    def test_dockerfile_syntax_and_structure(self):
        """Dockerfile must specify a slim Python base, expose port 8000, and define healthcheck."""
        dockerfile_path = Path("Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile must exist at repository root"

        content = dockerfile_path.read_text(encoding="utf-8")
        assert "FROM python:3.12-slim" in content
        assert "EXPOSE 8000" in content
        assert "HEALTHCHECK" in content
        assert "/health" in content
        assert "uvicorn" in content
        assert "src.api.main:app" in content

    def test_dockerignore_configuration(self):
        """.dockerignore must exclude virtualenvs, local databases, and protected files."""
        dockerignore_path = Path(".dockerignore")
        assert dockerignore_path.exists(), ".dockerignore must exist at repository root"

        content = dockerignore_path.read_text(encoding="utf-8")
        assert ".venv" in content
        assert "__pycache__" in content
        assert "5_TraceSleuth_Failure_Forensics_AI_Pipelines.md" in content

    def test_docker_compose_validity(self):
        """docker-compose.yml must define api and demo services with valid port mappings."""
        compose_path = Path("docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml must exist at repository root"

        parsed = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert "services" in parsed
        services = parsed["services"]

        assert "tracesleuth-api" in services
        api = services["tracesleuth-api"]
        assert "8000:8000" in api.get("ports", [])
        assert "healthcheck" in api

        assert "tracesleuth-demo" in services
        demo = services["tracesleuth-demo"]
        assert "demo" in demo.get("profiles", [])

    def test_makefile_developer_commands(self):
        """Makefile must provide targets for install, test, demo, run, and docker automation."""
        makefile_path = Path("Makefile")
        assert makefile_path.exists(), "Makefile must exist at repository root"

        content = makefile_path.read_text(encoding="utf-8")
        for target in ("install", "test", "demo", "run", "docker-build", "docker-up", "docker-down", "verify"):
            assert f"{target}:" in content, f"Makefile must define target: {target}"

    def test_github_actions_ci_workflow(self):
        """.github/workflows/ci.yml must test on Python 3.11/3.12 and validate Docker."""
        ci_path = Path(".github/workflows/ci.yml")
        assert ci_path.exists(), "GitHub Actions workflow must exist"

        parsed = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
        assert "jobs" in parsed
        assert "test" in parsed["jobs"]
        assert "docker-build" in parsed["jobs"]

        matrix = parsed["jobs"]["test"]["strategy"]["matrix"]["python-version"]
        assert "3.11" in matrix
        assert "3.12" in matrix

    def test_api_healthcheck_endpoint(self, client):
        """GET /health must return HTTP 200 and status=ok for container orchestration."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
        assert data.get("service") == "tracesleuth"

    def test_fresh_environment_verification_passes(self):
        """Milestone 11 exit criterion: Fresh environment can run a complete demo."""
        passed = verify_fresh_deployment()
        assert passed is True, "All fresh environment verification checkpoints must pass"
