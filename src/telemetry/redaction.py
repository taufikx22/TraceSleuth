"""Telemetry Redaction and Privacy Guard.

Implements strict redaction of Personally Identifiable Information (PII)
and sensitive credentials (API keys, bearer tokens, secrets) before
telemetry ingestion, in adherence with OpenTelemetry GenAI security recommendations.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Union


# Regex patterns for sensitive identifiers and credentials
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
API_KEY_PATTERN = re.compile(r"(?i)(api[-_]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?")
BEARER_TOKEN_PATTERN = re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]{16,})", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def hash_content(content: Union[str, bytes, Dict[str, Any], List[Any]]) -> str:
    """Generate deterministic SHA-256 fingerprint for payloads to avoid storing raw text."""
    if isinstance(content, (dict, list)):
        import json
        payload = json.dumps(content, sort_keys=True).encode("utf-8")
    elif isinstance(content, str):
        payload = content.encode("utf-8")
    else:
        payload = bytes(content)
    return hashlib.sha256(payload).hexdigest()[:16]


class RedactionEngine:
    """Sanitizes text and dictionary attributes according to enterprise telemetry policies."""

    def __init__(
        self,
        capture_raw_content: Optional[bool] = None,
        redact_pii: bool = True,
        replacement: str = "[REDACTED]",
    ) -> None:
        if capture_raw_content is None:
            env_val = os.getenv("TRACESLEUTH_CAPTURE_RAW_CONTENT", "false").lower()
            self.capture_raw_content = env_val in ("true", "1", "yes")
        else:
            self.capture_raw_content = capture_raw_content
        self.redact_pii = redact_pii
        self.replacement = replacement

    def redact_text(self, text: str) -> str:
        """Mask emails, tokens, keys, phone numbers, and SSNs in strings."""
        if not text:
            return text
        sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        sanitized = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED_TOKEN]", sanitized)
        sanitized = API_KEY_PATTERN.sub(r"\1: [REDACTED_CREDENTIAL]", sanitized)
        sanitized = PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)
        sanitized = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", sanitized)
        sanitized = SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)
        return sanitized

    def sanitize_payload(self, data: Any, store_raw: bool = False) -> Any:
        """Sanitizes strings, lists, and dicts recursively.
        
        If raw content capture is disabled, text longer than 80 characters
        will be replaced with a SHA-256 fingerprint unless store_raw is explicitly True.
        """
        if data is None:
            return None
        if isinstance(data, str):
            clean = self.redact_text(data) if self.redact_pii else data
            if not self.capture_raw_content and not store_raw and len(clean) > 80:
                return f"hash:{hash_content(clean)} (len={len(clean)})"
            return clean
        if isinstance(data, dict):
            return {k: self.sanitize_payload(v, store_raw) for k, v in data.items()}
        if isinstance(data, list):
            return [self.sanitize_payload(x, store_raw) for x in data]
        return data


# Global singleton instance for easy telemetry middleware reuse
default_redactor = RedactionEngine()
