"""Deterministic graders for the Email Triage environment."""

from __future__ import annotations

import re
from typing import Dict, Tuple

try:
    from .tasks import EmailItem
except ImportError:
    from tasks import EmailItem


# Minimum and maximum scores to return (strictly between 0 and 1)
MIN_SCORE = 0.001
MAX_SCORE = 0.999

DEPARTMENT_ALIASES = {
    "billing": {"billing", "payments", "invoice", "finance"},
    "technical": {"technical", "engineering", "api", "bug"},
    "sales": {"sales", "pricing", "quote", "enterprise"},
    "hr": {"hr", "human resources", "people"},
    "general": {"general", "support", "info"},
}

RELATED_DEPARTMENTS = {
    "billing": {"sales"},
    "technical": {"general"},
    "sales": {"billing"},
    "hr": {"general"},
    "general": {"technical", "hr"},
}


def normalize_label(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def urgency_score(predicted: str | None, email: EmailItem) -> float:
    if normalize_label(predicted) == email.urgency:
        return MAX_SCORE
    return MIN_SCORE


def department_score(predicted: str | None, email: EmailItem) -> float:
    normalized = normalize_label(predicted)
    if not normalized:
        return MIN_SCORE
    if _has_compliance_risk(email) and normalized not in {"hr", "billing"}:
        return MIN_SCORE
    if normalized == email.department:
        return MAX_SCORE
    if normalized in RELATED_DEPARTMENTS.get(email.department, set()):
        return 0.5
    if normalized in DEPARTMENT_ALIASES.get(email.department, set()):
        return 0.5
    return MIN_SCORE


def summary_score(summary: str | None, email: EmailItem) -> float:
    if not summary:
        return MIN_SCORE
    summary_tokens = set(_tokenize(summary))
    reference_tokens = set(_tokenize(email.reference_summary))
    if not summary_tokens or not reference_tokens:
        return MIN_SCORE
    overlap = len(summary_tokens & reference_tokens) / max(1, len(reference_tokens))
    return max(MIN_SCORE, min(MAX_SCORE, overlap * 1.5))


def grade_task1(predicted_urgency: str | None, email: EmailItem) -> Tuple[float, Dict[str, float]]:
    score = urgency_score(predicted_urgency, email)
    return max(MIN_SCORE, score), {"urgency": score}


def grade_task2(predicted_department: str | None, email: EmailItem) -> Tuple[float, Dict[str, float]]:
    score = department_score(predicted_department, email)
    return max(MIN_SCORE, score), {"department": score}


def grade_task3(
    predicted_urgency: str | None,
    predicted_department: str | None,
    summary: str | None,
    email: EmailItem,
) -> Tuple[float, Dict[str, float]]:
    urgency = urgency_score(predicted_urgency, email)
    department = department_score(predicted_department, email)
    summary_value = summary_score(summary, email)
    score = 0.3 * urgency + 0.3 * department + 0.4 * summary_value
    return max(MIN_SCORE, score), {"urgency": urgency, "department": department, "summary": summary_value}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _has_compliance_risk(email: EmailItem) -> bool:
    text = f"{email.subject} {email.body}".lower()
    keywords = ["ssn", "passport", "credit card", "pii", "legal", "compliance"]
    return any(keyword in text for keyword in keywords)
