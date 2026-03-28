"""Baseline inference script using deterministic heuristics."""

from __future__ import annotations

from typing import Dict, List

import requests


TASK_EMAIL_MAP = {
    "task-urgency": ["email-001", "email-003", "email-005"],
    "task-routing": ["email-001", "email-002", "email-004"],
    "task-full-triage": ["email-002", "email-005", "email-006"],
}


def run_baseline(base_url: str = "http://localhost:7860") -> Dict[str, float]:
    scores: Dict[str, float] = {}

    for task_id in ["task-urgency", "task-routing", "task-full-triage"]:
        email_ids = TASK_EMAIL_MAP[task_id]
        per_email_scores = [
            _evaluate_task_on_email(base_url=base_url, task_id=task_id, email_id=email_id)
            for email_id in email_ids
        ]
        scores[task_id] = sum(per_email_scores) / len(per_email_scores)

    return scores


def run_baseline_detailed(base_url: str = "http://localhost:7860") -> Dict[str, Dict[str, object]]:
    detailed: Dict[str, Dict[str, object]] = {}
    for task_id, email_ids in TASK_EMAIL_MAP.items():
        per_email: Dict[str, float] = {}
        for email_id in email_ids:
            per_email[email_id] = _evaluate_task_on_email(base_url, task_id, email_id)
        detailed[task_id] = {
            "per_email_scores": per_email,
            "average": sum(per_email.values()) / len(per_email),
        }
    return detailed


def _evaluate_task_on_email(base_url: str, task_id: str, email_id: str) -> float:
    observation = _reset_env(base_url, task_id, email_id)
    action = _build_action(observation)
    _step_env(base_url, action)
    return _grade_env(base_url, task_id, action)


def _reset_env(base_url: str, task_id: str, email_id: str | None = None) -> Dict[str, object]:
    payload = {"task_id": task_id}
    if email_id:
        payload["email_id"] = email_id
    response = requests.post(f"{base_url}/reset", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["observation"]


def _step_env(base_url: str, action: Dict[str, object]) -> None:
    response = requests.post(f"{base_url}/step", json=action, timeout=30)
    response.raise_for_status()


def _grade_env(base_url: str, task_id: str, action: Dict[str, object]) -> float:
    response = requests.post(
        f"{base_url}/grader",
        json={"task_id": task_id, "action": action},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return float(payload.get("score", 0.0))


def _build_action(observation: Dict[str, object]) -> Dict[str, object]:
    urgency, department = _fallback_labels(observation)
    summary = _fallback_summary(observation)
    queue_position = _fallback_queue_position(observation, urgency)
    escalate = _fallback_escalation(observation, urgency)
    return {
        "task_id": observation["task_id"],
        "urgency": urgency,
        "department": department,
        "summary": summary,
        "queue_position": queue_position,
        "escalate": escalate,
        "notes": "heuristic baseline",
    }


def _fallback_labels(observation: Dict[str, object]) -> tuple[str, str]:
    subject = str(observation.get("subject", "")).lower()
    body = str(observation.get("body", "")).lower()
    minutes_to_breach = int(observation.get("minutes_to_breach", 9999))
    compliance_risk = bool(observation.get("compliance_risk", False))

    urgency = "normal"
    if compliance_risk or minutes_to_breach <= 30:
        urgency = "urgent"
    if "failed" in subject or "error" in body or "outage" in body:
        urgency = "urgent"
    if any(token in body for token in ["dark mode", "feature request", "eta"]):
        urgency = "low"
    if "eta" in body or "when" in body:
        urgency = "low" if "dark mode" in body else urgency

    department = "general"
    if compliance_risk:
        department = "hr"
    if "invoice" in subject or "payment" in body:
        department = "billing"
    elif "api" in subject or "error" in body or "500" in body:
        department = "technical"
    elif "pricing" in subject or "quote" in body:
        department = "sales"
    elif "w-9" in body or "w9" in body:
        department = "hr"

    return urgency, department


def _fallback_summary(observation: Dict[str, object]) -> str:
    subject = str(observation.get("subject", ""))
    return (
        f"We are prioritizing '{subject}' because it may affect service reliability, "
        "and the team will follow up shortly."
    )


def _fallback_queue_position(observation: Dict[str, object], urgency: str) -> int:
    if observation.get("compliance_risk"):
        return 1
    if urgency == "urgent":
        return 1
    if urgency == "normal":
        return 2
    return 3


def _fallback_escalation(observation: Dict[str, object], urgency: str) -> bool:
    minutes_to_breach = int(observation.get("minutes_to_breach", 9999))
    business_impact = int(observation.get("business_impact", 0))
    sender_tier = str(observation.get("sender_tier", "standard")).lower()
    if observation.get("compliance_risk"):
        return True
    if minutes_to_breach <= 45:
        return True
    if sender_tier in {"enterprise", "strategic"} and business_impact >= 85:
        return True
    return urgency == "urgent" and minutes_to_breach <= 60


if __name__ == "__main__":
    scores = run_baseline()
    detailed = run_baseline_detailed()
    print("Baseline scores:")
    for task_id, score in scores.items():
        print(f"- {task_id}: {score:.3f}")
    print("\nDetailed per-email scores:")
    for task_id, payload in detailed.items():
        print(f"- {task_id}: avg={payload['average']:.3f}, per_email={payload['per_email_scores']}")
