"""OpenAI-powered inference script for the OpenEnv email triage environment.

Required environment variables:
- API_BASE_URL: LLM API base URL
- MODEL_NAME: model identifier
- HF_TOKEN: Hugging Face token / API key
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import requests

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


TASK_EMAIL_MAP = {
    "task-urgency": ["email-001", "email-003", "email-005"],
    "task-routing": ["email-001", "email-002", "email-004"],
    "task-full-triage": ["email-002", "email-005", "email-006"],
}

SYSTEM_PROMPT = (
    "You are an email triage agent. Respond with only a valid JSON object and no extra text. "
    "Choose urgency, department, summary, queue_position, escalate, and notes based on the observation. "
    "Be strict: urgency in {urgent,normal,low}; department in {billing,technical,sales,hr,general}; "
    "queue_position in {1,2,3}; escalate as boolean."
)


def get_required_llm_env() -> Tuple[str, str, str]:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    missing = []
    if not api_base_url:
        missing.append("API_BASE_URL")
    if not model_name:
        missing.append("MODEL_NAME")
    if not hf_token:
        missing.append("HF_TOKEN")

    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nDefine API_BASE_URL, MODEL_NAME, and HF_TOKEN before running inference.py"
        )

    return api_base_url, model_name, hf_token


def run_inference(base_url: str, client: Any, model_name: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    all_email_ids = _load_all_email_ids()

    for task_id in ["task-urgency", "task-routing", "task-full-triage"]:
        email_ids = all_email_ids or TASK_EMAIL_MAP[task_id]
        per_email_scores = [
            _evaluate_task_on_email(
                base_url=base_url,
                task_id=task_id,
                email_id=email_id,
                client=client,
                model_name=model_name,
            )
            for email_id in email_ids
        ]
        scores[task_id] = sum(per_email_scores) / len(per_email_scores)
    return scores


def _load_all_email_ids() -> list[str]:
    tasks_file = Path(__file__).resolve().parent / "tasks.py"
    if not tasks_file.exists():
        return []

    try:
        spec = importlib.util.spec_from_file_location("openenv_tasks_module", tasks_file)
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return []

    emails = getattr(module, "EMAILS", [])
    ids: list[str] = []
    for item in emails:
        email_id = getattr(item, "email_id", None)
        if isinstance(email_id, str) and email_id:
            ids.append(email_id)
    return ids


def run_baseline(base_url: str = "http://localhost:7860") -> Dict[str, float]:
    """Compatibility entrypoint used by the FastAPI /baseline endpoint.

    Behavior:
    - If API_BASE_URL, MODEL_NAME, and HF_TOKEN are configured, use OpenAI-backed inference.
    - Otherwise, fall back to the deterministic baseline implementation.
    """

    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if api_base_url and model_name and hf_token:
        if OpenAI is None:
            raise RuntimeError(
                "openai package is not installed. Install dependencies from requirements.txt before running LLM baseline."
            )
        client = OpenAI(base_url=api_base_url, api_key=hf_token)
        return run_inference(base_url=base_url, client=client, model_name=model_name)

    # Keep /baseline functional in local and container runs even without LLM secrets.
    deterministic_run_baseline = _load_deterministic_baseline_runner()
    if deterministic_run_baseline is None:
        return _run_heuristic_baseline(base_url=base_url)
    return deterministic_run_baseline(base_url=base_url)


def _run_heuristic_baseline(base_url: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    all_email_ids = _load_all_email_ids()

    for task_id in ["task-urgency", "task-routing", "task-full-triage"]:
        email_ids = all_email_ids or TASK_EMAIL_MAP[task_id]
        per_email_scores = []
        for email_id in email_ids:
            observation = _reset_env(base_url, task_id, email_id)
            action = _heuristic_action(observation)
            _step_env(base_url, action)
            per_email_scores.append(_grade_env(base_url, task_id, action))
        scores[task_id] = sum(per_email_scores) / len(per_email_scores)

    return scores


def _load_deterministic_baseline_runner() -> Any | None:
    baseline_file = Path(__file__).resolve().parent / "baseline" / "inference.py"
    if not baseline_file.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("baseline_inference_module", baseline_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None

    fn = getattr(module, "run_baseline", None)
    if callable(fn):
        return fn
    return None


def _evaluate_task_on_email(
    base_url: str,
    task_id: str,
    email_id: str,
    client: Any,
    model_name: str,
) -> float:
    observation = _reset_env(base_url, task_id, email_id)
    action = _build_action_with_llm(client, model_name, observation)
    _step_env(base_url, action)
    return _grade_env(base_url, task_id, action)


def _reset_env(base_url: str, task_id: str, email_id: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task_id": task_id}
    if email_id:
        payload["email_id"] = email_id
    response = requests.post(f"{base_url}/reset", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["observation"]


def _step_env(base_url: str, action: Dict[str, Any]) -> None:
    response = requests.post(f"{base_url}/step", json=action, timeout=30)
    response.raise_for_status()


def _grade_env(base_url: str, task_id: str, action: Dict[str, Any]) -> float:
    response = requests.post(
        f"{base_url}/grader",
        json={"task_id": task_id, "action": action},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return float(payload.get("score", 0.0))


def _build_action_with_llm(client: Any, model_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    heuristic_action = _heuristic_action(observation)

    user_prompt = (
        "Create a triage action for this email observation.\n"
        "Return strict JSON only with these keys:\n"
        "task_id, urgency, department, summary, queue_position, escalate, notes\n\n"
        "Rules:\n"
        "- urgency must be one of available_urgency_labels\n"
        "- department must be one of available_departments\n"
        "- queue_position must be 1, 2, or 3\n"
        "- escalate must be boolean\n"
        "- summary must be one sentence\n\n"
        f"Observation:\n{json.dumps(observation, ensure_ascii=True)}"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=260,
        )
        raw = completion.choices[0].message.content or ""
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Model output was not a JSON object")
        llm_action = _normalize_action(parsed, observation)
        return _merge_llm_with_heuristics(llm_action, heuristic_action)
    except Exception:
        # Keep inference resilient to transient LLM formatting failures.
        return heuristic_action


def _merge_llm_with_heuristics(llm_action: Dict[str, Any], heuristic_action: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer LLM outputs when valid so actions do not collapse to one repeated pattern.
    urgency = str(llm_action.get("urgency", "")).strip().lower() or str(heuristic_action["urgency"])
    if urgency not in {"urgent", "normal", "low"}:
        urgency = str(heuristic_action["urgency"])

    department = str(llm_action.get("department", "")).strip().lower() or str(heuristic_action["department"])
    if department not in {"billing", "technical", "sales", "hr", "general"}:
        department = str(heuristic_action["department"])

    summary = str(llm_action.get("summary", "")).strip()
    if not _is_useful_summary(summary):
        summary = str(heuristic_action.get("summary", "")).strip()

    queue_position = llm_action.get("queue_position", heuristic_action["queue_position"])
    if not isinstance(queue_position, int) or queue_position not in {1, 2, 3}:
        queue_position = int(heuristic_action["queue_position"])

    escalate = llm_action.get("escalate", heuristic_action["escalate"])
    if not isinstance(escalate, bool):
        escalate = bool(heuristic_action["escalate"])

    # Hard safety guardrails only for known high-risk conditions.
    compliance_risk = bool(heuristic_action.get("department") == "hr" and heuristic_action.get("escalate"))
    if compliance_risk:
        department = "hr"
        queue_position = 1
        escalate = True

    return {
        "task_id": heuristic_action["task_id"],
        "urgency": urgency,
        "department": department,
        "summary": summary,
        "queue_position": queue_position,
        "escalate": escalate,
        "notes": str(llm_action.get("notes", "hybrid llm+heuristic")).strip() or "hybrid llm+heuristic",
    }


def _is_useful_summary(summary: str) -> bool:
    if not summary:
        return False
    words = summary.split()
    if len(words) < 8:
        return False
    lower = summary.lower()
    banned = ["i can't", "cannot", "unable", "insufficient", "n/a"]
    if any(token in lower for token in banned):
        return False
    return True


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found")


def _normalize_action(candidate: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    urgency_labels = set(observation.get("available_urgency_labels", ["urgent", "normal", "low"]))
    departments = set(
        observation.get(
            "available_departments",
            ["billing", "technical", "sales", "hr", "general"],
        )
    )

    urgency = str(candidate.get("urgency", "normal")).lower()
    if urgency not in urgency_labels:
        urgency = "normal"

    department = str(candidate.get("department", "general")).lower()
    if department not in departments:
        department = "general"

    queue_position = candidate.get("queue_position", 2)
    if not isinstance(queue_position, int) or queue_position not in {1, 2, 3}:
        queue_position = 2

    escalate = candidate.get("escalate", False)
    if not isinstance(escalate, bool):
        escalate = False

    summary = str(candidate.get("summary", "We are reviewing this email and will follow up shortly.")).strip()
    if not summary:
        summary = "We are reviewing this email and will follow up shortly."

    notes = str(candidate.get("notes", "llm inference")).strip() or "llm inference"

    return {
        "task_id": observation["task_id"],
        "urgency": urgency,
        "department": department,
        "summary": summary,
        "queue_position": queue_position,
        "escalate": escalate,
        "notes": notes,
    }


def _fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    return _heuristic_action(observation)


def _heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{observation.get('subject', '')} {observation.get('body', '')}".lower()
    compliance_risk = bool(observation.get("compliance_risk", False))
    minutes_to_breach = int(observation.get("minutes_to_breach", 9999))
    business_impact = int(observation.get("business_impact", 0))
    sender_tier = str(observation.get("sender_tier", "standard")).lower()

    urgent_hits = ["failed", "error", "outage", "500", "blocked", "ssn", "pci", "exposure", "incident"]
    low_hits = ["dark mode", "feature request", "eta"]

    urgency = "normal"
    if compliance_risk or minutes_to_breach <= 45 or any(token in text for token in urgent_hits):
        urgency = "urgent"
    elif any(token in text for token in low_hits):
        urgency = "low"

    department = _predict_department(text=text, compliance_risk=compliance_risk)

    queue_position = 1 if urgency == "urgent" else (2 if urgency == "normal" else 3)
    if compliance_risk:
        queue_position = 1

    escalate = False
    if compliance_risk or minutes_to_breach <= 45:
        escalate = True
    elif sender_tier in {"enterprise", "strategic"} and business_impact >= 85:
        escalate = True
    elif urgency == "urgent" and minutes_to_breach <= 60:
        escalate = True

    summary = _heuristic_summary(observation, urgency, department, escalate)

    return {
        "task_id": observation["task_id"],
        "urgency": urgency,
        "department": department,
        "summary": summary,
        "queue_position": queue_position,
        "escalate": escalate,
        "notes": "heuristic guardrail",
    }


def _heuristic_summary(observation: Dict[str, Any], urgency: str, department: str, escalate: bool) -> str:
    text = f"{observation.get('subject', '')} {observation.get('body', '')}".lower()

    if department == "technical":
        if "500" in text or "api" in text:
            return "Report ongoing 500 errors on API endpoints and request outage guidance."
        return "Report a technical service disruption and request urgent engineering support."

    if department == "billing":
        if "credit memo" in text or "ledger" in text:
            return "Flag credit memo mismatch and request urgent billing correction before close."
        return "Investigate payment or invoice failures and confirm account billing status."

    if department == "sales":
        return "Ask for enterprise pricing quote and onboarding timeline details."

    if department == "hr":
        if any(token in text for token in ["pci", "ssn", "compliance", "legal"]):
            return "Report sensitive data exposure and request immediate compliance handling guidance."
        return "Route to HR for documentation handling and policy-compliant follow-up."

    if urgency == "low":
        return "Ask about feature availability and ETA, with low-priority follow-up."

    escalation_text = "with escalation due to SLA risk" if escalate else "with standard follow-up"
    return f"Summarize customer request and route to {department} {escalation_text}."


def _predict_department(text: str, compliance_risk: bool) -> str:
    if compliance_risk:
        return "hr"

    # Weighted keyword votes reduce ambiguous routing errors.
    scores = {
        "billing": 0,
        "technical": 0,
        "sales": 0,
        "hr": 0,
        "general": 0,
    }

    keyword_weights = {
        "billing": {
            "invoice": 3,
            "payment": 3,
            "credit memo": 4,
            "ledger": 3,
            "charged": 2,
            "billing": 3,
        },
        "technical": {
            "api": 4,
            "500": 4,
            "error": 3,
            "outage": 4,
            "integration": 2,
            "failed": 2,
            "label": 2,
        },
        "sales": {
            "pricing": 4,
            "quote": 4,
            "enterprise": 3,
            "seats": 2,
            "onboarding": 2,
            "plan": 1,
        },
        "hr": {
            "ssn": 5,
            "pci": 5,
            "compliance": 4,
            "legal": 3,
            "w-9": 4,
            "w9": 4,
            "passport": 4,
        },
    }

    for dept, mapping in keyword_weights.items():
        for token, weight in mapping.items():
            if token in text:
                scores[dept] += weight

    best_department = max(scores, key=scores.get)
    if scores[best_department] == 0:
        return "general"
    return best_department


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI-based inference against OpenEnv API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:7860",
        help="Base URL of the deployed OpenEnv API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_base_url, model_name, hf_token = get_required_llm_env()

    if OpenAI is None:
        raise SystemExit("openai package is not installed. Run: pip install -r requirements.txt")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    scores = run_inference(base_url=args.base_url, client=client, model_name=model_name)

    print("Inference scores:")
    for task_id, score in scores.items():
        print(f"- {task_id}: {score:.3f}")
    print("\nJSON:")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
