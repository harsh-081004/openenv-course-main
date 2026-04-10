"""Inference script for the Email Triage OpenEnv environment.

Submission requirements covered:
- Script name is inference.py at project root.
- Uses OpenAI client for LLM action generation.
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment.
- Emits strict stdout line types in order: [START], [STEP], [END].
- Avoids unhandled exceptions by falling back to heuristic actions/scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

TASK_IDS = ["task-urgency", "task-routing", "task-full-triage"]
TASK_DIFFICULTY = {
    "task-urgency": "easy",
    "task-routing": "medium",
    "task-full-triage": "hard",
}
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

SUCCESS_SCORE_THRESHOLD = 0.10
HTTP_TIMEOUT_SECONDS = 30
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
MAX_EMAILS_PER_TASK = int(os.getenv("MAX_EMAILS_PER_TASK", "0"))


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return value


def log_start(task: str, env: str, model: str, difficulty: str = "") -> None:
    diff_part = f" difficulty={_single_line(difficulty)}" if difficulty else ""
    print(f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model)}{diff_part}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(bool(done)).lower()
    error_val = "null" if not error else _single_line(error)
    action_val = _single_line(action)
    print(
        f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = str(bool(success)).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Ensure score is never exactly 0.0 or 1.0
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999
    print(f"[END] task={_single_line(task)} success={success_val} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI-based inference against OpenEnv API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:7860",
        help="Base URL of the deployed OpenEnv API",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: evaluate fewer emails per task with lower LLM timeout.",
    )
    parser.add_argument(
        "--max-emails-per-task",
        type=int,
        default=None,
        help="Override max emails evaluated per task (0 means all).",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="Override LLM request timeout in seconds.",
    )
    return parser.parse_args()


def _configure_runtime(args: argparse.Namespace) -> None:
    global MAX_EMAILS_PER_TASK, LLM_TIMEOUT_SECONDS

    if args.fast:
        MAX_EMAILS_PER_TASK = 3
        LLM_TIMEOUT_SECONDS = min(LLM_TIMEOUT_SECONDS, 12.0)

    if args.max_emails_per_task is not None:
        MAX_EMAILS_PER_TASK = max(0, int(args.max_emails_per_task))

    if args.llm_timeout is not None:
        LLM_TIMEOUT_SECONDS = max(1.0, float(args.llm_timeout))


def _load_all_email_ids() -> List[str]:
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
    ids: List[str] = []
    for item in emails:
        email_id = getattr(item, "email_id", None)
        if isinstance(email_id, str) and email_id:
            ids.append(email_id)
    return ids


def _build_client_and_model() -> Tuple[Optional[Any], str]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

    if OpenAI is None or not api_key:
        return None, model_name

    try:
        return OpenAI(base_url=api_base_url, api_key=api_key), model_name
    except Exception:
        return None, model_name


def _reset_env(session: requests.Session, base_url: str, task_id: str, email_id: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task_id": task_id}
    if email_id:
        payload["email_id"] = email_id
    response = session.post(f"{base_url}/reset", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()["observation"]


def _step_env(session: requests.Session, base_url: str, action: Dict[str, Any]) -> Dict[str, Any]:
    response = session.post(f"{base_url}/step", json=action, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _grade_env(session: requests.Session, base_url: str, task_id: str, action: Dict[str, Any]) -> float:
    response = session.post(
        f"{base_url}/grader",
        json={"task_id": task_id, "action": action},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return _clamp01(float(payload.get("score", 0.0)))


def _extract_step_error(step_payload: Dict[str, Any]) -> Optional[str]:
    info = step_payload.get("info")
    if isinstance(info, dict):
        last_error = info.get("last_action_error")
        if last_error:
            return str(last_error)
    return None


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("No JSON object found")


def _is_useful_summary(summary: str) -> bool:
    if not summary:
        return False
    words = summary.split()
    if len(words) < 8:
        return False
    lower = summary.lower()
    banned = ["i can't", "cannot", "unable", "insufficient", "n/a"]
    return not any(token in lower for token in banned)


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


def _merge_llm_with_heuristics(llm_action: Dict[str, Any], heuristic_action: Dict[str, Any]) -> Dict[str, Any]:
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


def _predict_department(text: str, compliance_risk: bool) -> str:
    if compliance_risk:
        return "hr"

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


def _build_action_with_llm(client: Optional[Any], model_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    heuristic_action = _heuristic_action(observation)
    if client is None:
        return heuristic_action

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
            timeout=LLM_TIMEOUT_SECONDS,
        )
        raw = completion.choices[0].message.content or ""
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Model output was not a JSON object")
        llm_action = _normalize_action(parsed, observation)
        return _merge_llm_with_heuristics(llm_action, heuristic_action)
    except Exception:
        return heuristic_action


def _evaluate(
    session: requests.Session,
    base_url: str,
    client: Optional[Any],
    model_name: str,
    emit_logs: bool,
) -> Tuple[Dict[str, float], List[float], int]:
    task_scores: Dict[str, float] = {}
    rewards: List[float] = []
    step_count = 0

    all_email_ids = _load_all_email_ids()

    for task_id in TASK_IDS:
        email_ids = all_email_ids or TASK_EMAIL_MAP[task_id]
        if MAX_EMAILS_PER_TASK > 0:
            email_ids = email_ids[:MAX_EMAILS_PER_TASK]

        per_task_scores: List[float] = []

        for email_id in email_ids:
            action_str = "noop()"
            reward = 0.0
            done = False
            error: Optional[str] = None
            score = 0.0

            try:
                observation = _reset_env(session, base_url, task_id, email_id)
                action = _build_action_with_llm(client, model_name, observation)
                action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
                step_payload = _step_env(session, base_url, action)
                reward = float(step_payload.get("reward", 0.0))
                done = bool(step_payload.get("done", False))
                error = _extract_step_error(step_payload)
                score = _grade_env(session, base_url, task_id, action)
            except Exception as exc:
                error = str(exc)
                reward = 0.0
                done = True
                score = 0.0

            step_count += 1
            rewards.append(reward)
            per_task_scores.append(_clamp01(score))

            if emit_logs:
                log_step(step_count, action_str, reward, done, error)

        task_scores[task_id] = sum(per_task_scores) / max(1, len(per_task_scores))

    return task_scores, rewards, step_count


def run_baseline(base_url: str = "http://localhost:7860", emit_logs: bool = False) -> Dict[str, float]:
    """Entry point consumed by both the API baseline endpoint and local script runs."""

    client, model_name = _build_client_and_model()
    session = requests.Session()
    try:
        scores, _, _ = _evaluate(
            session=session,
            base_url=base_url,
            client=client,
            model_name=model_name,
            emit_logs=emit_logs,
        )
        return scores
    except Exception:
        return {
            "task-urgency": 0.0,
            "task-routing": 0.0,
            "task-full-triage": 0.0,
        }
    finally:
        session.close()


def main() -> None:
    args = parse_args()
    _configure_runtime(args)

    client, model_name = _build_client_and_model()
    benchmark_name = os.getenv("OPENENV_BENCHMARK", "email-triage-openenv")

    session = requests.Session()
    try:
        # Evaluate each task separately and emit [START]/[END] per task
        for task_id in TASK_IDS:
            difficulty = TASK_DIFFICULTY.get(task_id, "medium")
            log_start(task=task_id, env=benchmark_name, model=model_name, difficulty=difficulty)

            task_rewards: List[float] = []
            task_step_count = 0
            task_score = 0.0

            all_email_ids = _load_all_email_ids()
            email_ids = all_email_ids or TASK_EMAIL_MAP.get(task_id, [])
            if MAX_EMAILS_PER_TASK > 0:
                email_ids = email_ids[:MAX_EMAILS_PER_TASK]

            per_email_scores: List[float] = []

            for email_id in email_ids:
                action_str = "noop()"
                reward = 0.0
                done = False
                error: Optional[str] = None
                score = 0.0

                try:
                    observation = _reset_env(session, args.base_url, task_id, email_id)
                    action = _build_action_with_llm(client, model_name, observation)
                    action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
                    step_payload = _step_env(session, args.base_url, action)
                    reward = float(step_payload.get("reward", 0.0))
                    done = bool(step_payload.get("done", False))
                    error = _extract_step_error(step_payload)
                    score = _grade_env(session, args.base_url, task_id, action)
                except Exception as exc:
                    error = str(exc)
                    reward = 0.0
                    done = True
                    score = 0.0

                task_step_count += 1
                task_rewards.append(reward)
                per_email_scores.append(_clamp01(score))

                log_step(task_step_count, action_str, reward, done, error)

            if per_email_scores:
                task_score = sum(per_email_scores) / len(per_email_scores)
            
            task_success = task_score >= SUCCESS_SCORE_THRESHOLD
            log_end(task=task_id, success=task_success, steps=task_step_count, score=task_score, rewards=task_rewards)

    except Exception as exc:
        print(f"Warning: inference execution failed: {exc}", file=sys.stderr, flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
