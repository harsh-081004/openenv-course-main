"""Pre-submission validator for the Email Triage environment."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import requests
import yaml


def main() -> int:
    base_url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    errors = []
    graded_tasks: List[Tuple[str, str]] = []
    task_probe_rows: List[Tuple[str, float, float]] = []

    try:
        with open("openenv.yaml", "r", encoding="utf-8") as file:
            manifest = yaml.safe_load(file)
    except FileNotFoundError:
        errors.append("openenv.yaml not found")
        manifest = {}

    tags = set(manifest.get("tags", []) or [])
    if "openenv" not in tags:
        errors.append("openenv tag missing in openenv.yaml")

    if manifest.get("port") != 7860:
        errors.append("openenv.yaml port must be 7860")

    _check_endpoint("GET", f"{base_url}/", errors)
    _check_endpoint("POST", f"{base_url}/reset", errors, payload={"task_id": "task-urgency"})
    _check_endpoint(
        "POST",
        f"{base_url}/step",
        errors,
        payload={"task_id": "task-urgency", "urgency": "urgent"},
    )
    _check_endpoint("GET", f"{base_url}/state", errors)
    graded_tasks = _collect_graded_tasks(base_url, errors)
    task_probe_rows = _probe_tasks(base_url, graded_tasks, errors)

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed")
    print("Detected grader-backed tasks:")
    for task_id, grader_name in graded_tasks:
        print(f"- {task_id}: {grader_name}")
    print("Per-task probe results (reward, grader_score):")
    for task_id, reward, score in task_probe_rows:
        print(f"- {task_id}: reward={reward:.4f}, score={score:.4f}")
    return 0


def _check_endpoint(method: str, url: str, errors: list[str], payload: Dict | None = None) -> None:
    try:
        if method == "GET":
            response = requests.get(url, timeout=20)
        else:
            response = requests.post(url, json=payload, timeout=20)
        if response.status_code != 200:
            errors.append(f"{method} {url} returned {response.status_code}")
    except requests.RequestException as exc:
        errors.append(f"{method} {url} failed: {exc}")


def _collect_graded_tasks(base_url: str, errors: list[str]) -> List[Tuple[str, str]]:
    discovered: Dict[str, str] = {}

    try:
        response = requests.get(f"{base_url}/tasks", timeout=20)
    except requests.RequestException as exc:
        errors.append(f"GET {base_url}/tasks failed: {exc}")
        return []

    if response.status_code != 200:
        errors.append(f"GET {base_url}/tasks returned {response.status_code}")
        return []

    try:
        payload = response.json()
    except ValueError:
        errors.append("GET /tasks did not return valid JSON")
        return []

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        errors.append("GET /tasks response missing tasks list")
        return []

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            continue

        grader_block = task.get("grader") if isinstance(task.get("grader"), dict) else {}
        grader_name = str(
            task.get("grader_name")
            or grader_block.get("grader_name")
            or grader_block.get("name")
            or "unknown"
        ).strip()
        has_grader = bool(task.get("has_grader")) or bool(grader_block) or bool(task.get("grader_name"))

        if has_grader:
            discovered[task_id] = grader_name

    # Optional compatibility: merge in any grader registry details if present.
    _merge_grader_registry(base_url, discovered)

    if len(discovered) < 3:
        errors.append(f"Not enough tasks with graders: found {len(discovered)}, require at least 3")

    ordered = sorted(discovered.items(), key=lambda item: item[0])
    return ordered


def _merge_grader_registry(base_url: str, discovered: Dict[str, str]) -> None:
    try:
        response = requests.get(f"{base_url}/grader", timeout=20)
    except requests.RequestException:
        return

    if response.status_code != 200:
        return

    try:
        payload = response.json()
    except ValueError:
        return

    graders = payload.get("graders")
    if isinstance(graders, list):
        for grader in graders:
            if not isinstance(grader, dict):
                continue
            task_id = str(grader.get("task_id") or "").strip()
            if not task_id:
                continue
            grader_name = str(grader.get("grader_name") or grader.get("name") or "unknown").strip()
            discovered.setdefault(task_id, grader_name)

    task_ids = payload.get("task_ids")
    if isinstance(task_ids, list):
        known_ids: Set[str] = {task_id for task_id in discovered}
        for task_id in task_ids:
            normalized = str(task_id or "").strip()
            if normalized and normalized not in known_ids:
                discovered[normalized] = "unknown"
                known_ids.add(normalized)


def _probe_tasks(
    base_url: str, graded_tasks: List[Tuple[str, str]], errors: list[str]
) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []

    for task_id, _grader_name in graded_tasks:
        action = _build_probe_action(task_id)

        try:
            reset_response = requests.post(
                f"{base_url}/reset",
                json={"task_id": task_id},
                timeout=20,
            )
        except requests.RequestException as exc:
            errors.append(f"POST {base_url}/reset failed for {task_id}: {exc}")
            continue

        if reset_response.status_code != 200:
            errors.append(f"POST {base_url}/reset returned {reset_response.status_code} for {task_id}")
            continue

        try:
            step_response = requests.post(
                f"{base_url}/step",
                json=action,
                timeout=20,
            )
        except requests.RequestException as exc:
            errors.append(f"POST {base_url}/step failed for {task_id}: {exc}")
            continue

        if step_response.status_code != 200:
            errors.append(f"POST {base_url}/step returned {step_response.status_code} for {task_id}")
            continue

        try:
            step_payload = step_response.json()
        except ValueError:
            errors.append(f"POST /step did not return valid JSON for {task_id}")
            continue

        reward_value = step_payload.get("reward")
        reward = _normalize_unit_interval_value(reward_value)
        if reward is None:
            errors.append(f"Reward out of range [0.0, 1.0] for {task_id}: {reward_value}")
            continue

        try:
            grader_response = requests.post(
                f"{base_url}/grader",
                json={"task_id": task_id, "action": action},
                timeout=20,
            )
        except requests.RequestException as exc:
            errors.append(f"POST {base_url}/grader failed for {task_id}: {exc}")
            continue

        if grader_response.status_code != 200:
            errors.append(f"POST {base_url}/grader returned {grader_response.status_code} for {task_id}")
            continue

        try:
            grader_payload = grader_response.json()
        except ValueError:
            errors.append(f"POST /grader did not return valid JSON for {task_id}")
            continue

        score_value = grader_payload.get("score")
        score = _normalize_unit_interval_value(score_value)
        if score is None:
            errors.append(f"Grader score out of range [0.0, 1.0] for {task_id}: {score_value}")
            continue

        rows.append((task_id, reward, score))

    return rows


def _build_probe_action(task_id: str) -> Dict[str, object]:
    return {
        "task_id": task_id,
        "urgency": "urgent",
        "department": "billing",
        "summary": "Automated validation probe summary.",
        "queue_position": 1,
        "escalate": True,
        "notes": "validator probe",
    }


def _normalize_unit_interval_value(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if 0.0 <= number <= 1.0:
        return number
    return None


if __name__ == "__main__":
    sys.exit(main())
