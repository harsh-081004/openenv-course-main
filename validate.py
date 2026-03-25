"""Pre-submission validator for the Email Triage environment."""

from __future__ import annotations

import os
import sys
from typing import Dict

import requests
import yaml


def main() -> int:
    base_url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
    errors = []

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
    _check_endpoint("GET", f"{base_url}/tasks", errors)
    _check_endpoint(
        "POST",
        f"{base_url}/grader",
        errors,
        payload={"task_id": "task-urgency", "action": {"task_id": "task-urgency", "urgency": "urgent"}},
    )

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation passed")
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


if __name__ == "__main__":
    sys.exit(main())
