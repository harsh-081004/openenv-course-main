# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simple HTTP client for the Email Triage environment."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

try:
    from .models import EmailAction, EmailObservation
except ImportError:
    from models import EmailAction, EmailObservation


class StepResult:
    """Result from a step or reset call."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        self.observation = data.get("observation", {})
        self.reward = data.get("reward", 0.0)
        self.done = data.get("done", False)
        self.info = data.get("info", {})

    def __repr__(self) -> str:
        return f"StepResult(reward={self.reward}, done={self.done})"


class EmailTriageEnv:
    """OpenEnv-compatible client for the Email Triage environment.
    
    Supports context manager pattern:
        with EmailTriageEnv(base_url="https://your-space.hf.space") as env:
            result = env.reset()
            result = env.step(EmailAction(task_id="task-urgency", urgency="urgent"))
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._session: Optional[requests.Session] = None

    def __enter__(self) -> "EmailTriageEnv":
        self._session = requests.Session()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def reset(self, task_id: Optional[str] = None, email_id: Optional[str] = None) -> StepResult:
        """Reset the environment to initial state."""
        payload: Dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        if email_id:
            payload["email_id"] = email_id
        response = self._get_session().post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return StepResult(response.json())

    def step(self, action: EmailAction) -> StepResult:
        """Take an action in the environment."""
        response = self._get_session().post(
            f"{self.base_url}/step", json=action.model_dump(), timeout=30
        )
        response.raise_for_status()
        return StepResult(response.json())

    def state(self) -> Dict[str, object]:
        """Get the current environment state."""
        response = self._get_session().get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()

    def grade(self, task_id: Optional[str] = None, action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Grade the current or provided action."""
        payload: Dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        if action:
            payload["action"] = action
        response = self._get_session().post(f"{self.base_url}/grader", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, str]:
        """Check if the environment is healthy."""
        response = self._get_session().get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()


# Alias for backwards compatibility
EmailTriageClient = EmailTriageEnv
