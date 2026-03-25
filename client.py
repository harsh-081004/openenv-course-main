# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simple HTTP client for the Email Triage environment."""

from typing import Dict, Optional

import requests

from .models import EmailAction


class EmailTriageClient:
    """Lightweight REST client for the environment."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: Optional[str] = None) -> Dict[str, object]:
        payload = {"task_id": task_id} if task_id else {}
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def step(self, action: EmailAction) -> Dict[str, object]:
        response = requests.post(
            f"{self.base_url}/step", json=action.model_dump(), timeout=30
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, object]:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()
