# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the Email Triage environment."""

from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI

try:
    from ..baseline.inference import run_baseline
    from ..environment import EmailTriageEnvironment
    from ..models import (
        EmailAction,
        GraderRequest,
        GraderResponse,
        ResetRequest,
        StepResponse,
        TaskInfo,
    )
    from ..tasks import TASKS
except ImportError:
    from baseline.inference import run_baseline
    from environment import EmailTriageEnvironment
    from models import EmailAction, GraderRequest, GraderResponse, ResetRequest, StepResponse, TaskInfo
    from tasks import TASKS


app = FastAPI(title="Email Triage Environment", version="1.0.0")
env = EmailTriageEnvironment()


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=StepResponse)
def reset(request: ResetRequest | None = None) -> StepResponse:
    payload = request or ResetRequest()
    observation = env.reset(task_id=payload.task_id, email_id=payload.email_id)
    return StepResponse(
        observation=observation,
        reward=0.0,
        done=False,
        info={"message": "reset"},
    )


@app.post("/step", response_model=StepResponse)
def step(action: EmailAction) -> StepResponse:
    observation, reward_detail, done, info = env.step(action)
    return StepResponse(
        observation=observation,
        reward=reward_detail.total,
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> Dict[str, object]:
    return env.state().model_dump()


@app.get("/tasks")
def tasks() -> Dict[str, object]:
    task_infos: List[TaskInfo] = [
        TaskInfo(
            task_id=task.task_id,
            name=task.name,
            difficulty=task.difficulty,
            objective=task.objective,
            description=task.description,
        )
        for task in TASKS
    ]
    return {
        "tasks": [task.model_dump() for task in task_infos],
        "action_schema": EmailAction.model_json_schema(),
    }


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest | None = None) -> GraderResponse:
    payload = request or GraderRequest()
    score, details = env.grade(action=payload.action, task_id=payload.task_id)
    task_id = payload.task_id or (payload.action.task_id if payload.action else env.state().task_id)
    return GraderResponse(task_id=task_id, score=score, details=details)


@app.post("/baseline")
def baseline() -> Dict[str, object]:
    results = run_baseline(base_url="http://localhost:7860")
    return {"scores": results}


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
