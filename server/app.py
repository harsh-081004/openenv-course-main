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
    from inference import run_baseline
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
    from inference import run_baseline
    from environment import EmailTriageEnvironment
    from models import EmailAction, GraderRequest, GraderResponse, ResetRequest, StepResponse, TaskInfo
    from tasks import TASKS


app = FastAPI(title="Email Triage Environment", version="1.0.0")
env = EmailTriageEnvironment()

TASK_GRADER_SPECS: Dict[str, Dict[str, object]] = {
    "task-urgency": {
        "name": "grade_task1",
        "endpoint": "/grader",
        "method": "POST",
        "score_type": "exact_match",
        "required_action_fields": ["urgency"],
    },
    "task-routing": {
        "name": "grade_task2",
        "endpoint": "/grader",
        "method": "POST",
        "score_type": "exact_or_partial_match",
        "required_action_fields": ["department"],
    },
    "task-full-triage": {
        "name": "grade_task3",
        "endpoint": "/grader",
        "method": "POST",
        "score_type": "weighted_composite",
        "weights": {"urgency": 0.3, "department": 0.3, "summary": 0.4},
        "required_action_fields": ["urgency", "department", "summary"],
    },
}


def _registered_graders() -> List[Dict[str, object]]:
    return [
        {
            "task_id": task.task_id,
            "name": TASK_GRADER_SPECS.get(task.task_id, {}).get("name", "grade_task3"),
            "endpoint": "/grader",
            **TASK_GRADER_SPECS.get(task.task_id, {}),
        }
        for task in TASKS
    ]


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
            grader_name=TASK_GRADER_SPECS.get(task.task_id, {}).get("name", "grade_task3"),
            has_grader=True,
            grader_endpoint="/grader",
            grader={
                "task_id": task.task_id,
                **TASK_GRADER_SPECS.get(task.task_id, {"name": "grade_task3", "endpoint": "/grader"}),
            },
        )
        for task in TASKS
    ]
    graders = _registered_graders()
    grader_count = sum(1 for task in task_infos if task.has_grader and bool(task.grader))
    return {
        "tasks": [task.model_dump() for task in task_infos],
        "action_schema": EmailAction.model_json_schema(),
        "grader_endpoint": "/grader",
        "graded_task_count": grader_count,
        "graders": graders,
    }


@app.get("/grader")
def grader_registry() -> Dict[str, object]:
    graders = _registered_graders()
    return {
        "grader_count": len(graders),
        "task_ids": [grader["task_id"] for grader in graders],
        "graders": graders,
    }


@app.post("/grader", response_model=GraderResponse)
def grader(request: GraderRequest | None = None) -> GraderResponse:
    payload = request or GraderRequest()
    action_payload = payload.action or {}
    requested_task_id = payload.task_id or str(action_payload.get("task_id") or "").strip() or env.state().task_id
    if requested_task_id not in {task.task_id for task in TASKS}:
        requested_task_id = env.state().task_id

    action_model = None
    if action_payload:
        action_model = EmailAction(
            task_id=requested_task_id,
            urgency=action_payload.get("urgency"),
            department=action_payload.get("department"),
            summary=action_payload.get("summary"),
            queue_position=action_payload.get("queue_position"),
            escalate=action_payload.get("escalate"),
            notes=action_payload.get("notes"),
        )

    score, details = env.grade(action=action_model, task_id=requested_task_id)
    print(f"[GRADER] task_id={requested_task_id} score={score:.4f} details_keys={list(details.keys())}")
    return GraderResponse(task_id=requested_task_id, score=score, details=details)


@app.post("/baseline")
def baseline() -> Dict[str, object]:
    results = run_baseline(base_url="http://localhost:7860")
    return {"scores": results}


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
