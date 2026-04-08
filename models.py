# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage & Prioritization environment.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EmailAction(BaseModel):
    """Agent action for email triage."""

    task_id: str = Field(..., description="Target task id")
    urgency: Optional[str] = Field(
        default=None, description="Urgency label: urgent|normal|low"
    )
    department: Optional[str] = Field(
        default=None,
        description="Department: billing|technical|sales|hr|general",
    )
    summary: Optional[str] = Field(
        default=None, description="One-sentence response summary"
    )
    queue_position: Optional[int] = Field(
        default=None, description="Desired queue position (1 = front)"
    )
    escalate: Optional[bool] = Field(
        default=None, description="Whether this email should be escalated to incident handling"
    )
    notes: Optional[str] = Field(default=None, description="Optional reasoning notes")


class EmailObservation(BaseModel):
    """Observation containing the email and task context."""

    email_id: str = Field(..., description="Unique email id")
    sender: str = Field(..., description="Sender name or address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    task_id: str = Field(..., description="Current task id")
    task_name: str = Field(..., description="Current task name")
    available_urgency_labels: List[str] = Field(
        default_factory=list, description="Allowed urgency labels"
    )
    available_departments: List[str] = Field(
        default_factory=list, description="Allowed departments"
    )
    compliance_risk: bool = Field(
        default=False, description="Email contains compliance-sensitive content"
    )
    sender_tier: str = Field(..., description="Customer tier: standard|enterprise|strategic")
    business_impact: int = Field(..., description="Estimated impact score (0-100)")
    minutes_to_breach: int = Field(..., description="Minutes until SLA breach")
    queue_position: int = Field(..., description="Current queue position (1 = front)")
    queue_size: int = Field(..., description="Total queue size")
    step_count: int = Field(..., description="Current step count")
    max_steps: int = Field(..., description="Max steps per episode")


class Reward(BaseModel):
    """Detailed reward breakdown for shaped rewards."""

    total: float = Field(..., description="Total reward in [0.0, 1.0]")
    components: Dict[str, float] = Field(
        default_factory=dict, description="Named reward components"
    )
    penalties: List[str] = Field(default_factory=list, description="Applied penalties")


class EnvState(BaseModel):
    """Current environment state."""

    episode_id: str = Field(..., description="Episode id")
    step_count: int = Field(..., description="Step count")
    task_id: str = Field(..., description="Task id")
    email_id: str = Field(..., description="Email id")
    done: bool = Field(..., description="Episode done flag")
    last_reward: float = Field(..., description="Last reward value")
    reset_count: int = Field(..., description="Number of resets")


class ResetRequest(BaseModel):
    """Optional reset request parameters."""

    task_id: Optional[str] = Field(default=None, description="Requested task id")
    email_id: Optional[str] = Field(default=None, description="Requested email id")


class StepResponse(BaseModel):
    """Response for step endpoint."""

    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, object]


class TaskInfo(BaseModel):
    """Task metadata for /tasks endpoint."""

    task_id: str
    name: str
    difficulty: str
    objective: str
    description: str
    grader_name: str = Field(..., description="Name of grader logic associated with this task")
    has_grader: bool = Field(default=True, description="Whether this task has an enabled grader")
    grader_endpoint: str = Field(default="/grader", description="HTTP endpoint used for grading")
    grader: Dict[str, object] = Field(default_factory=dict, description="Structured grader metadata")


class GraderRequest(BaseModel):
    """Request payload for /grader."""

    task_id: Optional[str] = None
    action: Optional[Dict[str, object]] = None


class GraderResponse(BaseModel):
    """Response payload for /grader."""

    task_id: str
    score: float
    details: Dict[str, float]
