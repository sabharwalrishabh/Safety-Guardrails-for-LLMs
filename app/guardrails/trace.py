from typing import Any, Literal
from pydantic import BaseModel, Field


Action = Literal["allow", "warn", "block"]


class CheckResult(BaseModel):
    name: str
    triggered: bool = False
    action: Literal["none", "allow", "warn", "block", "mask"] = "none"
    details: dict[str, Any] = Field(default_factory=dict)


class RepairAttempt(BaseModel):
    phase: Literal["pre", "post"]
    attempt: int
    candidate_action: Action
    blocked_by: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    note: str = ""


class DecisionTrace(BaseModel):
    policy_id: str
    action: Action = "allow"
    reasons: list[str] = Field(default_factory=list)
    checks_ran: list[CheckResult] = Field(default_factory=list)

    # Step-7: record repair loop behavior
    repair_attempts: list[RepairAttempt] = Field(default_factory=list)
