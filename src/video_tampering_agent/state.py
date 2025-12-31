"""Shared agent state definitions."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class DetectionStatus(TypedDict):
    verdict: Literal["unknown", "clean", "tampered"]
    confidence: float
    rationale: str


class AgentState(TypedDict, total=False):
    """LangGraph state contract for the detection agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    request_id: str
    video_locator: str
    video_metadata: dict[str, Any]
    detection_status: DetectionStatus
    evidence: list[dict[str, Any]]
    audit_trail: list[str]
    # List of tool names selected during planning stage (dynamic execution order)
    planned_tools: list[str]
    initial_tools: list[str]
    deferred_tools: list[str]
    current_phase: Literal["initial", "deferred"]
    planning_text: str
    retry_count: int
    reflection_notes: list[str]
    reflection_decision: Literal["replan", "run_deferred", "finalize"]
