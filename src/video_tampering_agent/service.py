"""FastAPI service exposing the detection workflow."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .config import AgentSettings, settings
from .custom_logging import configure_logging
from .runner import build_detection_runnable
from .tools import ToolRegistry


class DetectionRequest(BaseModel):
    video_locator: str = Field(..., description="Video location or identifier")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata about the video")
    prompt: str | None = Field(default=None, description="Optional user prompt for the agent")


class DetectionResponse(BaseModel):
    request_id: str
    verdict: str
    confidence: float
    rationale: str
    audit_trail: list[str]
    evidence: list[dict[str, Any]]


def create_app(
    tool_registry: Optional[ToolRegistry] = None,
    *,
    agent_settings: Optional[AgentSettings] = None,
) -> FastAPI:
    cfg = agent_settings or settings
    registry = tool_registry or ToolRegistry()

    configure_logging(json_output=cfg.environment == "prod")
    runnable = build_detection_runnable(registry, agent_settings=cfg)

    app = FastAPI(title="Video Tampering Detection Agent", version="0.1.0")

    def get_runnable():
        return runnable

    @app.post("/detect", response_model=DetectionResponse)
    async def detect(
        payload: DetectionRequest,
        compiled_graph = Depends(get_runnable),
    ) -> DetectionResponse:
        from langchain_core.messages import HumanMessage

        request_id = str(uuid.uuid4())
        initial_state = {
            "request_id": request_id,
            "video_locator": payload.video_locator,
            "video_metadata": payload.metadata or {},
            "messages": [
                HumanMessage(content=payload.prompt or "请分析该视频是否存在篡改迹象，并给出理由。"),
            ],
            "audit_trail": ["api: request received"],
        }

        result_state = await compiled_graph.ainvoke(initial_state)
        status = result_state.get("detection_status", {
            "verdict": "unknown",
            "confidence": 0.0,
            "rationale": "流程未返回检测结果。",
        })

        return DetectionResponse(
            request_id=request_id,
            verdict=status["verdict"],
            confidence=status["confidence"],
            rationale=status["rationale"],
            audit_trail=result_state.get("audit_trail", []),
            evidence=result_state.get("evidence", []),
        )

    return app
