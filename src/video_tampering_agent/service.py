"""FastAPI service exposing the detection workflow."""

from __future__ import annotations

import uuid
from pathlib import Path
import shutil
import re
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import AgentSettings, settings
from .custom_logging import configure_logging
from .reporting import generate_detection_report
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
    planning_text: str | None = None
    initial_tools: list[str] = Field(default_factory=list)
    deferred_tools: list[str] = Field(default_factory=list)
    reflection_notes: list[str] = Field(default_factory=list)
    reflection_decision: str | None = None
    evidence_frames: list[dict[str, Any]] = Field(default_factory=list)
    audit_trail: list[str]
    evidence: list[dict[str, Any]]
    report_artifacts: dict[str, Any] | None = None


class UploadResponse(BaseModel):
    video_locator: str = Field(..., description="Saved video path returned to client")
    file_name: str = Field(..., description="Original file name")
    content_type: str | None = Field(default=None, description="Uploaded file content-type")


def create_app(
    tool_registry: Optional[ToolRegistry] = None,
    *,
    agent_settings: Optional[AgentSettings] = None,
) -> FastAPI:
    cfg = agent_settings or settings
    registry = tool_registry or ToolRegistry()
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = repo_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    configure_logging(json_output=cfg.environment == "prod")
    runnable = build_detection_runnable(registry, agent_settings=cfg)

    app = FastAPI(title="Video Tampering Detection Agent", version="0.1.0")
    app.mount("/artifacts", StaticFiles(directory=str(cache_root)), name="artifacts")

    # Development-friendly CORS (Vite default origin).
    # If you use Vite proxy, CORS is less critical, but keeping it on is convenient.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"] ,
        allow_headers=["*"],
    )

    def get_runnable():
        return runnable

    def _to_artifact_url(path_str: str) -> str | None:
        try:
            path = Path(path_str).resolve()
            rel = path.relative_to(cache_root.resolve())
            return "/artifacts/" + rel.as_posix()
        except Exception:
            return None

    filename_pattern = re.compile(
        r"^evidence_frame_(?P<frame>\d+)_Track_(?P<track>\d+)_(?P<phase>Pre1|Start|Post1)\.(jpg|jpeg|png)$",
        re.IGNORECASE,
    )

    def _collect_evidence_frames(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, int], dict[str, Any]] = {}

        for item in evidence:
            if item.get("tool") != "car_detection":
                continue
            detail = item.get("detail") or {}
            images = detail.get("evidence_images") or []
            for image_path in images:
                name = Path(str(image_path)).name
                match = filename_pattern.match(name)
                if not match:
                    continue

                frame_index = int(match.group("frame"))
                track_id = int(match.group("track"))
                phase = match.group("phase")
                # 将 Start 帧作为配对主键；Pre1 的 frame+1 对应 Start
                start_frame = frame_index + 1 if phase == "Pre1" else frame_index
                key = (track_id, start_frame)
                entry = grouped.setdefault(
                    key,
                    {
                        "track_id": track_id,
                        "start_frame": start_frame,
                        "pre_image_url": None,
                        "start_image_url": None,
                        "pre_image_name": None,
                        "start_image_name": None,
                    },
                )

                url = _to_artifact_url(str(image_path))
                if phase == "Pre1":
                    entry["pre_image_url"] = url
                    entry["pre_image_name"] = name
                elif phase == "Start":
                    entry["start_image_url"] = url
                    entry["start_image_name"] = name

        pairs = [
            item
            for item in grouped.values()
            if item.get("pre_image_url") or item.get("start_image_url")
        ]
        pairs.sort(key=lambda x: (x.get("track_id", 0), x.get("start_frame", 0)))
        return pairs

    @app.post("/upload", response_model=UploadResponse)
    async def upload_video(file: UploadFile = File(...)) -> UploadResponse:
        """Accept a video file upload and persist it for downstream detection.

        The returned `video_locator` can be passed directly into `/detect`.
        """

        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov"}:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Save into repo-local cache folder.
        repo_root = Path(__file__).resolve().parents[2]
        dest_dir = repo_root / ".cache" / "uploads"
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename).name
        dest_path = dest_dir / f"{uuid.uuid4().hex}_{safe_name}"

        try:
            with dest_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        return UploadResponse(
            video_locator=str(dest_path),
            file_name=safe_name,
            content_type=file.content_type,
        )

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
        evidence = result_state.get("evidence", [])
        audit_trail = result_state.get("audit_trail", [])
        evidence_frames = _collect_evidence_frames(evidence)

        report_artifacts: dict[str, Any] | None = None
        try:
            generated = generate_detection_report(
                cache_root=cache_root,
                request_id=request_id,
                video_locator=payload.video_locator,
                detection_status=status,
                planning_text=result_state.get("planning_text"),
                initial_tools=result_state.get("initial_tools", []),
                deferred_tools=result_state.get("deferred_tools", []),
                reflection_notes=result_state.get("reflection_notes", []),
                reflection_decision=result_state.get("reflection_decision"),
                evidence=evidence,
                audit_trail=audit_trail,
            )
            report_artifacts = {
                **generated,
                "docx_url": _to_artifact_url(generated.get("docx_path") or ""),
                "pdf_url": _to_artifact_url(generated.get("pdf_path") or ""),
            }
        except Exception as exc:
            report_artifacts = {
                "docx_path": None,
                "pdf_path": None,
                "docx_url": None,
                "pdf_url": None,
                "generation_error": f"报告生成失败：{exc}",
            }

        return DetectionResponse(
            request_id=request_id,
            verdict=status["verdict"],
            confidence=status["confidence"],
            rationale=status["rationale"],
            planning_text=result_state.get("planning_text"),
            initial_tools=result_state.get("initial_tools", []),
            deferred_tools=result_state.get("deferred_tools", []),
            reflection_notes=result_state.get("reflection_notes", []),
            reflection_decision=result_state.get("reflection_decision"),
            evidence_frames=evidence_frames,
            audit_trail=audit_trail,
            evidence=evidence,
            report_artifacts=report_artifacts,
        )

    return app
