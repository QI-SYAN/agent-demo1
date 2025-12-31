"""Background modelling residual analysis tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np

from ..config import settings


class BackgroundResidualTool:
    """Compute background subtraction residuals to flag tampering candidates."""

    name = "background_residual"
    description = "通过背景建模统计残差比例，定位可能被篡改的帧区间。"

    def __init__(
        self,
        *,
        history: int = 500,
        var_threshold: float = 16.0,
        residual_threshold: float = 0.15,
        base_output_dir: Optional[Path] = None,
        warmup_frames: int = 3,
    ) -> None:
        self.history = history
        self.var_threshold = var_threshold
        self.residual_threshold = residual_threshold
        self.warmup_frames = max(0, warmup_frames)
        self.base_output_dir = (
            Path(base_output_dir)
            if base_output_dir is not None
            else Path(settings.cache_directory) / "background_residual"
        )
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        video_path = context.get("video_locator")
        if not video_path:
            raise ValueError("background_residual tool requires 'video_locator' in context")

        normalized_path = self._normalize_path(str(video_path))
        cap = cv2.VideoCapture(normalized_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"unable to open video: {normalized_path}")

        request_id = str(context.get("request_id") or uuid4())
        output_dir = self._prepare_output_dir(request_id)
        report_path = output_dir / "residual_report.txt"

        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )

        frame_idx = 0
        ratios: list[float] = []
        suspicious: list[int] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = subtractor.apply(gray)
            residual = mask > 0
            ratio = float(np.count_nonzero(residual)) / residual.size
            if frame_idx >= self.warmup_frames:
                ratios.append(ratio)
                if ratio >= self.residual_threshold:
                    suspicious.append(frame_idx)
            frame_idx += 1

        cap.release()

        average_ratio = float(np.mean(ratios)) if ratios else 0.0
        max_ratio = float(np.max(ratios)) if ratios else 0.0

        report_lines = [
            f"request_id: {request_id}",
            f"total_frames: {frame_idx}",
            f"warmup_frames: {self.warmup_frames}",
            f"average_residual_ratio: {average_ratio:.6f}",
            f"max_residual_ratio: {max_ratio:.6f}",
            f"residual_threshold: {self.residual_threshold}",
            f"suspicious_frames(sample<=100): {suspicious[:100]}",
        ]
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        return {
            "request_id": request_id,
            "output_directory": str(output_dir),
            "report_file": str(report_path),
            "average_residual_ratio": average_ratio,
            "max_residual_ratio": max_ratio,
            "suspicious_frames": suspicious[:100],
            "total_frames": frame_idx,
        }

    async def ainvoke(self, context: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.invoke, context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_output_dir(self, request_id: str) -> Path:
        target = self.base_output_dir / request_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _normalize_path(path: str) -> str:
        if path.startswith("file://"):
            return path[7:]
        return path
# *** End of File
