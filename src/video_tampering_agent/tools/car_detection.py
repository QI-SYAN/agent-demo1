"""Vehicle tampering analysis tool backed by the YOLOv10 pipeline."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import json
import cv2
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict, Set
from uuid import uuid4

from ..config import settings


class CarDetectionTool:
    """Run the vehicle detection and tracking pipeline on a video asset."""

    name = "car_detection"
    description = (
        "检测并跟踪视频中的车辆，生成带注释的视频、统计结果与审计日志。"
    )

    def __init__(
        self,
        *,
        weight_path: Optional[str] = None,
        base_output_dir: Optional[Path] = None,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.weight_path = weight_path or os.environ.get("CAR_TOOL_WEIGHT")
        self.base_output_dir = (
            Path(base_output_dir)
            if base_output_dir is not None
            else Path(settings.cache_directory) / "car_detection"
        )
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.extra_args = extra_args or []

    # ---------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        video_path = context.get("video_locator")
        if not video_path:
            raise ValueError("car_detection tool requires 'video_locator' in context")

        normalized_path = self._normalize_path(str(video_path))
        source_path = Path(normalized_path)
        if not source_path.exists():
            raise FileNotFoundError(f"video source not found: {normalized_path}")
        source_path = source_path.resolve()

        request_id = str(context.get("request_id") or uuid4())
        output_dir = self._prepare_output_dir(request_id)
        output_video = (output_dir / "annotated.mp4").resolve()
        log_path = output_dir / "car_detection.log"
        command = self._build_command(str(source_path), output_video)

        result = self._run_pipeline(command, log_path)
        if result.returncode != 0:
            raise RuntimeError(
                "car detection pipeline failed; see log for details"
            )

        summary: dict[str, Any] = {
            "request_id": request_id,
            "output_directory": str(output_dir),
            "output_video": str(output_video),
            "log_file": str(log_path),
        }

        if result.stdout:
            tail = "\n".join(result.stdout.strip().splitlines()[-5:])
            summary["log_tail"] = tail

        result_file = output_dir / "result.txt"
        if result_file.exists():
            summary["result_file"] = str(result_file)
            try:
                result_text = result_file.read_text(encoding="utf-8")
                summary["result_excerpt"] = result_text[:2000]
                suspects = self._summarize_track_map(result_text)
                if suspects:
                    summary["suspected_vanish_tracks"] = suspects
                    
                    # === 新增：提取证据图片 ===
                    evidence_images = []
                    try:
                        video_for_evidence = output_video if output_video.exists() else source_path
                        evidence_images = self._extract_evidence_frames(
                            str(video_for_evidence),
                            output_dir,
                            suspects,
                        )
                        summary["evidence_images"] = evidence_images
                    except Exception as e:
                        # 不让截图失败影响主流程
                        print(f"Warning: Failed to extract evidence frames: {e}")
                        
            except OSError:
                pass

        return summary

    async def ainvoke(self, context: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.invoke, context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_output_dir(self, request_id: str) -> Path:
        target = self.base_output_dir / request_id
        target.mkdir(parents=True, exist_ok=True)
        return target.resolve()

    def _build_command(self, video_path: str, output_video: Path) -> list[str]:
        script_path = Path(__file__).resolve().parent / "_car.py"
        command = [
            sys.executable,
            str(script_path),
            "--input_video_path",
            video_path,
            "--output_video_path",
            str(output_video),
        ]
        if self.weight_path:
            command.extend(["--weight", self.weight_path])
        command.extend(self.extra_args)
        return command

    def _run_pipeline(self, command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
        try:
            log_path.write_text(result.stdout or "", encoding="utf-8")
        except OSError:
            pass
        return result

    @staticmethod
    def _normalize_path(path: str) -> str:
        if path.startswith("file://"):
            return path[7:]
        return path

    @staticmethod
    def _extract_zero_segments(series: List[int]) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for idx, value in enumerate(series):
            if value == 0:
                if start is None:
                    start = idx
            elif start is not None:
                segments.append((start, idx - 1))
                start = None
        if start is not None:
            segments.append((start, len(series) - 1))
        return segments

    def _summarize_track_map(self, text: str) -> List[dict[str, Any]]:
        pattern = re.compile(r"(\d+):\s*\[([0-1,\s]+)\]", re.MULTILINE)
        findings: List[dict[str, Any]] = []
        max_ratio = 0.0
        for match in pattern.finditer(text):
            track_id = int(match.group(1))
            payload = match.group(2).replace("\n", " ")
            values = [int(token) for token in payload.split(",") if token.strip()]
            segments = self._extract_zero_segments(values)
            if not segments:
                continue
            total_frames = len(values)
            missing_total = sum(end - start + 1 for start, end in segments)
            if total_frames:
                max_ratio = max(max_ratio, missing_total / total_frames)
            findings.append(
                {
                    "track_id": track_id,
                    "missing_segments": [
                        {"start_offset": start, "end_offset": end}
                        for start, end in segments
                    ],
                    "missing_total_frames": missing_total,
                    "total_frames": total_frames,
                    "missing_ratio": (missing_total / total_frames) if total_frames else 0.0,
                }
            )
        if findings:
            max_ratio = max(item.get("missing_ratio", 0.0) for item in findings)
        if findings:
            findings.sort(key=lambda item: item.get("missing_total_frames", 0), reverse=True)
        return findings

    def _extract_evidence_frames(self, video_path: str, output_dir: Path, findings: List[dict]) -> List[str]:
        """
        提取异常轨迹的关键帧（消失起点前两帧、起点帧以及后两帧）。
        """
        evidence_paths = []
        if not findings:
            return evidence_paths

        # 创建专门的图片子目录
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # 尝试加载由 _car.py 导出的检测框元数据（每帧每车的 bbox）
        bbox_index: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
        bbox_file = output_dir / "bboxes.json"
        if bbox_file.exists():
            try:
                with bbox_file.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                for frame_key, tracks in raw.items():
                    try:
                        frame_idx = int(frame_key)
                    except ValueError:
                        continue
                    frame_boxes: Dict[int, Tuple[int, int, int, int]] = {}
                    for tid_str, box in tracks.items():
                        try:
                            tid = int(tid_str)
                            x1, y1, x2, y2 = map(int, box)
                        except Exception:
                            continue
                        frame_boxes[tid] = (x1, y1, x2, y2)
                    if frame_boxes:
                        bbox_index[frame_idx] = frame_boxes
            except Exception as e:
                print(f"Warning: failed to load bboxes.json: {e}")

        # 尝试加载由 _car.py 导出的原始消失起止帧（check_map），用于更精确地定义 Start 帧
        segments_index: Dict[int, Tuple[int, int]] = {}
        segments_file = output_dir / "segments.json"
        if segments_file.exists():
            try:
                with segments_file.open("r", encoding="utf-8") as f:
                    raw_segments = json.load(f)
                for tid_str, seg in raw_segments.items():
                    try:
                        tid = int(tid_str)
                    except ValueError:
                        continue
                    start_idx: Optional[int] = None
                    end_idx: Optional[int] = None
                    if isinstance(seg, dict):
                        if "start" in seg:
                            try:
                                start_idx = int(seg["start"])
                            except Exception:
                                start_idx = None
                        if "end" in seg:
                            try:
                                end_idx = int(seg["end"])
                            except Exception:
                                end_idx = None
                    elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                        try:
                            start_idx = int(seg[0])
                            end_idx = int(seg[1])
                        except Exception:
                            start_idx = None
                            end_idx = None
                    if start_idx is not None:
                        # 如果 end 缺失，就让 end == start
                        if end_idx is None:
                            end_idx = start_idx
                        segments_index[tid] = (start_idx, end_idx)
            except Exception as e:
                print(f"Warning: failed to load segments.json: {e}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return evidence_paths
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # 记录每个帧需要导出的多个变体（不同 Track / Pre / Start 各一张图）
        # frame_variants[frame_idx] = [ {"label": str, "tracks": set[int]}, ... ]
        frame_variants: Dict[int, List[dict]] = {}

        for item in findings:
            track_id = item["track_id"]

            # 若有 segments.json 中的绝对起始帧，则优先使用该起始帧作为 Start
            abs_start: Optional[int] = None
            seg_tuple = segments_index.get(track_id)
            if seg_tuple is not None:
                abs_start = seg_tuple[0]

            if abs_start is not None:
                starts_to_use = [abs_start]
            else:
                # 回退：仍然使用基于 track_map 0/1 串解析出的 start_offset（相对索引）
                starts_to_use = [seg["start_offset"] for seg in item["missing_segments"]]

            offsets = [
                (-1, "Pre1"),
                (0, "Start"),
                # (1, "Post1"),
            ]

            for base_start in starts_to_use:
                for offset, suffix in offsets:
                    frame_idx = base_start + offset
                    if frame_idx < 0:
                        continue
                    if frame_count and frame_idx >= frame_count:
                        continue
                    label_text = f"Track_{track_id}_{suffix}"
                    variant = {
                        "label": label_text,
                        "tracks": {track_id},
                    }
                    frame_variants.setdefault(frame_idx, []).append(variant)

        # 遍历视频提取帧
        # 优化：按帧号排序，顺序读取
        if not frame_variants:
            cap.release()
            return evidence_paths

        sorted_frames = sorted(frame_variants.keys())
        current_frame_idx = 0
        
        target_idx = 0
        while cap.isOpened() and target_idx < len(sorted_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame_idx == sorted_frames[target_idx]:
                f_idx = sorted_frames[target_idx]
                variants = frame_variants.get(f_idx, [])
                frame_boxes = bbox_index.get(f_idx, {})

                for variant in variants:
                    label_text = variant.get("label", f"Frame_{f_idx}")
                    overlay_text = f"Frame {f_idx}: {label_text}"

                    # 单独复制一张图，避免不同变体的绘制互相覆盖
                    frame_copy = frame.copy()

                    # 如果有 bbox 信息，则对该帧中关注的轨迹画红圈高亮
                    tracks_for_frame = variant.get("tracks", set())
                    for tid in tracks_for_frame:
                        box = frame_boxes.get(tid)
                        if not box:
                            continue
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        # 半径取框较小边的一半，至少 5 像素
                        r = max(5, min(x2 - x1, y2 - y1) // 2)
                        cv2.circle(
                            frame_copy,
                            (cx, cy),
                            r,
                            (0, 0, 255),  # 红色高亮疑似篡改车辆
                            3,
                        )

                    # 在图上画当前变体的文字标记
                    cv2.putText(
                        frame_copy,
                        overlay_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                    safe_label = label_text.replace("|", "_")
                    filename = f"evidence_frame_{f_idx}_{safe_label}.jpg"
                    save_path = images_dir / filename
                    cv2.imwrite(str(save_path), frame_copy)
                    evidence_paths.append(str(save_path))
                
                target_idx += 1
                # 处理可能有重复帧号的情况（虽然set去重了，但逻辑上保持严谨）
                while target_idx < len(sorted_frames) and sorted_frames[target_idx] == current_frame_idx:
                    target_idx += 1
            
            current_frame_idx += 1

        cap.release()
        return evidence_paths
# *** End of File
