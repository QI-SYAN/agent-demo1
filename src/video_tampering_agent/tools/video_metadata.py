import json
import subprocess
import shutil
from typing import Any, Dict

class VideoMetadataTool:
    """Extract and analyze video metadata for tampering traces."""

    name = "video_metadata"
    description = "快速提取视频元数据，检测编辑软件标签、流时长不一致等篡改痕迹。适用于初筛。"

    def invoke(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具入口：分析视频元数据，寻找篡改痕迹。
        """
        video_path = context.get("video_locator")
        if not video_path:
            return {"error": "No video path provided"}

        raw_data = self._get_video_metadata(str(video_path))
        if "error" in raw_data:
            return {"status": "error", "detail": raw_data["error"]}

        format_info = raw_data.get("format", {})
        tags = format_info.get("tags", {})
        
        suspicious_signals = []
        
        # 1. 检查编码软件标签
        encoder = tags.get("encoder", "") or tags.get("handler_name", "")
        suspicious_keywords = ["Lavf", "Adobe", "Premiere", "After Effects", "CapCut", "HandBrake"]
        for kw in suspicious_keywords:
            if kw.lower() in encoder.lower():
                suspicious_signals.append(f"检测到编辑软件痕迹: {encoder}")

        # 2. 检查创建时间与修改时间
        creation_time = tags.get("creation_time")
        # 这里可以加更复杂的逻辑，比如和文件系统时间比对

        # 3. 检查流时长一致性
        duration = float(format_info.get("duration", 0))
        streams = raw_data.get("streams", [])
        for stream in streams:
            s_duration = float(stream.get("duration", 0) or duration) # 有些流可能没duration
            if abs(s_duration - duration) > 1.0: # 容差1秒
                suspicious_signals.append(f"流时长不一致: 容器={duration}s, 流({stream.get('codec_type')})={s_duration}s")

        # 评分逻辑
        score = 0.0
        if suspicious_signals:
            score = 0.4 # 元数据异常通常是强信号，但不足以直接定罪，给个中等分数

        return {
            "summary": f"发现 {len(suspicious_signals)} 个元数据疑点",
            "suspicious_signals": suspicious_signals,
            "encoder": encoder,
            "duration": duration,
            "score": score,
            # 这是一个关键信号，用于触发 Reflection
            "has_editing_traces": len(suspicious_signals) > 0
        }

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """使用 ffprobe 提取视频元数据"""
        ffprobe_cmd = shutil.which("ffprobe")
        if not ffprobe_cmd:
            # 如果环境变量没配，尝试使用绝对路径，或者抛出更友好的错误
            # 这里假设用户环境里有 ffmpeg/ffprobe
            return {"error": "ffprobe not found"}

        cmd = [
            ffprobe_cmd,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return data
        except Exception as e:
            return {"error": str(e)}
