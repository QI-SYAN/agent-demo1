"""VLM Analysis tool for tampering detection."""

from __future__ import annotations

import os
import re
import base64
import json
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

from openai import OpenAI
from ..config import settings

# === 新增：确保加载 .env ===
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def _load_env_if_needed():
    """确保环境变量已加载"""
    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("VLM_API_KEY"):
        return

    if load_dotenv is None:
        return

    # 从当前文件向上寻找 .env
    current_dir = Path(__file__).resolve().parent
    # 向上找 4 层足够覆盖到项目根目录
    for _ in range(4):
        env_file = current_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
        current_dir = current_dir.parent

# 在模块导入时执行加载
_load_env_if_needed()

# === Helper Functions ===

FILENAME_PATTERN = re.compile(
    r"^evidence_frame_(?P<frame>\d+)_Track_(?P<track>\d+)_(?P<phase>Pre1|Start|Post1)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

def parse_filename(path: Path):
    """从文件名中解析出 frame_index, track_id, phase(Pre1/Start/Post1)。"""
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    frame_idx = int(m.group("frame"))
    track_id = int(m.group("track"))
    phase = m.group("phase")
    return frame_idx, track_id, phase

def find_image_pairs(images_dir: Path, max_pairs: int = 7) -> List[Tuple[Path, Path]]:
    """在指定目录中，根据命名规则寻找 (Pre1, Start) 成对的图片。"""
    images = [p for p in images_dir.iterdir() if p.is_file()]

    # 索引结构：by (frame, track, phase)
    index = {}
    for img in images:
        parsed = parse_filename(img)
        if not parsed:
            continue
        frame_idx, track_id, phase = parsed
        index[(frame_idx, track_id, phase)] = img

    pairs: List[Tuple[Path, Path]] = []

    # 遍历所有 Start，寻找上一帧的 Pre1
    for (frame_idx, track_id, phase), start_img in index.items():
        if phase != "Start":
            continue
        pre_key = (frame_idx - 1, track_id, "Pre1")
        pre_img = index.get(pre_key)
        if not pre_img:
            continue
        pairs.append((pre_img, start_img))

    # 按 Start 帧号排序
    pairs.sort(key=lambda pair: parse_filename(pair[1])[0])

    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    return pairs

def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _save_results(images_dir: Path, results: List[Dict[str, Any]], prompt: str) -> None:
    """将每一对图片的 VLM 结果保存为 JSON 和 Markdown 方便复盘。"""
    if not results:
        return

    # JSON 结果
    json_path = images_dir / "vlm_results.json"
    payload = {
        "images_dir": str(images_dir),
        "prompt": prompt,
        "pairs": results,
    }
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to write JSON results: {e}")

    # Markdown 结果
    md_path = images_dir / "vlm_results.md"
    try:
        lines: List[str] = []
        lines.append("# VLM 对比结果\n")
        lines.append(f"图片目录：`{images_dir}`\n\n")
        lines.append("## 全局提示词\n\n")
        lines.append("```text\n")
        lines.append(prompt.rstrip("\n") + "\n")
        lines.append("```\n\n")

        for item in results:
            idx = item.get("index")
            pre_name = item.get("pre_image")
            start_name = item.get("start_image")
            response = item.get("response")
            error = item.get("error")

            lines.append(f"## Pair {idx}: {pre_name} -> {start_name}\n\n")
            if error and not response:
                lines.append(f"**调用出错**: {error}\n\n")
            elif response is not None:
                lines.append("**模型输出：**\n\n")
                # 用引用块保留换行
                for line in str(response).splitlines() or [""]:
                    lines.append(f"> {line}\n")
                lines.append("\n")
            else:
                lines.append("*(无返回内容)*\n\n")

        with md_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Warning: failed to write Markdown results: {e}")


class VLMAnalysisTool:
    """Run VLM analysis on extracted evidence frames."""

    name = "vlm_analysis"
    description = (
        "调用视觉大模型(VLM)对生成的证据图片对进行分析，判断是否存在篡改。"
    )

    def __init__(self) -> None:
        pass

    def invoke(self, context: dict[str, Any]) -> str:
        """
        Execute the VLM analysis.
        
        Expected context keys:
        - images_dir (optional): Path to the directory containing evidence images.
                                 If not provided, defaults to settings.cache_directory / "car_detection" / "demo-forgery" / "images"
                                 or similar logic.
        """
        # Determine images directory
        images_dir_str = context.get("images_dir")
        if not images_dir_str:
             # Fallback to env var or default location if not provided in context
             images_dir_str = os.environ.get("VLM_IMAGES_DIR")
        
        if not images_dir_str:
             # Try to guess based on car_detection output structure if possible, 
             # but for now let's require it or use a default relative path
             images_dir_str = "./images"

        images_dir = Path(images_dir_str).resolve()
        
        if not images_dir.exists() or not images_dir.is_dir():
            return f"Error: Images directory not found: {images_dir}"

        # Load configuration
        # 优先使用 .env 中的 VLM_PROMPT，如果未设置则使用默认值
        _raw_prompt = os.environ.get(
            "VLM_PROMPT",
            "请对这两帧视频截图进行对比，判断是否存在车辆篡改或消失现象，并给出理由。",
        )
        prompt = _raw_prompt.replace("\\n", "\n") if _raw_prompt is not None else ""
        
        max_pairs = int(os.environ.get("VLM_MAX_PAIRS", "100"))
        
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("VLM_API_KEY")
        base_url = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.environ.get("VLM_MODEL_NAME", "qwen-vl-plus")

        if not api_key:
            return "Error: VLM_API_KEY or DASHSCOPE_API_KEY not set in environment."

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Find pairs
        pairs = find_image_pairs(images_dir, max_pairs=max_pairs)
        if not pairs:
            return "No image pairs found for analysis."

        results = []
        summary_lines = []

        for idx, (pre_img, start_img) in enumerate(pairs, start=1):
            record: Dict[str, Any] = {
                "index": idx,
                "pre_image": pre_img.name,
                "start_image": start_img.name,
            }
            
            try:
                # Call VLM
                img1_b64 = encode_image_to_base64(pre_img)
                img2_b64 = encode_image_to_base64(start_img)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"},
                            },
                        ],
                    }
                ]

                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )

                choice = resp.choices[0].message
                content = choice.content
                text_response = ""
                if isinstance(content, list) and content and hasattr(content[0], "text"):
                    text_response = content[0].text
                else:
                    text_response = str(content)
                
                record["response"] = text_response
                summary_lines.append(f"Pair {idx} ({pre_img.name} -> {start_img.name}):\n{text_response}\n")

            except Exception as e:
                error_msg = str(e)
                record["error"] = error_msg
                summary_lines.append(f"Pair {idx} Error: {error_msg}\n")

            results.append(record)

        # Save detailed results to disk
        _save_results(images_dir, results, prompt)

        return "\n---\n".join(summary_lines)
