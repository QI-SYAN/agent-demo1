import os
import re
import base64
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import requests
from openai import OpenAI


def _load_env_from_project_root() -> None:
    """从项目根目录加载 .env（不依赖 python-dotenv）。

     根目录假定为包含 src 的目录：
     video_tampering_agent/.env
     └─ src/
         └─ video_tampering_agent/tools/vlm_pair_uploader.py
    """
    try:
        here = Path(__file__).resolve()
        # 结构：video_tampering_agent/src/video_tampering_agent/tools/vlm_pair_uploader.py
        # parents[0] = .../tools
        # parents[1] = .../video_tampering_agent (包)
        # parents[2] = .../src
        # parents[3] = .../video_tampering_agent （项目根，.env 所在处）
        project_root = here.parents[3]
        env_path = project_root / ".env"
    except Exception:
        return

    if not env_path.exists():
        # 如果将来你把 .env 移到最外层 LangGraph，也做一个兜底
        alt_env = project_root.parent / ".env"
        if alt_env.exists():
            env_path = alt_env
        else:
            return

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # 不覆盖已经在进程环境里显式设置的变量
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # 加载失败时静默忽略，保持脚本可用
        pass

# 先尝试从项目根目录的 .env 加载配置，再读取环境变量
_load_env_from_project_root()

# === 配置区域：按需修改 ===
# 证据截图所在文件夹，例如：r"d:\\bishe\\LangGraph\\video_tampering_agent\\.cache\\car_detection\\demo-forgery\\images"
IMAGES_DIR = os.environ.get("VLM_IMAGES_DIR", "./images")

# 你已经写好的提示词，可以替换成自己的内容
_raw_prompt = os.environ.get(
    "VLM_PROMPT",
    "请对这两帧视频截图进行对比，判断是否存在车辆篡改或消失现象，并给出理由。",
)
# 支持在 .env 中使用 "\n" 写多行提示词
DEFAULT_PROMPT = _raw_prompt.replace("\\n", "\n") if _raw_prompt is not None else ""

# 最多发送多少对图片
MAX_PAIRS = int(os.environ.get("VLM_MAX_PAIRS", "100"))

# 使用 DashScope 的 OpenAI 兼容接口
_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("VLM_API_KEY")
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "qwen-vl-plus")

client = OpenAI(
    api_key=_DASHSCOPE_API_KEY,
    base_url=_DASHSCOPE_BASE_URL,
)


FILENAME_PATTERN = re.compile(
    r"^evidence_frame_(?P<frame>\d+)_Track_(?P<track>\d+)_(?P<phase>Pre1|Start|Post1)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_filename(path: Path):
    """从文件名中解析出 frame_index, track_id, phase(Pre1/Start/Post1)。

    期望格式：evidence_frame_38_Track_1_Pre1.jpg
    """
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    frame_idx = int(m.group("frame"))
    track_id = int(m.group("track"))
    phase = m.group("phase")
    return frame_idx, track_id, phase


def find_image_pairs(images_dir: Path, max_pairs: int = 7) -> List[Tuple[Path, Path]]:
    """在指定目录中，根据命名规则寻找 (Pre1, Start) 成对的图片。

    规则：
    - 文件名形如 evidence_frame_38_Track_1_Pre1.jpg
    - 对同一个 track_id，若存在 frame=k 的 Start，对应 pair 期望为
      frame=k-1 的 Pre1，即 (frame=k-1, Pre1) 和 (frame=k, Start)。
    - 最多返回 max_pairs 对，按 Start 帧号从小到大排序。
    """
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


def call_vlm_api(img1: Path, img2: Path, prompt: str = DEFAULT_PROMPT) -> str:
    """使用 DashScope(OpenAI 兼容)接口，发送两张图 + 提示词并返回文本结果。"""
    if not _DASHSCOPE_API_KEY:
        raise RuntimeError("请在 .env 中配置 DASHSCOPE_API_KEY 或 VLM_API_KEY")

    img1_b64 = encode_image_to_base64(img1)
    img2_b64 = encode_image_to_base64(img2)

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
        model=_MODEL_NAME,
        messages=messages,
    )

    # Qwen-VL 的兼容模式通常返回一个 message.content 列表
    choice = resp.choices[0].message
    content = choice.content
    if isinstance(content, list) and content and hasattr(content[0], "text"):
        return content[0].text
    # 兜底转成字符串
    return str(content)


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
        print(f"Saved JSON results to {json_path}")
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
        print(f"Saved Markdown results to {md_path}")
    except Exception as e:
        print(f"Warning: failed to write Markdown results: {e}")


def main():
    images_dir = Path(IMAGES_DIR).resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir not found or not a directory: {images_dir}")

    print(f"Using images dir: {images_dir}")

    pairs = find_image_pairs(images_dir, max_pairs=MAX_PAIRS)
    if not pairs:
        print("No image pairs found.")
        return

    print(f"Found {len(pairs)} pairs, sending to VLM...")

    all_results: List[Dict[str, Any]] = []

    for idx, (pre_img, start_img) in enumerate(pairs, start=1):
        print(f"\n=== Pair {idx} ===")
        print(f"Pre1 : {pre_img.name}")
        print(f"Start: {start_img.name}")
        record: Dict[str, Any] = {
            "index": idx,
            "pre_image": pre_img.name,
            "start_image": start_img.name,
        }
        try:
            text = call_vlm_api(pre_img, start_img, DEFAULT_PROMPT)
            print("VLM response:")
            print(text)
            record["response"] = text
        except Exception as e:
            print(f"Error calling VLM API for pair {idx}: {e}")
            record["error"] = str(e)

        all_results.append(record)

    _save_results(images_dir, all_results, DEFAULT_PROMPT)


if __name__ == "__main__":
    main()
