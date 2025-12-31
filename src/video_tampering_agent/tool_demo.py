"""Command-line demo that exercises all registered tools on a single video."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from .config import AgentSettings
from .tools import ToolRegistry, register_builtin_tools
from .workflow import build_detection_runnable


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return registry


async def run_once(video_locator: str, question: str, metadata: Dict[str, Any]) -> None:
    settings = AgentSettings()
    registry = build_registry()
    runnable = build_detection_runnable(registry, agent_settings=settings)

    initial_state = {
        "request_id": f"demo-{Path(video_locator).stem}",
        "video_locator": video_locator,
        "video_metadata": metadata,
        "messages": [HumanMessage(content=question)],
        "audit_trail": ["tool-demo: start"],
    }

    # Use astream to show progress
    final_state = {}
    async for event in runnable.astream(initial_state):
        node = event.get("metadata", {}).get("langgraph_node")
        # astream returns chunks, we need to extract the state update
        # The event structure depends on stream_mode, default is 'updates'
        # But here we iterate over the stream directly.
        # Let's just print what's happening based on the node name if available
        # Actually, standard astream yields output of each node.
        
        # For LangGraph, event is usually a dict where key is node name and value is state update
        for node_name, state_update in event.items():
            final_state.update(state_update) # Keep track of final state
            
            if node_name == "initialize":
                print("[stage] 初始化完成")
            elif node_name == "planning":
                print("[stage] 规划完成")
            elif node_name == "reflect_on_findings":
                decision = state_update.get("reflection_decision")
                notes = state_update.get("reflection_notes", [])
                print(f"\n[reflection]  正在反思初筛结果...")
                for note in notes:
                    print(f"  - 思考: {note}")
                
                if decision == "run_deferred":
                    print(f"  => 决定:  触发深查阶段 (Run Deferred Tools)")
                elif decision == "finalize":
                    print(f"  => 决定:  结束分析 (Finalize)")
            elif node_name == "finalize":
                print("[stage] 分析完成")

    result_state = final_state
    # Fallback if final_state is incomplete (e.g. if astream behavior differs)
    # But usually for this simple graph it should work.
    # If not, we can just use invoke() but we lose streaming.
    # Let's trust the accumulated state.
    
    # Merge with initial state to ensure we have all keys if some nodes didn't return them
    full_state = initial_state.copy()
    full_state.update(result_state)
    result_state = full_state

    status = result_state.get(
        "detection_status",
        {"verdict": "unknown", "confidence": 0.0, "rationale": "无检测结论"},
    )

    print("=== Detection Status ===")
    print(f"verdict: {status['verdict']}")
    print(f"confidence: {status['confidence']:.2f}")
    print(f"rationale:\n{status['rationale']}")

    print("\n--- Evidence ---")
    evidence = result_state.get("evidence", [])
    if not evidence:
        print("(no evidence)")
    for item in evidence:
        tool_name = item.get("tool") or item.get("tool_name", "unknown")
        detail = item.get("detail", item)
        print(f"* tool: {tool_name}")
        print(f"  detail: {detail}")

    print("\n--- Audit Trail ---")
    for entry in result_state.get("audit_trail", []):
        print(f"- {entry}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle tampering tools on a single video")
    parser.add_argument("video", help="视频路径或 file:// URL")
    parser.add_argument(
        "--question",
        default="请分析视频中车辆轨迹是否存在篡改迹象，并给出理由。",
        help="传递给智能体的初始问题",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="(可选) 视频时长，将写入元数据供工具参考",
    )
    parser.add_argument(
        "--extra-metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="附加元数据，格式为 key=value，可重复使用",
    )
    return parser.parse_args()


def build_metadata(duration: float, kv_pairs: list[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if duration > 0:
        metadata["duration_seconds"] = duration
    for pair in kv_pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        metadata[key] = value
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_metadata(args.duration, args.extra_metadata)
    asyncio.run(run_once(args.video, args.question, metadata))


if __name__ == "__main__":
    main()
