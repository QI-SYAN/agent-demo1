"""LangGraph nodes composing the agent."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..config import AgentSettings
from ..custom_logging import get_logger
from ..state import AgentState, DetectionStatus
from ..tools import ToolRegistry
from ..llm import build_chat_model

LOGGER = get_logger(__name__)


def initialize_state(state: AgentState, *, system_prompt: str) -> AgentState:
    """Seed the conversation and audit trail."""

    LOGGER.debug("initialize_state", state_keys=list(state.keys()))
    audit = list(state.get("audit_trail", []))
    audit.append("initialize_state: received request")
    if state.get("messages"):
        new_messages: list[BaseMessage] = []
    else:
        new_messages = [SystemMessage(content=system_prompt)]
    return {
        "messages": new_messages,
        "audit_trail": audit,
        "retry_count": 0,
        "reflection_notes": [],
    }


def _extract_planned_tools(text: str, registry: ToolRegistry) -> list[str]:
    """Very lightweight parser to discover which tools the LLM intends to use.

    Strategy:
    - Lowercase whole response
    - Match registered tool names directly
    - Also allow simple Chinese synonyms hard-coded for current tools
    - Preserve registry ordering for deterministic execution order
    """
    lowered = text.lower()
    planned: list[str] = []
    # Build alias map (could be extended / externalized later)
    aliases: dict[str, tuple[str, ...]] = {
        "car_detection": ("car_detection", "车辆检测", "车辆轨迹"),
        "background_residual": ("background_residual", "背景残差", "背景建模"),
    }
    for tool in registry.all_tools():
        name = getattr(tool, "name", "")
        if not name:
            continue
        for token in aliases.get(name, (name,)):
            if token.lower() in lowered:
                planned.append(name)
                break
    # Deduplicate while keeping order
    seen: set[str] = set()
    ordered: list[str] = []
    for item in planned:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


async def analyze_with_llm(state: AgentState, *, settings: AgentSettings, registry: ToolRegistry) -> AgentState:
    """Let the selected LLM propose an initial assessment plan and select tools dynamically."""

    chat_model = build_chat_model(settings.model)
    # Dynamically enumerate tools for the prompt (include description if available)
    tool_specs: list[str] = []
    for tool in registry.all_tools():
        name = getattr(tool, "name", "unknown")
        desc = getattr(tool, "description", "")
        if desc:
            tool_specs.append(f"{name}（{desc}）")
        else:
            tool_specs.append(name)
    available_tools = ", ".join(tool_specs)
    reflection_notes = state.get("reflection_notes") or []
    reflection_text = (
        " 上一轮工具反馈：" + "；".join(reflection_notes) + "。请针对这些问题调整策略。"
        if reflection_notes
        else ""
    )
    human_prompt = (
        "你是视频取证分析师。请制定分阶段的分析计划：\n"
        "1. **初筛阶段 (Initial)**：必须先使用 `background_residual` 和 `video_metadata` (如果可用) 进行多维度扫描。\n"
        "2. **深查阶段 (Deferred)**：将 `car_detection` 列为深查工具。注意：如果初筛发现任何背景异常、元数据痕迹，或者为了排除高风险的轨迹擦除篡改，都必须在后续执行此工具。\n"
        f"当前可用工具：{available_tools}。\n"
        "请在最后给出一行 JSON: {\"initial_tools\": [\"background_residual\", \"video_metadata\"], \"deferred_tools\": [\"car_detection\"]}。\n"
        f"{reflection_text}"
    )
    audit = list(state.get("audit_trail", []))
    audit.append("analyze_with_llm: drafting plan")

    prior_messages = list(state.get("messages", []))
    request = state.get("video_locator", "未知视频")
    # 使用 f-string 并对示例 JSON 的花括号进行转义 (双花括号 -> 单花括号) 以避免 format 引发 KeyError
    plan_prompt = HumanMessage(
        content=(
            f"待检测的对象: {request}。请按照以下格式回答:\n"
            "1. 分析计划步骤（区分初筛和深查）\n"
            "2. 每个工具的调用阶段及原因\n"
            "3. 最后一行只输出 JSON: {\"initial_tools\": [...], \"deferred_tools\": [...]}"
        )
    )

    try:
        response = await chat_model.ainvoke([
            SystemMessage(content="你正在协助检测视频篡改，请采用'初筛-深查'的分阶段策略。"),
            *prior_messages,
            HumanMessage(content=human_prompt),
            plan_prompt,
        ])
    except Exception as exc:  # 兜底：LLM 不可用时给出确定性计划
        LOGGER.warning("LLM 调用失败，使用降级计划: %s", exc)
        fallback_plan = (
            "1. 初筛：使用 background_residual 和 video_metadata\n"
            "2. 深查：使用 car_detection\n"
            '{"initial_tools": ["background_residual", "video_metadata"], "deferred_tools": ["car_detection"]}'
        )
        response = AIMessage(content=fallback_plan)

    raw_content = str(response.content)
    # 去除末尾常见的“请确认”语句，保持纯计划可读性
    for phrase in ["请确认以上分析计划，我将按此执行工具调用与结果分析。", "请确认以上分析计划", "请确认上述分析计划"]:
        raw_content = raw_content.replace(phrase, "").strip()

    initial_tools = []
    deferred_tools = []
    
    # Try strict JSON extraction from the last line first
    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    json_parsed = False
    if lines:
        last = lines[-1]
        if last.startswith("{") and "tools" in last:
            import json
            try:
                data = json.loads(last)
                if isinstance(data, dict):
                    initial_tools = [str(x) for x in data.get("initial_tools", []) if isinstance(x, str)]
                    deferred_tools = [str(x) for x in data.get("deferred_tools", []) if isinstance(x, str)]
                    json_parsed = True
            except Exception:  # pragma: no cover
                pass
    
    if not json_parsed:
        # Fallback: put everything found by heuristic into initial_tools
        initial_tools = _extract_planned_tools(raw_content, registry)
        deferred_tools = []

    if not initial_tools and not deferred_tools:
        # 默认回退
        initial_tools = ["background_residual", "video_metadata"]
        deferred_tools = ["car_detection"]

    audit.append("analyze_with_llm: LLM plan ready")
    audit.append(f"analyze_with_llm: plan -> {raw_content}")
    audit.append(f"analyze_with_llm: initial -> {initial_tools}, deferred -> {deferred_tools}")
    
    LOGGER.info(
        "planning_complete",
        initial_tools=initial_tools,
        deferred_tools=deferred_tools,
        plan_excerpt=raw_content[:400],
    )
    updated_messages = prior_messages + [response]
    return {
        "messages": updated_messages,
        "audit_trail": audit,
        "initial_tools": initial_tools,
        "deferred_tools": deferred_tools,
        "current_phase": "initial", # Start with initial phase
        "planning_text": raw_content,
        "reflection_notes": [],
    }


async def execute_registered_tools(
    state: AgentState, *, registry: ToolRegistry
) -> AgentState:
    """Invoke selected analysis tools sequentially.

    If the planning phase produced a non-empty `planned_tools` list, only those
    tools will be executed (in registry order filtered by that list). Otherwise
    we fall back to executing all registered tools as before.
    """

    audit = list(state.get("audit_trail", []))
    audit.append("execute_registered_tools: started")

    evidence: list[dict[str, Any]] = list(state.get("evidence", []))
    context = {
        "video_locator": state.get("video_locator"),
        "video_metadata": state.get("video_metadata", {}),
        "request_id": state.get("request_id"),
    }

    requested: list[str] = []
    phase = state.get("current_phase", "initial")
    if phase == "initial":
        requested = state.get("initial_tools", [])
    elif phase == "deferred":
        requested = state.get("deferred_tools", [])
    
    # Fallback for backward compatibility or manual overrides
    if not requested and not state.get("initial_tools") and not state.get("deferred_tools"):
         requested = list(state.get("planned_tools", []))

    all_tools = registry.all_tools()
    if requested:
        # Preserve registry order
        tools = [t for t in all_tools if getattr(t, "name", None) in requested]
        audit.append(f"execute_registered_tools: phase={phase}, filtered -> {', '.join([getattr(t,'name','?') for t in tools]) or 'NONE'}")
    else:
        # If explicit phase has no tools, we might skip execution or run all if it was a legacy state
        # But here we assume if lists are empty, we run nothing for this phase
        tools = []
        if not state.get("initial_tools") and not state.get("deferred_tools"):
             tools = all_tools # Legacy fallback

    if not tools:
        audit.append(f"execute_registered_tools: no tools for phase {phase}")
        # Don't return early, pass through to return empty evidence update if needed
        # return {"audit_trail": audit, "evidence": evidence}
    
    # We append to existing evidence, not replace it
    # But wait, the loop below appends to `evidence` list which is a copy of state['evidence']
    # We need to make sure we don't lose previous evidence if we are in deferred phase.
    # The `evidence` variable is initialized as `list(state.get("evidence", []))` at the top of function.
    # So we are good.

    for tool in tools:
        LOGGER.info("invoking_tool", tool=tool.name)
        maybe_async = getattr(tool, "ainvoke", None)
        try:
            if callable(maybe_async):
                tool_result = await maybe_async(context)
            else:
                tool_result = tool.invoke(context)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("tool_failed", tool=tool.name)
            evidence.append({"tool": tool.name, "status": "error", "detail": str(exc)})
            continue

        evidence.append({"tool": tool.name, "status": "ok", "detail": tool_result})

    audit.append("execute_registered_tools: completed")
    return {
        "audit_trail": audit,
        "evidence": evidence,
    }


def reflect_on_findings(state: AgentState, *, max_retries: int) -> AgentState:
    """Assess tool outputs to decide whether to replan, run deferred tools, or finalize."""

    audit = list(state.get("audit_trail", []))
    audit.append("reflect_on_findings: evaluating evidence")

    retry_count = int(state.get("retry_count") or 0)
    evidence = state.get("evidence", [])
    reasons: list[str] = []
    
    phase = state.get("current_phase", "initial")
    LOGGER.info(f"reflect_on_findings: analyzing evidence from {phase} phase...")

    if not evidence and not state.get("initial_tools") and not state.get("deferred_tools"):
         # Only complain if we really expected tools to run but none did
         reasons.append("未获取任何工具输出")

    for clue in evidence:
        if clue.get("status") == "error":
            reasons.append(f"工具 {clue.get('tool', 'unknown')} 执行失败: {clue.get('detail')}")

    # Check for suspicious signals in the current evidence
    has_signal = False
    reasons_for_deferred = []

    for clue in evidence:
        tool_name = clue.get("tool", "unknown")
        detail = clue.get("detail")
        
        if isinstance(detail, dict):
            # 1. 检查背景残差
            if tool_name == "background_residual":
                residual = float(detail.get("max_residual_ratio") or 0.0)
                # 调低阈值，体现“宁可错杀不可放过”的严谨性
                if residual >= 0.15: 
                    has_signal = True
                    reasons_for_deferred.append(f"背景残差({residual:.2f})存在波动")
            
            # 2. 检查元数据
            if tool_name == "video_metadata":
                if detail.get("has_editing_traces"):
                    has_signal = True
                    reasons_for_deferred.append("发现编辑软件标签")
            
            # 3. 检查轨迹消失 (如果 car_detection 已经跑了)
            if detail.get("suspected_vanish_tracks"):
                has_signal = True
                # 如果已经在 deferred 阶段，这会加强 finalize 的信心
    
    # Decision Logic
    
    # 1. Handle Errors -> Replan
    if reasons and retry_count < max_retries:
        retry_reason = "；".join(reasons)
        audit.append(f"reflect_on_findings: requesting replanning -> {retry_reason}")
        feedback_message = HumanMessage(
            content=(
                "上一轮工具执行反馈: "
                f"{retry_reason}。请提出新的分析计划或调整工具配置。"
            )
        )
        messages = list(state.get("messages", []))
        messages.append(feedback_message)
        return {
            "audit_trail": audit,
            "retry_count": retry_count + 1,
            "reflection_notes": reasons,
            "reflection_decision": "replan",
            "planned_tools": [], # Clear legacy field
            # "evidence": [], # Keep evidence? Maybe clear it if we replan completely? 
            # Usually replan means "try again", so maybe clear evidence from failed attempt.
            # But existing code cleared it. Let's keep clearing it for replan.
            "evidence": [],
            "messages": messages,
        }

    # 2. Handle Phase Transition (Initial -> Deferred)
    phase = state.get("current_phase", "initial")
    deferred_tools = state.get("deferred_tools", [])
    
    # 检查 deferred_tools 里是否有 car_detection
    has_car_det_deferred = any("car" in t for t in deferred_tools)

    if phase == "initial" and deferred_tools:
        # 逻辑分支 A: 初筛发现了问题 -> 必须深查
        if has_signal:
            note = f"初筛发现疑点({'; '.join(reasons_for_deferred)})，触发深查工具"
            LOGGER.info(f"reflect_on_findings: 发现异常，决定进入深查阶段 -> {deferred_tools}")
            audit.append(f"reflect_on_findings: {note}")
            return {
                "audit_trail": audit,
                "reflection_decision": "run_deferred",
                "current_phase": "deferred",
                "reflection_notes": [note],
            }
        
        # 逻辑分支 B: 初筛没问题，但 deferred 里有 car_detection
        # 此时 Agent 决定：“虽然背景看起来没问题，但为了排除隐蔽的轨迹擦除篡改，我必须执行车辆检测”
        elif has_car_det_deferred:
            note = "初筛未见明显异常，但为排除隐蔽的轨迹擦除篡改，决定继续执行车辆检测验证"
            LOGGER.info(f"reflect_on_findings: 启用严谨模式，决定继续执行深查 -> {deferred_tools}")
            audit.append(f"reflect_on_findings: {note}")
            return {
                "audit_trail": audit,
                "reflection_decision": "run_deferred",
                "current_phase": "deferred",
                "reflection_notes": [note],
            }
            
        else:
            audit.append("reflect_on_findings: no suspicious signals in initial phase, skipping deferred tools")
            # Fall through to finalize
    
    # 3. Finalize
    LOGGER.info("reflect_on_findings: 证据充足或无更多工具，给出结论")
    audit.append("reflect_on_findings: proceed to finalize")
    return {
        "audit_trail": audit,
        "reflection_notes": [],
        "reflection_decision": "finalize",
    }


def finalize_assessment(state: AgentState, *, threshold: float) -> AgentState:
    """Aggregate intermediate findings and emit a verdict."""

    audit = list(state.get("audit_trail", []))
    audit.append("finalize_assessment: computing verdict")

    confidence = 0.0
    residual_signal = 0.0
    vanish_signal = 0.0
    rationale_parts: list[str] = []
    error_messages: list[str] = []

    for clue in state.get("evidence", []):
        tool_name = clue.get("tool", "unknown")
        status = clue.get("status", "ok")
        detail = clue.get("detail")

        if status == "error":
            error_messages.append(f"{tool_name}: {detail}")
            continue

        entry = f"工具 {tool_name}: {detail}"
        if isinstance(detail, dict):
            entry_parts: list[str] = []
            score = detail.get("score")
            if score is not None:
                confidence = max(confidence, float(score))

            if "max_residual_ratio" in detail:
                residual = float(detail.get("max_residual_ratio") or 0.0)
                avg_residual = float(detail.get("average_residual_ratio") or 0.0)
                residual_signal = max(residual_signal, residual)
                entry_parts.append(
                    "背景残差 max={:.3f}, avg={:.3f}, suspicious_frames={}".format(
                        residual,
                        avg_residual,
                        detail.get("suspicious_frames", []),
                    )
                )

            vanish_tracks = detail.get("suspected_vanish_tracks") or []
            if vanish_tracks:
                track_count = len(vanish_tracks)
                max_ratio = detail.get("max_vanish_ratio")
                if max_ratio is None:
                    max_ratio = max(
                        (float(item.get("missing_ratio") or 0.0) for item in vanish_tracks),
                        default=0.0,
                    )
                max_ratio = float(max_ratio)
                vanish_confidence = min(1.0, 0.6 + max_ratio * 0.4)
                vanish_signal = max(vanish_signal, vanish_confidence)
                confidence = max(confidence, vanish_confidence)

                sample = vanish_tracks[:3]
                summaries: list[str] = []
                for item in sample:
                    track_id = item.get("track_id")
                    ratio = item.get("missing_ratio")
                    if ratio is not None:
                        ratio_text = f"{ratio:.1%}"
                    else:
                        missing_total = item.get("missing_total_frames", 0)
                        total_frames = item.get("total_frames", "?")
                        ratio_text = f"{missing_total}/{total_frames}"
                    summaries.append(f"#{track_id} 缺失 {ratio_text}")
                if track_count > len(sample):
                    summaries.append("其余轨迹亦检测到缺失段")

                entry_parts.append(
                    "车辆轨迹异常：{} 条轨迹存在消失段，最大缺失比例 {:.1%}。样例：{}".format(
                        track_count,
                        max_ratio,
                        "; ".join(summaries),
                    )
                )

            if entry_parts:
                entry = f"工具 {tool_name}: " + "; ".join(entry_parts)

        rationale_parts.append(entry)

    confidence = max(confidence, residual_signal, vanish_signal)

    if error_messages:
        verdict: DetectionStatus = {
            "verdict": "unknown",
            "confidence": 0.0,
            "rationale": "\n".join(
                ["以下工具执行失败，需要人工复核:", *error_messages, *rationale_parts]
            ),
        }
    elif confidence >= threshold:
        verdict = {
            "verdict": "tampered",
            "confidence": confidence,
            "rationale": "\n".join(rationale_parts) or "工具判断为篡改。",
        }
    else:
        verdict = {
            "verdict": "unknown" if not rationale_parts else "clean",
            "confidence": confidence,
            "rationale": "\n".join(rationale_parts) or "缺少足够的证据。",
        }

    audit.append("finalize_assessment: done")
    return {
        "audit_trail": audit,
        "detection_status": verdict,
    }


def route_after_planning(state: AgentState) -> str:
    """Decide next branch based on planned tool selection.

    Primary signal: `initial_tools` list populated by planning node.
    Fallback: heuristic keywords in last AI message (legacy behavior).
    """
    initial = state.get("initial_tools") or []
    # Also check legacy planned_tools for backward compatibility
    planned = state.get("planned_tools") or []
    
    if initial or planned:
        return "tools"

    # Fallback to previous heuristic if no explicit selection present
    planned_messages = [msg for msg in state.get("messages", []) if isinstance(msg, AIMessage)]
    if not planned_messages:
        return "finalize"
    last_message = planned_messages[-1]
    content = str(last_message.content)
    lowered = content.lower()
    if any(keyword in content for keyword in ("调用", "工具", "执行")) or "tool" in lowered:
        return "tools"
    if any(term in content for term in ("无需工具", "不需调用", "无需执行")):
        return "finalize"
    return "tools"


def route_after_reflection(state: AgentState) -> str:
    decision = state.get("reflection_decision")
    if decision == "replan":
        return "planning"
    if decision == "run_deferred":
        return "tools"
    return "finalize"


def build_default_system_prompt(settings: AgentSettings) -> str:
    """Construct the system prompt used for the agent."""

    return (
        "你是一个专业的多模态视频篡改检测智能体。"
        "你可以访问多个外部工具（由调用方提供），对视频帧、音频轨道、元数据进行取证分析。"
        "请记录详细推理过程，便于后续审计。"
        f"系统环境: {settings.environment}."
    )
