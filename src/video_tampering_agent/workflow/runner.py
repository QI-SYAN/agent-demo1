"""Compile the LangGraph for the video tampering detection agent."""

from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.graph import StateGraph

from ..config import AgentSettings, settings
from ..state import AgentState
from ..tools import ToolRegistry, register_builtin_tools
from .nodes import (
    analyze_with_llm,
    build_default_system_prompt,
    execute_registered_tools,
    finalize_assessment,
    initialize_state,
    reflect_on_findings,
    route_after_planning,
    route_after_reflection,
)


def build_detection_graph(
    tool_registry: Optional[ToolRegistry] = None,
    *,
    agent_settings: Optional[AgentSettings] = None,
) -> StateGraph:
    """Create the state graph wiring for the detection agent."""

    registry = tool_registry or ToolRegistry()
    register_builtin_tools(registry)
    cfg = agent_settings or settings
    system_prompt = build_default_system_prompt(cfg)

    graph = StateGraph(AgentState)
    graph.add_node("initialize", partial(initialize_state, system_prompt=system_prompt))
    graph.add_node("planning", partial(analyze_with_llm, settings=cfg, registry=registry))
    graph.add_node("tools", partial(execute_registered_tools, registry=registry))
    graph.add_node("reflection", partial(reflect_on_findings, max_retries=cfg.max_replan_loops))
    graph.add_node("finalize", partial(finalize_assessment, threshold=cfg.detection_threshold))

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "planning")
    graph.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )
    graph.add_edge("tools", "reflection")
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "planning": "planning",
            "tools": "tools",
            "finalize": "finalize",
        },
    )
    graph.set_finish_point("finalize")

    return graph


def build_detection_runnable(
    tool_registry: Optional[ToolRegistry] = None,
    *,
    agent_settings: Optional[AgentSettings] = None,
):
    """Helper returning a compiled runnable ready to be invoked."""

    graph = build_detection_graph(tool_registry, agent_settings=agent_settings)
    return graph.compile()
