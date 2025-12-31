"""Convenience wrappers for constructing the LangGraph runnable."""

from __future__ import annotations

from typing import Optional

from .config import AgentSettings
from .tools import ToolRegistry
from .workflow.runner import build_detection_graph as _build_graph
from .workflow.runner import build_detection_runnable as _build_runnable


def build_detection_graph(
    tool_registry: Optional[ToolRegistry] = None,
    *,
    agent_settings: Optional[AgentSettings] = None,
):
    return _build_graph(tool_registry, agent_settings=agent_settings)


def build_detection_runnable(
    tool_registry: Optional[ToolRegistry] = None,
    *,
    agent_settings: Optional[AgentSettings] = None,
):
    return _build_runnable(tool_registry, agent_settings=agent_settings)
