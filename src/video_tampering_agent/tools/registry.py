"""Simple registry for pluggable detection tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.tools import BaseTool

class ToolLike(Protocol):
    name: str
    description: str

    def invoke(self, *args, **kwargs) -> Any:  # pragma: no cover - protocol definition
        ...

    async def ainvoke(self, *args, **kwargs) -> Any:  # pragma: no cover - protocol definition
        ...

ToolFactory = Callable[[], ToolLike | BaseTool]

class ToolRegistry:
    """Central location to wire real tampering detection tools."""

    def __init__(self) -> None:
        self._factories: dict[str, ToolFactory] = {}

    def register(
        self,
        name: str,
        factory: ToolFactory | None = None,
        *,
        override: bool = False,
    ) -> ToolFactory:
        def decorator(builder: ToolFactory) -> ToolFactory:
            if not override and name in self._factories:
                msg = f"Tool {name!r} already registered"
                raise ValueError(msg)
            self._factories[name] = builder
            return builder

        if factory is not None:
            return decorator(factory)
        return decorator

    def get(self, name: str) -> ToolLike | BaseTool:
        try:
            factory = self._factories[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Tool {name!r} not registered") from exc
        tool = factory()
        if isinstance(tool, BaseTool):
            return tool
        return tool

    def all_tools(self) -> list[ToolLike | BaseTool]:
        return [factory() for factory in self._factories.values()]
