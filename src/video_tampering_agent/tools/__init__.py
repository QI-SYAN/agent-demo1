"""Tool registry and built-in tool wiring for the agent."""

from __future__ import annotations

from .background_residual import BackgroundResidualTool
from .car_detection import CarDetectionTool
from .video_metadata import VideoMetadataTool
from .registry import ToolRegistry


def register_builtin_tools(registry: ToolRegistry, *, override: bool = False) -> None:
	"""Register the built-in detection tools on the given registry."""

	builtins = {
		CarDetectionTool.name: lambda: CarDetectionTool(),
		BackgroundResidualTool.name: lambda: BackgroundResidualTool(),
		VideoMetadataTool.name: lambda: VideoMetadataTool(),
	}

	for name, factory in builtins.items():
		try:
			registry.register(name, factory, override=override)
		except ValueError:
			if override:
				raise


__all__ = [
	"ToolRegistry",
	"CarDetectionTool",
	"BackgroundResidualTool",
	"VideoMetadataTool",
	"register_builtin_tools",
]
