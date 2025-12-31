"""Video tampering detection agent package."""

from .service import create_app
from .runner import build_detection_graph

__all__ = ["create_app", "build_detection_graph"]
