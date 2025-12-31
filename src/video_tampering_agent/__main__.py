"""Run the FastAPI service with uvicorn."""

from __future__ import annotations

import uvicorn

from video_tampering_agent.service import create_app


def main() -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
