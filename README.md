# Video Tampering Detection Agent

This package scaffolds a LangGraph-powered agent that orchestrates LLM reasoning and pluggable forensic tools to detect video tampering. It exposes a FastAPI service that other systems can call once you plug in your own detection tools and knowledge base.

## Features

- Pydantic-based configuration loaded from environment variables.
- LangGraph state machine that separates planning, tool execution, and final verdict stages.
- Tool registry ready for custom analysis components (frame hashing, metadata diffing, model-based classifiers, etc.).
- FastAPI application factory with a `/detect` endpoint returning structured results.
- Structured logging via `structlog`.

## Getting Started

```cmd
cd video_tampering_agent
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

If you want to use the local LangGraph source located at `..\langgraph-main`, install it in editable mode too:

```cmd
pip install -e ..\langgraph-main
```

Create a `.env` file to hold credentials:

```
MODEL_PROVIDER=openai
MODEL_API_KEY=sk-...
MODEL_MODEL_NAME=gpt-4o-mini
```

## Plugging in Tools

Register your detection tools by extending the `ToolRegistry` in `src/video_tampering_agent/tools/registry.py`. A tool only needs a `name`, `description`, and either an `invoke` or `ainvoke` method that accepts the current state context and returns a result. Example:

```python
from video_tampering_agent.tools import ToolRegistry

registry = ToolRegistry()

@registry.register("metadata_diff", override=True)
def build_metadata_tool():
    return MyMetadataDiffTool()
```

Pass this registry into `create_app(registry)` or `build_detection_runnable(registry)` when wiring the service.

## Running the API

```cmd
uvicorn video_tampering_agent.service:create_app --factory --host 0.0.0.0 --port 8000
```

Send a request:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
        "video_locator": "s3://bucket/sample.mp4",
        "metadata": {"fps": 25},
        "prompt": "请重点检查关键帧是否被篡改"
      }'
```

## Next Steps

- Implement concrete tampering detection tools (deepfake detectors, frame hashing, audio desync analysis, etc.).
- Connect to a vector store or other knowledge base inside `execute_registered_tools`.
- Add persistence or telemetry by extending the state before returning the response.
- Write unit tests under `tests/` to validate the workflow.
