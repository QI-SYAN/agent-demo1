"""LLM factory helpers."""

from __future__ import annotations
from typing import Any, Protocol
from .custom_logging import get_logger
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .config import ModelSettings


class SupportsAcall(Protocol):
    async def ainvoke(self, *args, **kwargs):  # pragma: no cover - protocol signature
        ...


class _LocalPlanner:
    """Deterministic fallback model used when remote LLM is unavailable.

    Behavior: returns a short generic plan suggestion; not stochastic.
    """

    async def ainvoke(self, messages: list[Any], **_: Any) -> AIMessage:
        return AIMessage(
            content=(
                "1. 调用 car_detection 分析车辆轨迹与异常消失\n"
                "2. 调用 background_residual 计算背景残差与可疑区域\n"
                "3. 汇总两者证据做篡改判定\n"
                '{"tools": ["car_detection", "background_residual"]}'
            )
        )


LOGGER = get_logger(__name__)


def build_chat_model(settings: ModelSettings) -> SupportsAcall:
    """Instantiate a chat model based on configuration with graceful fallback.

    Fallback conditions:
    - Missing `api_key`
    - Unsupported provider value
    Returns a deterministic local planner instead of raising.
    """

    missing_key = not settings.api_key or str(settings.api_key).strip() == ""

    if settings.provider == "openai":
        if missing_key:
            LOGGER.warning("OpenAI api_key 缺失，使用本地规划回退模型。")
            return _LocalPlanner()
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            base_url=settings.base_url,
        )

    if settings.provider == "deepseek":
        if missing_key:
            LOGGER.warning("DeepSeek api_key 缺失，使用本地规划回退模型。")
            return _LocalPlanner()
        base_url = str(settings.base_url) if settings.base_url else "https://api.deepseek.com/v1"
        return ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
            base_url=base_url,
        )

    if settings.provider == "azure_openai":
        if missing_key:
            LOGGER.warning("Azure OpenAI api_key 缺失，使用本地规划回退模型。")
            return _LocalPlanner()
        if settings.base_url is None:
            LOGGER.warning("Azure OpenAI 未设置 base_url，使用本地规划回退模型。")
            return _LocalPlanner()
        return AzureChatOpenAI(
            api_key=settings.api_key,
            azure_endpoint=str(settings.base_url),
            deployment_name=settings.model_name,
            temperature=settings.temperature,
            timeout=settings.request_timeout,
        )

    if settings.provider in {"custom", "local"}:
        return _LocalPlanner()

    LOGGER.warning("未知模型提供者 %r，使用本地规划回退模型。", settings.provider)
    return _LocalPlanner()
