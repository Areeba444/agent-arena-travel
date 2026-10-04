"""LLM caller integration for Groq models with rate limit handling and usage tracking."""
import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, List, Optional
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import BaseMessage

from app.config import settings

logger = logging.getLogger("llm")


class ModelReply(BaseModel):
    """Normalized response from an LLM call."""
    model_config = ConfigDict(extra='ignore')

    text: str = ""
    finish_reason: Optional[str] = None  # 'stop' or 'length'
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


ModelCaller = Callable[[List[BaseMessage]], Awaitable[ModelReply]]


def get_model_caller(model_identifier: str = "") -> ModelCaller:
    """Creates a model caller for the requested model identifier.

    Enforces allowed model list, API key security, 429 retries, and usage normalization.
    """
    selected_model = model_identifier if (model_identifier and model_identifier != "unconfigured") else settings.model_name
    allowed = settings.allowed_models_list

    # Validate model selection
    if selected_model and selected_model != "unconfigured" and allowed and selected_model not in allowed:
        raise ValueError(
            f"Model '{selected_model}' is not permitted. Allowed models: {', '.join(allowed)}"
        )

    async def _caller(messages: List[BaseMessage]) -> ModelReply:
        api_key = settings.groq_api_key

        if api_key and selected_model and selected_model != "unconfigured":
            from langchain_groq import ChatGroq

            # Configure ChatGroq with reasoning_format='hidden' and reasoning_effort='low' so output tokens
            # are not wasted on internal chain-of-thought tokens for supported models
            llm = ChatGroq(
                model=selected_model,
                api_key=api_key,
                max_tokens=settings.max_output_tokens,
                temperature=0.1,
                reasoning_format="hidden",
                reasoning_effort="low",
            )

            # Execution with one bounded 429 retry
            max_429_retries = 1
            for attempt in range(max_429_retries + 1):
                try:
                    res = await llm.ainvoke(messages)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "rate limit" in err_str or "ratelimit" in err_str
                    if is_rate_limit and attempt < max_429_retries:
                        logger.warning("Groq rate limit encountered (429). Retrying after 1.5s backoff...")
                        await asyncio.sleep(1.5)
                        continue
                    logger.error("LLM invocation failed: %s", type(e).__name__)
                    raise

            # Extract text content (handling string vs list of parts)
            raw_content = res.content
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                final_text = "".join(text_parts)
            else:
                final_text = str(raw_content)

            # Map finish reason
            finish_reason = "stop"
            if hasattr(res, "response_metadata") and res.response_metadata:
                raw_finish = str(res.response_metadata.get("finish_reason", "")).lower()
                if raw_finish in ("length", "max_tokens"):
                    finish_reason = "length"

            # Extract token usage (None if missing, never 0)
            input_tokens = None
            output_tokens = None
            if hasattr(res, "usage_metadata") and res.usage_metadata:
                in_tok = res.usage_metadata.get("input_tokens")
                out_tok = res.usage_metadata.get("output_tokens")
                if in_tok is not None and isinstance(in_tok, int) and in_tok > 0:
                    input_tokens = in_tok
                if out_tok is not None and isinstance(out_tok, int) and out_tok > 0:
                    output_tokens = out_tok

            return ModelReply(
                text=final_text,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Fallback when unconfigured
        return ModelReply(
            text=json.dumps({
                "status": "failed",
                "action": None,
                "arguments": {},
                "user_message": "Model provider is unconfigured or credentials missing.",
                "reasoning_summary": "Unconfigured model provider fallback."
            }),
            finish_reason="stop",
            input_tokens=None,
            output_tokens=None,
        )

    return _caller
