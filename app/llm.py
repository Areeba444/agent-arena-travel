"""LLM caller integration for Groq models with explicit rate limit handling and usage tracking."""
import asyncio
import json
import logging
import re
import time
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
    calls_made: int = 1


ModelCaller = Callable[..., Awaitable[ModelReply]]


def _parse_retry_after(err_str: str) -> float:
    """Extracts suggested retry delay in seconds from error string if available."""
    match = re.search(r'try again in ([\d\.]+)s', err_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    match_sec = re.search(r'retry[- ]after[:\s]+([\d\.]+)', err_str)
    if match_sec:
        try:
            return float(match_sec.group(1))
        except ValueError:
            pass
    return 1.5


_daily_quota_exhausted_models: set[str] = set()


def get_model_caller(model_identifier: str = "") -> ModelCaller:
    """Creates a model caller with explicit 429/timeout retries and max_retries=0."""
    selected_model = model_identifier if (model_identifier and model_identifier != "unconfigured") else settings.model_name
    allowed = settings.allowed_models_list

    if selected_model and selected_model != "unconfigured" and allowed and selected_model not in allowed:
        raise ValueError(f"Model '{selected_model}' is not permitted. Allowed models: {', '.join(allowed)}")

    async def _caller(messages: List[BaseMessage], remaining_budget: float = 40.0, per_call_timeout: Optional[float] = None) -> ModelReply:
        api_key = settings.groq_api_key

        if api_key and selected_model and selected_model != "unconfigured":
            from langchain_groq import ChatGroq

            # Select active model (bypassing any models known to have exhausted daily quota)
            active_model = selected_model
            if active_model in _daily_quota_exhausted_models and len(allowed) > 1:
                alt = next((m for m in allowed if m not in _daily_quota_exhausted_models), None)
                if alt:
                    active_model = alt

            call_timeout = per_call_timeout or min(getattr(settings, "model_call_timeout_seconds", 12.0), remaining_budget)
            calls_made = 0
            res = None
            max_attempts = 4

            for attempt in range(1, max_attempts + 1):
                calls_made += 1
                llm = ChatGroq(
                    model=active_model,
                    api_key=api_key,
                    max_tokens=settings.max_output_tokens,
                    temperature=0.1,
                    max_retries=0,
                    reasoning_format="hidden",
                    reasoning_effort="low",
                )
                try:
                    res = await asyncio.wait_for(llm.ainvoke(messages), timeout=call_timeout)
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    if attempt < max_attempts and (remaining_budget - call_timeout) >= 6.0:
                        logger.warning("Model call timed out; retrying...")
                        continue
                    raise
                except Exception as e:
                    err_str = str(e).lower()
                    is_daily_quota = "tokens per day" in err_str or "tpd" in err_str
                    is_rate_limit = "429" in err_str or "rate limit" in err_str or "ratelimit" in err_str

                    if is_daily_quota:
                        _daily_quota_exhausted_models.add(active_model)
                        alt = next((m for m in allowed if m not in _daily_quota_exhausted_models), None)
                        if alt and alt != active_model:
                            logger.warning("Daily quota reached on %s; switching to %s", active_model, alt)
                            active_model = alt
                            continue

                    if is_rate_limit and attempt < max_attempts and remaining_budget > 5.0:
                        retry_hint = _parse_retry_after(str(e))
                        wait_time = min(retry_hint, 3.5, max(1.0, remaining_budget - 4.0))
                        logger.warning("Groq rate limit (429) on %s. Retrying after %.1fs pause...", active_model, wait_time)
                        await asyncio.sleep(wait_time)
                        continue

                    # If rate limit persists and alternative allowed model exists, try it
                    if is_rate_limit and len(allowed) > 1:
                        alt = next((m for m in allowed if m != active_model and m not in _daily_quota_exhausted_models), None)
                        if alt:
                            logger.warning("429 rate limit on %s; attempting %s", active_model, alt)
                            active_model = alt
                            continue

                    raise

            if res is None:
                raise RuntimeError("Failed to receive LLM response.")

            # Extract text
            raw_content = res.content
            if isinstance(raw_content, list):
                text_parts = [str(p if isinstance(p, str) else p.get("text", "")) for p in raw_content]
                final_text = "".join(text_parts)
            else:
                final_text = str(raw_content)

            # Map finish reason
            finish_reason = "stop"
            if hasattr(res, "response_metadata") and res.response_metadata:
                raw_finish = str(res.response_metadata.get("finish_reason", "")).lower()
                if raw_finish in ("length", "max_tokens"):
                    finish_reason = "length"

            # Extract usage tokens
            input_tokens = None
            output_tokens = None
            if hasattr(res, "usage_metadata") and res.usage_metadata:
                in_tok = res.usage_metadata.get("input_tokens")
                out_tok = res.usage_metadata.get("output_tokens")
                if in_tok and isinstance(in_tok, int) and in_tok > 0:
                    input_tokens = in_tok
                if out_tok and isinstance(out_tok, int) and out_tok > 0:
                    output_tokens = out_tok

            return ModelReply(
                text=final_text,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                calls_made=calls_made
            )

        # Fallback when unconfigured
        return ModelReply(
            text=json.dumps({
                "status": "failed",
                "action": None,
                "arguments": {},
                "user_message": "Model provider is unconfigured.",
                "reasoning_summary": "Unconfigured model provider."
            }),
            finish_reason="stop",
            input_tokens=None,
            output_tokens=None,
            calls_made=1
        )

    return _caller
