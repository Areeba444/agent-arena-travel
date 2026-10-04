"""Agent execution loop and decision management for Travel Itinerary Builder.

Enforces:
- Bounded model decision loop (max_steps budget enforcement).
- Typed Pydantic schema and semantic validation.
- Shared per-run repair path (at most one repair attempt).
- Controlled Arena fault injection (invalid_agent_decision).
- Strict output bounding and clean error degradation.
"""
import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from langchain_core.messages import BaseMessage

from app.config import settings
from app.models import (
    AgentDecision,
    AgentState,
    ArenaRequest,
    ArenaResponse,
    Metrics,
    validate_decision_semantics,
)
from app.prompts import (
    build_messages,
    build_repair_message,
    input_too_large,
)
from app.tools import call_tool

logger = logging.getLogger("agent")


class ModelReply(BaseModel):
    """Normalized response from an LLM call."""
    model_config = ConfigDict(extra='ignore')

    text: str = ""
    finish_reason: Optional[str] = None  # e.g., 'stop', 'length', 'max_tokens'
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


ModelCaller = Callable[[List[BaseMessage]], Awaitable[ModelReply]]


def _create_default_model_caller(model_identifier: str) -> ModelCaller:
    """Creates a default model caller using LangChain or returns a safe unconfigured stub."""
    async def _caller(messages: List[BaseMessage]) -> ModelReply:
        provider = settings.model_provider.lower()
        model_name = model_identifier if model_identifier != 'unconfigured' else settings.model_name

        if provider == 'groq' and model_name:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model=model_name,
                    max_tokens=settings.max_output_tokens,
                    temperature=0.1,
                )
                res = await llm.ainvoke(messages)
                
                # Extract token usage if present
                input_tokens = None
                output_tokens = None
                if hasattr(res, "usage_metadata") and res.usage_metadata:
                    input_tokens = res.usage_metadata.get("input_tokens")
                    output_tokens = res.usage_metadata.get("output_tokens")
                
                # Extract finish reason
                finish_reason = None
                if hasattr(res, "response_metadata") and res.response_metadata:
                    finish_reason = res.response_metadata.get("finish_reason")

                return ModelReply(
                    text=str(res.content),
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
            except ImportError:
                logger.warning("langchain_groq package not installed; falling back to unconfigured response.")
            except Exception as e:
                logger.error("Provider invocation error: %s", e)
                raise

        # Default fallback when unconfigured
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
            output_tokens=None
        )

    return _caller


async def run_agent(
    request: ArenaRequest,
    history: Optional[List[BaseMessage]] = None,
    model: Any = "unconfigured",
) -> ArenaResponse:
    """Executes the travel agent bounded loop.

    Guarantees:
    - Never crashes or loops indefinitely.
    - Always returns a valid ArenaResponse with steps <= max_steps.
    - Code enforces validation, stopping conditions, and repair limits.
    """
    # 1. Resolve Model Caller
    if callable(model):
        caller: ModelCaller = model
    else:
        caller = _create_default_model_caller(str(model))

    # 2. Initialize Agent State
    start_time = time.perf_counter()
    max_steps = min(request.arena_config.max_steps, settings.max_steps)
    state = AgentState(
        goal=request.task,
        step_count=0,
        max_steps=max_steps,
        repair_used=False,
        fault_fired=False,
        bookings=[],
        conflicts=[],
        itinerary=None,
        observations=[],
        known_facts=[],
        last_observation="",
        status="in_progress",
        stop_reason="",
        tool_calls=[],
        errors=[],
        events=[],
        model_calls=0,
        input_tokens=None,
        output_tokens=None,
    )

    def _build_final_response(status: str, message: str, stop_reason: str) -> ArenaResponse:
        """Helper to build a strictly validated ArenaResponse."""
        clean_msg = (message or "").strip()
        if not clean_msg:
            clean_msg = f"Execution stopped with status: {status} ({stop_reason})"
        if len(clean_msg) > 2000:
            clean_msg = clean_msg[:1980] + " ... [truncated]"

        state.events.append({
            "step": min(6, state.step_count),
            "event": "agent_stop",
            "status": status,
            "reason": stop_reason,
        })

        return ArenaResponse(
            arena_version="0.1",
            request_id=request.request_id,
            status=status,
            final_response=clean_msg,
            steps=min(6, max(0, state.step_count)),
            stop_reason=stop_reason,
            tool_calls=state.tool_calls,
            errors=state.errors,
            events=state.events,
            metrics=Metrics(
                latency_ms=0,  # arena.py calculates total latency
                model_calls=min(6, state.model_calls),
                input_tokens=state.input_tokens,
                output_tokens=state.output_tokens,
                estimated_cost_usd=None
            )
        )

    # 3. Input Budget Guard (before loop, no model call)
    if input_too_large(request):
        state.errors.append({"step": 0, "error": "Task and external context exceeded input budget."})
        return _build_final_response(
            status="budget_exceeded",
            message="The provided input size exceeded the maximum allowed character budget.",
            stop_reason="input_budget_exceeded"
        )

    # Track fault injection for model decision
    invalid_decision_fault_fired = False
    pending_repair_message: Optional[BaseMessage] = None

    # 4. Main Bounded Loop
    while state.step_count < state.max_steps:
        # Wall-clock Guard: Check remaining time budget
        elapsed = time.perf_counter() - start_time
        remaining = settings.run_timeout_seconds - elapsed - 2.0
        if remaining <= 1.0:
            state.errors.append({"step": state.step_count, "error": "Run reached time budget limit."})
            return _build_final_response("budget_exceeded", "The run reached the time budget limit.", "time_budget_reached")

        # Assemble message layers
        messages = build_messages(
            request=request,
            history=history,
            state=state,
            observations=state.observations[-3:] if state.observations else []
        )
        if pending_repair_message is not None:
            messages.append(pending_repair_message)
            pending_repair_message = None

        # Execute Model Call
        state.step_count += 1
        state.model_calls += 1
        state.events.append({"step": state.step_count, "event": "model_call"})

        model_timeout = min(getattr(settings, "model_call_timeout_seconds", 10.0), max(0.5, remaining))
        try:
            reply: ModelReply = await asyncio.wait_for(caller(messages), timeout=model_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            state.errors.append({"step": state.step_count, "error": "Model call timed out."})
            return _build_final_response("budget_exceeded", "The model call timed out.", "model_timeout")
        except Exception as e:
            state.errors.append({"step": state.step_count, "error": f"Model invocation error: {e}"})
            return _build_final_response("failed", "The agent failed to receive a model response.", "model_error")

        # Accumulate token metrics
        if reply.input_tokens is not None:
            state.input_tokens = (state.input_tokens or 0) + reply.input_tokens
        if reply.output_tokens is not None:
            state.output_tokens = (state.output_tokens or 0) + reply.output_tokens

        # Check 1: Provider Token Truncation
        if reply.finish_reason in ("length", "max_tokens"):
            state.errors.append({"step": state.step_count, "error": "Model output truncated by token limit."})
            return _build_final_response(
                status="budget_exceeded",
                message="The model output was truncated due to output token budget limit.",
                stop_reason="token_budget_reached"
            )

        raw_text = (reply.text or "").strip()

        # Check 2: Fault Injection 'invalid_agent_decision' (first model reply only)
        fault_type = request.arena_config.fault.type if request.arena_config.fault else "none"
        if fault_type == "invalid_agent_decision" and not invalid_decision_fault_fired:
            invalid_decision_fault_fired = True
            raw_text = "{ corrupted_decision_payload: true "  # invalid JSON triggers repair path

        # Check 3: Empty / Whitespace Output
        repair_error: Optional[str] = None
        decision: Optional[AgentDecision] = None

        if not raw_text:
            repair_error = "Model returned empty or whitespace response."
        else:
            # Check 4: Parse JSON into AgentDecision
            text_to_parse = raw_text
            if text_to_parse.startswith("```"):
                lines = text_to_parse.splitlines()
                if len(lines) >= 2 and lines[-1].startswith("```"):
                    text_to_parse = "\n".join(lines[1:-1]).strip()
            try:
                parsed_data = json.loads(text_to_parse)
                decision = AgentDecision.model_validate(parsed_data)
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                repair_error = f"Malformed JSON or invalid AgentDecision schema: {e}"

        # Check 5: Semantic Validation (if parsing succeeded)
        if decision is not None and repair_error is None:
            semantic_errors = validate_decision_semantics(decision)
            if semantic_errors:
                repair_error = "; ".join(semantic_errors)

        # ─── SHARED REPAIR PATH ─────────────────────────────────────
        if repair_error is not None:
            repair_error = repair_error[:300]
            state.errors.append({"step": state.step_count, "error": repair_error})
            state.events.append({
                "step": state.step_count,
                "event": "validation_failure",
                "error": repair_error
            })

            # Check if repair is allowed
            if not state.repair_used:
                if state.step_count < state.max_steps:
                    state.repair_used = True
                    pending_repair_message = build_repair_message(repair_error, invalid_output=raw_text)
                    state.events.append({"step": state.step_count, "event": "repair_attempt"})
                    continue
                else:
                    return _build_final_response(
                        status="contract_error",
                        message="Decision validation failed and no steps remain for repair.",
                        stop_reason="no_steps_remaining_for_repair"
                    )
            else:
                return _build_final_response(
                    status="contract_error",
                    message=f"Decision validation failed after repair attempt: {repair_error}",
                    stop_reason="contract_error"
                )

        # Decision is valid
        assert decision is not None
        logger.info("Step %d decision: status=%s, action=%s", state.step_count, decision.status, decision.action)
        state.events.append({
            "step": state.step_count,
            "event": "decision_validated",
            "status": decision.status,
            "action": decision.action
        })

        # ─── ACT ON DECISION STATUS ─────────────────────────────────

        # A. Status: continue (execute tool via gateway)
        if decision.status == "continue":
            tool_name = str(decision.action)
            tool_result = await call_tool(
                name=tool_name,
                raw_args=decision.arguments,
                state=state,
                fault=request.arena_config.fault,
                settings=settings
            )

            state.events.append({
                "step": state.step_count,
                "event": "tool_execution",
                "tool": tool_name,
                "outcome": tool_result.outcome
            })

            if tool_result.ok:
                obs = f"Tool '{tool_name}' succeeded. Result: {json.dumps(tool_result.data)}"
                state.observations.append(obs)
                state.last_observation = obs
                continue
            else:
                if tool_result.outcome == "rejected":
                    obs = f"Tool '{tool_name}' was rejected: {tool_result.error}"
                    state.observations.append(obs)
                    state.last_observation = obs
                    continue
                else:
                    state.errors.append({"step": state.step_count, "error": tool_result.error})
                    return _build_final_response(
                        status="tool_error",
                        message=f"Tool '{tool_name}' failed after retries: {tool_result.error}",
                        stop_reason="tool_failure"
                    )

        # B. Status: completed (Finish Guard: requires itinerary)
        elif decision.status == "completed":
            if state.itinerary is not None:
                summary_lines = ["\n\nItinerary Schedule:"]
                if state.itinerary.get("title"):
                    summary_lines.append(f"Trip: {state.itinerary.get('title')}")
                if "days" in state.itinerary:
                    for day in state.itinerary["days"]:
                        summary_lines.append(f"\nDate: {day.get('date', '')}")
                        for item in day.get("items", []):
                            t = item.get("start_time") or item.get("start_text") or ""
                            title = item.get("title") or ""
                            summary_lines.append(f"  - [{t}] {title}")
                if state.itinerary.get("conflicts"):
                    summary_lines.append("\nFlagged Conflicts:")
                    for conf in state.itinerary["conflicts"]:
                        summary_lines.append(f"  - {conf.get('description', '')}")

                full_msg = (decision.user_message or "Itinerary assembled successfully.") + "\n".join(summary_lines)
                if len(full_msg) > 2000:
                    full_msg = full_msg[:1980] + " ... [truncated]"

                return _build_final_response(
                    status="completed",
                    message=full_msg,
                    stop_reason="completed"
                )
            else:
                obs = "Cannot complete: no itinerary has been built yet. Please call build_itinerary first."
                state.observations.append(obs)
                state.last_observation = obs
                continue

        # C. Status: needs_clarification
        elif decision.status == "needs_clarification":
            msg = decision.user_message or "Clarification is needed to proceed."
            if state.bookings:
                summary_items = [
                    f"- [{b.confirmation_code}] {b.title} ({b.start_text} to {b.end_text})"
                    for b in state.bookings
                ]
                summary_block = "\n\nKnown so far:\n" + "\n".join(summary_items)
                if len(summary_block) > 600:
                    summary_block = summary_block[:580] + " ... [truncated]"
                msg += summary_block
            return _build_final_response("needs_clarification", msg, "needs_clarification")

        # D. Status: blocked
        elif decision.status == "blocked":
            return _build_final_response(
                status="blocked",
                message=decision.user_message or "Requested action is blocked.",
                stop_reason="blocked"
            )

        # E. Status: approval_required
        elif decision.status == "approval_required":
            return _build_final_response(
                status="approval_required",
                message=decision.user_message or "Human approval is required for this action.",
                stop_reason="approval_required"
            )

        # F. Status: failed
        elif decision.status == "failed":
            return _build_final_response(
                status="failed",
                message=decision.user_message or "Agent encountered an irrecoverable failure.",
                stop_reason="model_failed"
            )

    # 5. Fall out of loop: Step Budget Reached
    return _build_final_response(
        status="budget_exceeded",
        message="The agent reached the maximum allowed step budget without finishing.",
        stop_reason="step_budget_reached"
    )
