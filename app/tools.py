"""Deterministic sandbox tools and execution gateway for Travel Itinerary Builder.

Tools are pure operations over AgentState data; the LLM performs raw text extraction,
and tools only receive typed Pydantic models.
All executions route through call_tool, which enforces timeouts, retries,
output validation, and controlled Arena fault injection.
"""
import asyncio
import time
from typing import Any, Callable, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models import (
    AgentState,
    BookingRecord,
    BuildItineraryInput,
    BuildItineraryOutput,
    CheckConflictsInput,
    CheckConflictsOutput,
    Fault,
    RecordBookingsInput,
    RecordBookingsOutput,
    ToolTrace,
)
from app.config import settings as default_settings


class ToolRejectedError(Exception):
    """Raised when a tool rejects execution due to business logic rules."""
    pass


class ToolResult(BaseModel):
    """Normalized response from call_tool gateway. Never raises."""
    model_config = ConfigDict(extra='forbid')

    ok: bool
    outcome: Literal['success', 'timeout', 'malformed_output', 'rejected', 'exception']
    data: Optional[dict] = None
    error: Optional[str] = None


# ─── Helper Functions ───────────────────────────────────────────────

def _check_interval_overlap(s1, e1, s2, e2):
    """Checks [start, end) interval overlap for two normalized UTC datetime ranges.

    Supports point events (end == start) and overnight spans.
    Adjacent intervals [10:00, 12:00) and [12:00, 14:00) do NOT overlap.
    """
    if s1 == e1 and s2 == e2:
        if s1 == s2:
            return True, s1, s1
        return False, None, None
    if s1 == e1:
        if s2 <= s1 < e2:
            return True, s1, s1
        return False, None, None
    if s2 == e2:
        if s1 <= s2 < e1:
            return True, s2, s2
        return False, None, None
    if s1 < e2 and s2 < e1:
        return True, max(s1, s2), min(e1, e2)
    return False, None, None


def _detect_conflicts(bookings: list[BookingRecord]) -> list[dict]:
    """Pure conflict detection across bookings in the same category."""
    conflicts: list[dict] = []
    n = len(bookings)
    for i in range(n):
        for j in range(i + 1, n):
            b1 = bookings[i]
            b2 = bookings[j]
            # Conflict matrix: only bookings in the SAME category can conflict
            if b1.category != b2.category:
                continue

            overlaps, ov_start, ov_end = _check_interval_overlap(
                b1.start_datetime, b1.end_datetime,
                b2.start_datetime, b2.end_datetime
            )
            if overlaps and ov_start is not None and ov_end is not None:
                conflicts.append({
                    "category": b1.category,
                    "booking_1": {
                        "confirmation_code": b1.confirmation_code,
                        "title": b1.title,
                        "start_datetime": b1.start_datetime.isoformat(),
                        "end_datetime": b1.end_datetime.isoformat(),
                        "start_text": b1.start_text,
                        "end_text": b1.end_text,
                    },
                    "booking_2": {
                        "confirmation_code": b2.confirmation_code,
                        "title": b2.title,
                        "start_datetime": b2.start_datetime.isoformat(),
                        "end_datetime": b2.end_datetime.isoformat(),
                        "start_text": b2.start_text,
                        "end_text": b2.end_text,
                    },
                    "overlap_start": ov_start.isoformat(),
                    "overlap_end": ov_end.isoformat(),
                    "description": (
                        f"Conflict in {b1.category}: '{b1.title}' ({b1.confirmation_code}) "
                        f"overlaps with '{b2.title}' ({b2.confirmation_code}) from "
                        f"{ov_start.isoformat()} to {ov_end.isoformat()}."
                    )
                })
    return conflicts


# ─── Pure Tool Implementations ──────────────────────────────────────
# Each tool returns a tuple of (output_model, state_mutator_callable).
# State mutations are applied by the gateway ONLY after output validation succeeds.

def tool_record_bookings(state: AgentState, input_data: RecordBookingsInput) -> tuple[RecordBookingsOutput, Callable[[AgentState], None]]:
    """Records extracted bookings into state.

    Replaces existing bookings with matching confirmation_code; appends new ones.
    Clears stored conflicts and itinerary whenever the booking list changes.
    """
    existing_map = {b.confirmation_code: b for b in state.bookings}
    for b in input_data.bookings:
        existing_map[b.confirmation_code] = b
    new_bookings = list(existing_map.values())

    output = RecordBookingsOutput(
        status="success",
        recorded=len(new_bookings),
        bookings=[b.model_dump(mode="json") for b in new_bookings]
    )

    def mutate(s: AgentState) -> None:
        s.bookings = new_bookings
        s.conflicts = []
        s.itinerary = None

    return output, mutate


def tool_check_conflicts(state: AgentState, input_data: CheckConflictsInput) -> tuple[CheckConflictsOutput, Callable[[AgentState], None]]:
    """Checks for datetime interval conflicts among same-category bookings.

    Rejects if no bookings are currently recorded.
    """
    if not state.bookings:
        raise ToolRejectedError("No bookings recorded to check conflicts.")

    conflicts = _detect_conflicts(state.bookings)
    output = CheckConflictsOutput(
        status="success",
        conflicts=conflicts,
        conflict_count=len(conflicts)
    )

    def mutate(s: AgentState) -> None:
        s.conflicts = conflicts

    return output, mutate


def tool_build_itinerary(state: AgentState, input_data: BuildItineraryInput) -> tuple[BuildItineraryOutput, Callable[[AgentState], None]]:
    """Builds a day-by-day itinerary grouped by UTC day and sorted by start time.

    Displays original start_text/end_text. Recomputes conflicts and includes them.
    Rejects if no bookings are currently recorded.
    """
    if not state.bookings:
        raise ToolRejectedError("No bookings recorded to build an itinerary.")

    conflicts = _detect_conflicts(state.bookings)

    # Group bookings by UTC start date
    grouped: dict[str, list[BookingRecord]] = {}
    for b in sorted(state.bookings, key=lambda x: x.start_datetime):
        day_str = b.start_datetime.strftime("%Y-%m-%d")
        grouped.setdefault(day_str, []).append(b)

    days = []
    for day_str in sorted(grouped.keys()):
        day_items = grouped[day_str]
        days.append({
            "date": day_str,
            "items": [
                {
                    "confirmation_code": b.confirmation_code,
                    "title": b.title,
                    "category": b.category,
                    "type": b.type,
                    "start_time": b.start_text,
                    "end_time": b.end_text,
                    "location": b.location,
                    "details": b.details
                }
                for b in day_items
            ]
        })

    sorted_dates = sorted(grouped.keys())
    itinerary_dict = {
        "title": input_data.title or "My Trip",
        "start_date": input_data.start_date or (sorted_dates[0] if sorted_dates else None),
        "end_date": input_data.end_date or (sorted_dates[-1] if sorted_dates else None),
        "days": days,
        "total_bookings": len(state.bookings),
        "conflicts": conflicts
    }

    output = BuildItineraryOutput(
        status="success",
        itinerary=itinerary_dict,
        conflicts=conflicts,
        conflict_count=len(conflicts)
    )

    def mutate(s: AgentState) -> None:
        s.conflicts = conflicts
        s.itinerary = itinerary_dict

    return output, mutate


# ─── Tool Registry ──────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "record_bookings": {
        "input_model": RecordBookingsInput,
        "output_model": RecordBookingsOutput,
        "func": tool_record_bookings,
    },
    "check_conflicts": {
        "input_model": CheckConflictsInput,
        "output_model": CheckConflictsOutput,
        "func": tool_check_conflicts,
    },
    "build_itinerary": {
        "input_model": BuildItineraryInput,
        "output_model": BuildItineraryOutput,
        "func": tool_build_itinerary,
    },
}

# Exposed tool mapping
TOOLS = {k: v["func"] for k, v in TOOL_REGISTRY.items()}


# ─── Unified Execution Gateway ──────────────────────────────────────

async def call_tool(
    name: str,
    raw_args: dict,
    state: AgentState,
    fault: Any = None,
    settings: Any = None,
) -> ToolResult:
    """Unified execution gateway for all tool invocations.

    Guarantees:
    - Never raises an unhandled exception.
    - Unknown tool or schema validation failure -> outcome 'rejected', 1 trace, NO retries.
    - Retries ONLY timeouts, exceptions, and malformed outputs (up to max_tool_retries).
    - Injects Arena faults (tool_timeout, malformed_tool_output) once per run on the first real tool attempt.
    - State mutations apply ONLY after output validation succeeds.
    """
    active_settings = settings or default_settings
    max_retries = getattr(active_settings, "max_tool_retries", 2)
    timeout_seconds = getattr(active_settings, "tool_timeout_seconds", 5.0)
    current_step = max(1, min(state.step_count if (state and state.step_count > 0) else 1, 6))

    # 1. Check tool existence
    if name not in TOOL_REGISTRY:
        trace = ToolTrace(
            step=current_step,
            tool=name,
            attempt=1,
            outcome="rejected",
            latency_ms=0.0
        )
        state.tool_calls.append(trace)
        return ToolResult(
            ok=False,
            outcome="rejected",
            error=f"Unknown tool '{name}'. Allowed tools: {', '.join(sorted(TOOL_REGISTRY.keys()))}"
        )

    spec = TOOL_REGISTRY[name]
    input_model = spec["input_model"]
    output_model = spec["output_model"]
    func = spec["func"]

    # 2. Strict Input Argument Validation
    t0 = time.perf_counter()
    try:
        validated_input = input_model.model_validate(raw_args or {})
    except ValidationError as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        trace = ToolTrace(
            step=current_step,
            tool=name,
            attempt=1,
            outcome="rejected",
            latency_ms=round(latency_ms, 2)
        )
        state.tool_calls.append(trace)
        return ToolResult(
            ok=False,
            outcome="rejected",
            error=f"Invalid arguments for tool '{name}': {e}"
        )

    # Extract fault type
    fault_type = "none"
    if isinstance(fault, Fault):
        fault_type = fault.type
    elif isinstance(fault, dict):
        fault_type = fault.get("type", "none")
    elif isinstance(fault, str):
        fault_type = fault

    last_outcome: Literal["timeout", "malformed_output", "exception"] = "exception"
    last_error: str = "Tool execution failed"

    # 3. Execution & Retry Loop (Attempt 1 .. 1 + max_retries)
    for attempt in range(1, 1 + max_retries + 1):
        att_t0 = time.perf_counter()

        # Check for single fault injection per run
        inject_fault = None
        if fault_type in ("tool_timeout", "malformed_tool_output") and not state.fault_fired:
            state.fault_fired = True
            inject_fault = fault_type

        try:
            # Handle injected tool_timeout fault
            if inject_fault == "tool_timeout":
                await asyncio.sleep(0.01)
                raise asyncio.TimeoutError("Simulated tool_timeout fault")

            # Execute tool in a thread with timeout
            output_obj, mutate_fn = await asyncio.wait_for(
                asyncio.to_thread(func, state, validated_input),
                timeout=timeout_seconds
            )

            # Handle injected malformed_tool_output fault
            if inject_fault == "malformed_tool_output":
                raw_output_data = {"corrupted_fault": True, "missing_required_fields": None}
            else:
                raw_output_data = output_obj.model_dump(mode="json") if hasattr(output_obj, "model_dump") else output_obj

            # 4. Strict Output Validation
            validated_output = output_model.model_validate(raw_output_data)

            # 5. Output valid -> Apply mutations to state & record success trace
            mutate_fn(state)
            latency_ms = (time.perf_counter() - att_t0) * 1000.0
            trace = ToolTrace(
                step=current_step,
                tool=name,
                attempt=attempt,
                outcome="success",
                latency_ms=round(latency_ms, 2)
            )
            state.tool_calls.append(trace)

            return ToolResult(
                ok=True,
                outcome="success",
                data=validated_output.model_dump(mode="json")
            )

        except ToolRejectedError as e:
            # Business rule rejection (e.g. no bookings) -> outcome 'rejected', NO retry
            latency_ms = (time.perf_counter() - att_t0) * 1000.0
            trace = ToolTrace(
                step=current_step,
                tool=name,
                attempt=attempt,
                outcome="rejected",
                latency_ms=round(latency_ms, 2)
            )
            state.tool_calls.append(trace)
            return ToolResult(
                ok=False,
                outcome="rejected",
                error=str(e)
            )

        except (asyncio.TimeoutError, TimeoutError):
            latency_ms = (time.perf_counter() - att_t0) * 1000.0
            trace = ToolTrace(
                step=current_step,
                tool=name,
                attempt=attempt,
                outcome="timeout",
                latency_ms=round(latency_ms, 2)
            )
            state.tool_calls.append(trace)
            last_outcome = "timeout"
            last_error = f"Tool '{name}' timed out after {timeout_seconds}s (attempt {attempt})"

        except ValidationError as e:
            # Output model validation failed (or malformed output fault)
            latency_ms = (time.perf_counter() - att_t0) * 1000.0
            trace = ToolTrace(
                step=current_step,
                tool=name,
                attempt=attempt,
                outcome="malformed_output",
                latency_ms=round(latency_ms, 2)
            )
            state.tool_calls.append(trace)
            last_outcome = "malformed_output"
            last_error = f"Tool '{name}' produced malformed output: {e}"

        except Exception as e:
            # Unhandled runtime exception inside tool
            latency_ms = (time.perf_counter() - att_t0) * 1000.0
            trace = ToolTrace(
                step=current_step,
                tool=name,
                attempt=attempt,
                outcome="exception",
                latency_ms=round(latency_ms, 2)
            )
            state.tool_calls.append(trace)
            last_outcome = "exception"
            last_error = f"Tool '{name}' raised an error: {e}"

    # Retries exhausted
    return ToolResult(
        ok=False,
        outcome=last_outcome,
        error=last_error
    )
