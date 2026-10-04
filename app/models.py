"""Domain-independent HTTP contracts. Add your own decision/state models below."""
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')
class ExternalContext(Contract):
    source: str = Field(min_length=1, max_length=100)
    content: str = Field(max_length=10000)
    trust: Literal['untrusted'] = 'untrusted'
class Fault(Contract):
    type: Literal['none', 'tool_timeout', 'malformed_tool_output', 'invalid_agent_decision'] = 'none'
    trigger: Literal['first_matching_operation'] = 'first_matching_operation'
class ArenaConfig(Contract):
    max_steps: int = Field(default=6, ge=1, le=6)
    fault: Fault = Field(default_factory=Fault)
    @field_validator('fault', mode='before')
    @classmethod
    def accept_assignment_shorthand(cls, value):
        return {'type': value} if isinstance(value, str) else value
class ArenaRequest(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=80)
    task: str = Field(min_length=1, max_length=10000)
    external_context: list[ExternalContext] = Field(default_factory=list, max_length=20)
    arena_config: ArenaConfig = Field(default_factory=ArenaConfig)
    @field_validator('task')
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError('Task must not be blank')
        return value
class ToolTrace(Contract):
    step: int = Field(ge=1, le=6)
    tool: str
    attempt: int = Field(default=1, ge=1, le=3)
    outcome: Literal['success', 'timeout', 'malformed_output', 'rejected', 'exception']
    latency_ms: float = Field(default=0, ge=0)
class Metrics(Contract):
    latency_ms: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0, le=6)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
class ArenaResponse(Contract):
    arena_version: Literal['0.1'] = '0.1'
    request_id: str
    status: Literal['completed', 'needs_clarification', 'blocked', 'approval_required', 'tool_error', 'contract_error', 'budget_exceeded', 'failed']
    final_response: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=0, ge=0, le=6)
    stop_reason: str
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
class ChatRequest(ArenaRequest):
    session_id: str = Field(min_length=16, max_length=80)
    model: str = Field(default='unconfigured', max_length=120)
# ─── Domain Models: Travel Itinerary Builder ────────────────────────

# Action allowlists — checked by semantic validators, not by Pydantic.
# Tool actions route through the gateway; terminal statuses end the run.
TOOL_ACTIONS: set[str] = {'record_bookings', 'check_conflicts', 'build_itinerary'}
TERMINAL_STATUSES: set[str] = {'needs_clarification', 'completed', 'blocked', 'approval_required', 'failed'}
ALLOWED_ACTIONS: set[str] = TOOL_ACTIONS  # only tool names are valid in the action field


class AgentDecision(BaseModel):
    """Parsed from the model's JSON output each step.

    Uses extra='ignore' so stray LLM fields (e.g. "confidence", "thought")
    are silently dropped rather than triggering a repair for harmless noise.
    Only status, action, arguments, user_message, and reasoning_summary matter.
    """
    model_config = ConfigDict(extra='ignore')

    status: Literal['continue', 'needs_clarification', 'completed', 'blocked', 'approval_required', 'failed']
    action: Optional[str] = None                               # tool name, only meaningful when status='continue'
    arguments: dict = Field(default_factory=dict)               # tool arguments
    user_message: Optional[str] = Field(default=None, max_length=2000)  # required for non-continue statuses
    reasoning_summary: str = Field(default='', max_length=300)  # short explanation for tracing


class BookingRecord(Contract):
    """A single validated booking extracted by the model, stored by record_bookings.

    - category (Literal) drives the conflict matrix: same-category overlaps conflict.
    - type (free text) is for human-readable display only.
    - Datetimes are normalised to UTC on ingestion; originals kept in *_text fields.
    - end_datetime >= start_datetime is allowed (end == start OK); end < start rejected.
    """
    category: Literal['transport', 'accommodation', 'activity']
    type: str = Field(min_length=1, max_length=100)             # free text: "flight", "Airbnb", etc.
    confirmation_code: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=200)
    start_datetime: datetime                                    # normalised to UTC
    end_datetime: datetime                                      # normalised to UTC
    start_text: str = Field(min_length=1, max_length=100)       # original string for display
    end_text: str = Field(min_length=1, max_length=100)         # original string for display
    location: str = Field(min_length=1, max_length=200)
    details: Optional[str | dict] = None

    @field_validator('start_datetime', 'end_datetime', mode='before')
    @classmethod
    def normalize_to_utc(cls, v):
        """Aware → convert to UTC.  Naive → treat as UTC.

        This guarantees every comparison is apples-to-apples and can never
        raise a mixed-offset TypeError.
        """
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f'Expected ISO-8601 string or datetime, got {type(v).__name__}')
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @model_validator(mode='after')
    def end_not_before_start(self):
        """Reject bookings where end is strictly before start (end == start is allowed)."""
        if self.end_datetime < self.start_datetime:
            raise ValueError(
                f'end_datetime ({self.end_datetime.isoformat()}) must not be '
                f'before start_datetime ({self.start_datetime.isoformat()})'
            )
        return self


# ─── Tool Input / Output Schemas ────────────────────────────────────
# All use extra='forbid' (inherited from Contract) so unexpected fields
# from the model cause a clean 'rejected' outcome, not silent data loss.

class RecordBookingsInput(Contract):
    """Input for record_bookings: at least one booking required.

    The model extracts bookings from raw text and passes them here.
    Duplicate confirmation_codes replace existing records; new codes append.
    Any change clears stored conflicts and itinerary.
    """
    bookings: list[BookingRecord] = Field(min_length=1)


class RecordBookingsOutput(Contract):
    """Output from record_bookings: echoes back all stored bookings."""
    status: str
    recorded: int
    bookings: list[dict]


class CheckConflictsInput(Contract):
    """Input for check_conflicts: no arguments (operates on stored bookings)."""
    pass


class CheckConflictsOutput(Contract):
    """Output from check_conflicts: list of same-category interval overlaps."""
    status: str
    conflicts: list[dict]
    conflict_count: int


class BuildItineraryInput(Contract):
    """Input for build_itinerary.

    Returns 'rejected' if no bookings are recorded.
    Recomputes conflicts internally and includes them in output.
    """
    title: Optional[str] = Field(default='My Trip', max_length=200)
    start_date: Optional[str] = Field(default=None, max_length=50)
    end_date: Optional[str] = Field(default=None, max_length=50)


class BuildItineraryOutput(Contract):
    """Output from build_itinerary: day-by-day schedule with conflicts."""
    status: str
    itinerary: dict | None = None
    conflicts: list[dict] = Field(default_factory=list)
    conflict_count: int = 0


# ─── Per-Run Agent State ────────────────────────────────────────────

class AgentState(BaseModel):
    """Mutable state for a single agent run.  Not serialised over HTTP.

    step_count tracks every model call (including repairs).
    repair_used is a shared flag: the entire run gets at most one repair.
    """
    model_config = ConfigDict(extra='forbid')

    goal: str = ''
    step_count: int = 0
    max_steps: int = 6
    repair_used: bool = False               # shared: one repair per run total
    fault_fired: bool = False               # has the arena fault been injected this run?
    bookings: list[BookingRecord] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    itinerary: dict | None = None           # set by build_itinerary; finish guard checks this
    observations: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    last_observation: str = ''
    status: str = 'in_progress'             # terminal status when the run ends
    stop_reason: str = ''
    tool_calls: list[ToolTrace] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    model_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


# ─── Semantic Validators ────────────────────────────────────────────
# Separate from Pydantic schema validation.  These check business rules
# that the schema alone cannot express.  Called by the agent loop AFTER
# Pydantic parsing succeeds.

def validate_decision_semantics(decision: AgentDecision) -> list[str]:
    """Check business rules on a successfully parsed AgentDecision.

    Returns a list of human-readable error strings.  Empty list = valid.
    Any error here feeds into the shared repair path (one repair per run).
    """
    errors: list[str] = []

    # 'continue' must name an allowed tool action so the gateway knows what to call
    if decision.status == 'continue':
        if decision.action is None:
            errors.append("status is 'continue' but no action was specified.")
        elif decision.action not in ALLOWED_ACTIONS:
            errors.append(
                f"Unknown action '{decision.action}'. "
                f"Allowed actions: {', '.join(sorted(ALLOWED_ACTIONS))}."
            )

    # Every other status requires a user-facing message
    if decision.status in TERMINAL_STATUSES and not decision.user_message:
        errors.append(
            f"status is '{decision.status}' but user_message is missing or empty."
        )

    return errors
