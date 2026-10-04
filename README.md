# Agent Arena student starter

This is runnable infrastructure, NOT a completed assignment. `/arena/run` and `/chat`
return `failed` with `stop_reason: not_implemented` until you implement app/agent.py.
No model API is called by this starter. It includes no instructor keys or solution.

## Start locally on Windows
Use Python 3.12 or newer. Extract this folder first, then open a terminal inside it.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe run.py
```

Open http://127.0.0.1:8000/ for the interface or /docs for API requests.
Stop with Ctrl+C in the server terminal. If port 8000 is occupied, stop your old
server or change PORT in .env to 8001 and open that port. Do not run two servers
on the same port. On macOS/Linux use python3 and .venv/bin/python equivalents.

## What to implement
1. Choose a narrow domain and complete the design canvas below.
2. Replace data/sample_data.json with safe sandbox data.
3. Define typed decisions and state in models.py; implement at least three tools.
4. Write prompts.py and model integration. LangChain is allowed for models,
   tools and messages; add your chosen provider package to requirements.txt.
5. Implement the bounded loop, semantic validation, autonomy limits and recovery
   in agent.py. Read request.arena_config.max_steps; count repair attempts too.
6. Implement controlled fault injection at the model/tool boundary. The wrapper
   parses faults but DOES NOT simulate them for you.
7. Consume the history argument in your model messages to resolve clarification.
   Add configured model identifiers to api.py /models and validate selection.
8. Add rate/concurrency and spend controls before enabling paid calls publicly.
9. Replace the starter manifest and examples, test, deploy and submit.

## File map
- main.py: FastAPI composition and static files.
- api.py: routes, selected-model validation, session handoff.
- config.py: environment configuration; limits require enforcement in your loop.
- models.py: public contracts; add domain-specific state/decisions.
- agent.py, prompts.py, tools.py: student TODOs.
- memory.py: LangChain HumanMessage/AIMessage storage (six recent turns,
  24,000-character ceiling, 100 sessions). In-memory only; one worker. New chat
  clears server history; browser reload starts a new session. Not long-term memory.
- arena.py: timeout wrapper, response bound, safe event logging.
- static/: basic chat, selector, external note, status and observations.

## Arena contract
GET /health; GET /arena/manifest; POST /arena/run. POST /chat is for the UI.
Arena runs are independent: request_id is correlation only, not a memory key.
Chat session_id links turns; each student chooses and documents sandbox persistence.

The minimal request printed in the assignment works:
```json
{"task":"Your domain task","external_context":[],"arena_config":{"max_steps":6,"fault":"none"}}
```
The equivalent expanded fault is {"type":"none"}. Types: none, tool_timeout,
malformed_tool_output, invalid_agent_decision. Trigger the first matching operation
once per run. Supply request_id if desired; omitted IDs are generated.
External entries are {"source":"note","content":"text","trust":"untrusted"}.
Preserve the status/steps/stop_reason/tool_calls/errors response fields. Tool names
are domain-specific. Optional events and metrics support the interface; report
unknown token usage/cost as null, never as a fabricated zero.
`blocked` and `approval_required` are accepted; use and document them consistently.

## Test without spending API credits
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
The bundled tests check infrastructure only. They are NOT the hidden evaluator.
Add your own scripted-model tests. Cover ambiguity, injection, invalid decisions,
tool failure, budget termination, autonomy, and multi-turn clarification.

With the server running:
```powershell
.\.venv\Scripts\python.exe evaluation/run_public_tests.py --url http://127.0.0.1:8000
```
This initial one-case check expects the not_implemented placeholder. Replace it
with actual domain cases and expected outcomes. After model integration, this HTTP
runner can cost money. Run deliberately; do not repeatedly call paid models on CI.

## Render deployment
Create a private GitHub repository from this extracted folder (not the ZIP file).
Give the instructor access as announced in Google Classroom. In Render create a
Web Service, connect your repository, and choose the Python runtime.
- Build: pip install -r requirements.txt
- Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
- Health check: /health
Select the free instance if available; confirm the displayed plan before deploying.
render.yaml is an optional Blueprint alternative. Dockerfile is another option.
Add your chosen model name and provider keys using Render environment settings.
Never commit .env. No keys are needed to deploy the incomplete scaffold.
Free services may restart or sleep; in-memory history is lost after restart.
Verify both the browser interface and a POST through /docs on the public URL.
A successful health response proves availability, not assignment completion.
Deployment has not been performed for you by this ZIP.

References: https://render.com/docs/deploy-fastapi
and https://docs.langchain.com/oss/python/langchain/messages

## Complete your project documentation
- Problem and measurable completion condition
- Design canvas: goal, completion, system boundary, observations, actions,
  state, autonomy boundary, primary risks, evaluation criteria
- Architecture diagram and file map
- Model comparison: two models/tiers on approximately ten common inputs;
  success, contract validity, action choice, latency, tokens and estimated cost
- Prompt/context design, session isolation, reset and memory limitations
- Typed schema and semantic validation rules
- Loop, retry/time/token/step limits and stop conditions
- Tool/fault handling and test evidence
- Setup, deployment, limitations and submission links

## Google Classroom submission
Rename the outer project folder to your lowercase roll number,(mine is i230570) e.g. i221234.
Prepare i221234.zip plus i221234_submission.pdf. Include SUBMISSION.md in the ZIP;
put the same working interface URL, repository URL, endpoints and final source
commit in the PDF. Submission metadata may be prepared after the source commit;
identify the source version being evaluated clearly. Exclude .env, .venv, .git,
node_modules, caches and secrets. Follow Section 19 of the assignment and click
Turn in after attaching both files. Pushes alone are not official submissions.

---

## Agent Design Canvas

### 1. Operational Goal
Build a day-by-day sandbox travel itinerary from booking confirmations provided
in the task, external context, or conversation history. Extract structured booking
records, detect scheduling conflicts, and present an organized itinerary.

### 2. Completion Condition
The agent returns `completed` only after `build_itinerary` has successfully
produced a day-by-day itinerary incorporating all recorded bookings, with any
detected conflicts flagged in the output.

### 3. System Boundary
Sandbox only. The agent reads and structures booking text, stores records in
per-run memory, checks for time conflicts, and assembles itineraries. It never
calls external APIs, makes real bookings, processes payments, sends emails, or
modifies any system outside the run's in-memory state.

### 4. Observations
| Source | Content | Trust |
|---|---|---|
| Task text | The user's request, possibly containing booking details | Trusted |
| External context | Booking confirmations (flight emails, hotel vouchers, etc.) | **Untrusted** — isolated in a delimited message |
| Conversation history | Prior `/chat` turns with additional bookings or clarification answers | Trusted |
| Tool results | Structured output from record_bookings, check_conflicts, build_itinerary | Trusted |
| Runtime context | Today's date (for temporal awareness, not for rejecting past dates) | Trusted |

### 5. Actions / Tools
| Action | Type | Purpose |
|---|---|---|
| `record_bookings` | Tool | Store model-extracted structured bookings (Pydantic-validated) |
| `check_conflicts` | Tool | Detect time overlaps among same-category bookings |
| `build_itinerary` | Tool | Assemble day-by-day itinerary; recomputes and includes conflicts |
| `finish` | Terminal (status=completed) | Return the completed itinerary |
| `ask_clarification` | Terminal (status=needs_clarification) | Request missing information from the user |
| `request_approval` | Terminal (status=blocked) | Flag actions needing human approval |
| `refuse` | Terminal (status=blocked) | Decline requests outside sandbox scope |

### 6. State (per run)
- Extracted bookings (`list[BookingRecord]`)
- Detected conflicts (`list[dict]`)
- Built itinerary (`dict | None`)
- Step counter and shared repair flag (one repair per run)
- Observations from each tool call
- Known facts established during the run
- Tool call traces, errors, and events

### 7. Autonomy Boundary
The agent operates entirely in a sandbox:
- ✅ Extract and structure booking data from text
- ✅ Detect conflicts and build itineraries
- ✅ Ask for clarification when information is missing
- ❌ Never make real bookings, payments, or cancellations
- ❌ Never send emails or contact third parties
- ❌ Never access external APIs or databases

Any request for real-world action → `blocked` or `approval_required`.

### 8. Primary Risks
| Risk | Mitigation |
|---|---|
| Prompt injection via external_context | Structural isolation: untrusted content in a clearly delimited user message, never in the system prompt |
| Hallucinated bookings | Prompt instructs: never invent dates, times, or confirmation codes not present in the provided text |
| Infinite loops | Bounded loop with hard step limit (≤ 6); code owns stopping, not the model |
| Step budget exhaustion | Happy path uses ≤ 4 steps, leaving room for 1 repair + 1 fault |
| Timezone confusion | All datetimes normalized to UTC; naive datetimes treated as UTC (documented limitation) |
| Token truncation | Provider finish_reason checked; truncated output → budget_exceeded immediately |
| Unknown / consequential actions | One repair attempt listing allowed actions; if still unknown → contract_error |

### 9. Evaluation Criteria
- Valid `ArenaResponse` on every run (never crash)
- Correct extraction of bookings from varied text formats
- Accurate conflict detection (same-category interval overlap only)
- Sensible day-by-day itinerary with conflicts flagged
- Graceful handling of all fault types (tool_timeout, malformed_tool_output, invalid_agent_decision)
- Stays within step budget (≤ 6 steps)
- Resists prompt injection attempts
- Asks for clarification when critical information is missing
- Multi-turn resolution via chat history

---

## Architecture Diagram
<!-- TODO: Add architecture diagram showing request flow through api → arena → agent → tools -->

## Folder Map
<!-- TODO: Add annotated folder/file tree -->

## Model Comparison
<!-- TODO: Compare two models/tiers on ~10 common inputs: success rate, contract validity, action choice, latency, tokens, estimated cost -->

## Prompt and Context Design
<!-- TODO: Document system prompt structure, message assembly order, external context isolation strategy -->

## Memory Policy
<!-- TODO: Document session isolation, reset semantics, /chat vs /arena/run, memory size limits, restart behaviour -->

## Structured-Output Schema
<!-- TODO: Document AgentDecision fields, BookingRecord schema, tool I/O models -->

## Validation Rules
<!-- TODO: Document schema validation (Pydantic), semantic validation (separate functions), repair policy, gateway retry rules -->

## Stopping Conditions
<!-- TODO: Document all loop termination conditions: completed, needs_clarification, blocked, contract_error, budget_exceeded, finish guard -->

## Limitations
- **Timezone handling**: Naive datetimes (without timezone offset) are treated as
  UTC. Bookings in different cities without explicit timezone offsets will be
  compared as if in the same timezone. Users should include timezone offsets
  (e.g. `+05:00`, `-04:00`) in booking times for accurate cross-timezone
  conflict detection.
- **In-memory state**: All session data and per-run state are in RAM and lost on
  server restart.
- **Single worker**: The server runs one Uvicorn worker; not horizontally
  scalable.
- **Token budget**: `max_output_tokens` defaults to 512. Long itineraries or
  detailed reasoning may be truncated by the provider.

## Local Setup
<!-- TODO: Document any additional local setup steps beyond the starter README instructions -->

## Deployment
<!-- TODO: Document Render deployment configuration and verification steps -->
