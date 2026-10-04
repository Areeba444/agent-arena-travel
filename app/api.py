"""API endpoints for Arena runner, chat UI, and model management."""
import asyncio
from collections import defaultdict
import json
import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import ROOT, settings
from app.models import ArenaRequest, ArenaResponse, ChatRequest
from app.arena import execute

router = APIRouter()

# ─── Rate Limiting & Concurrency Management ─────────────────────────

ip_request_history: dict[str, list[float]] = defaultdict(list)
concurrency_semaphore = asyncio.Semaphore(settings.max_concurrency)


def get_client_ip(request: Request) -> str:
    """Extracts client IP, respecting X-Forwarded-For when behind reverse proxies like Render."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(client_ip: str) -> None:
    """Enforces per-IP sliding window rate limit."""
    now = time.time()
    cutoff = now - 60.0
    history = ip_request_history[client_ip]
    # Prune timestamps older than 60 seconds
    ip_request_history[client_ip] = [t for t in history if t > cutoff]
    if len(ip_request_history[client_ip]) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded (maximum {settings.rate_limit_per_minute} requests per minute). Please retry shortly."
        )
    ip_request_history[client_ip].append(now)


class ConcurrencyGuard:
    """Async context manager for global concurrency capping; waits up to 15s for an available slot before 429."""
    def __init__(self, sem: asyncio.Semaphore, timeout: float = 15.0):
        self.sem = sem
        self.timeout = timeout
        self.acquired = False

    async def __aenter__(self):
        try:
            await asyncio.wait_for(self.sem.acquire(), timeout=self.timeout)
            self.acquired = True
            return self
        except (asyncio.TimeoutError, TimeoutError):
            raise HTTPException(
                status_code=429,
                detail="Global concurrency limit exceeded (maximum concurrent runs active). Please retry shortly."
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.sem.release()


# ─── Endpoints ──────────────────────────────────────────────────────

@router.get('/')
def index():
    return FileResponse(ROOT / 'app/static/index.html')


@router.get('/health')
def health():
    return {'status': 'ok', 'implementation': 'student'}


@router.get('/arena/manifest')
def manifest():
    return json.loads((ROOT / 'arena_manifest.json').read_text())


@router.get('/models')
def models():
    """Exposes only configured, allowed models; never exposes keys."""
    available = settings.allowed_models_list
    if not available:
        available = ['unconfigured']
    return {'models': available}


@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest, request: Request):
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    async with ConcurrencyGuard(concurrency_semaphore):
        return await execute(payload, model=settings.model_name or 'unconfigured')


@router.post('/chat', response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    allowed_list = models()['models']
    selected_model = payload.model
    if selected_model == 'unconfigured' and settings.model_name:
        selected_model = settings.model_name
    elif selected_model not in allowed_list and selected_model != 'unconfigured':
        raise HTTPException(400, 'Model is not enabled')

    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory = request.app.state.memory

    try:
        async with ConcurrencyGuard(concurrency_semaphore):
            result = await execute(payload, memory.get(payload.session_id), selected_model)
            memory.add(payload.session_id, payload.task, result.final_response)
            return result
    finally:
        busy.discard(payload.session_id)


@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy:
        raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    return {'status': 'cleared'}
