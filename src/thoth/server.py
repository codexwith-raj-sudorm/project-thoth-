"""FastAPI presentation layer for Project Thoth Mobile."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .adapters import GeminiAdapter, ModelAdapter, OllamaAdapter, OpenAICompatibleAdapter
from .arena import default_roster, run_arena
from .exceptions import MaxTurnsExceeded, StagnantLoopException, ThothError
from .github import GitHubService
from .locking import SingleInstanceLock
from .orchestrator import Orchestrator
from .schemas import ArenaRequest, GistRequest, PushRequest, RunRequest, SSEEvent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = Path.home() / ".thoth" / "logs"
AdapterFactory = Callable[[str, str], tuple[ModelAdapter, ModelAdapter]]


def available_memory_mb() -> float | None:
    """Return Linux/Termux MemAvailable, or None on unsupported systems."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def default_adapter_factory(provider: str, model: str) -> tuple[ModelAdapter, ModelAdapter]:
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY is not configured in the Termux environment")
        make = lambda: OpenAICompatibleAdapter(  # noqa: E731
            model, "https://api.groq.com/openai/v1", key, requests_per_minute=12
        )
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured in the Termux environment")
        make = lambda: GeminiAdapter(key, model)  # noqa: E731
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        make = lambda: OllamaAdapter(model, base_url)  # noqa: E731
    return make(), make()


def _sse(event: SSEEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def read_sessions(log_directory: Path, limit: int = 20) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    files = sorted(
        log_directory.glob("session_*.jsonl") if log_directory.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in files[:limit]:
        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
        if not records:
            continue
        prompt = next((str(r.get("prompt", "")) for r in records if r.get("prompt")), "")
        verdicts = [r for r in records if r.get("verifier_status")]
        last = verdicts[-1] if verdicts else records[-1]
        sessions.append(
            {
                "session_id": records[0].get("session_id", path.stem.removeprefix("session_")),
                "timestamp": records[0].get("timestamp"),
                "prompt": prompt[:80],
                "status": last.get("verifier_status") or (
                    "BLOCKED" if last.get("ast_scan_result") != "pass" else "INCOMPLETE"
                ),
                "turns_used": max((int(r.get("turn_index", 0)) for r in records), default=0),
            }
        )
    return sessions


def create_app(
    *,
    adapter_factory: AdapterFactory = default_adapter_factory,
    log_directory: Path | None = None,
    static_root: Path | None = None,
    lock_path: Path | None = None,
    memory_reader: Callable[[], float | None] = available_memory_mb,
    github_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    logs = log_directory or DEFAULT_LOG_DIR
    web = static_root or PROJECT_ROOT
    started_at = time.monotonic()
    run_gate = asyncio.Lock()
    active_tasks: set[asyncio.Task[Any]] = set()
    state = {"provider": "groq", "model": "llama-3.1-8b-instant"}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    app = FastAPI(title="Project Thoth Mobile", version="4.0.0", lifespan=lifespan)

    assets = web / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(web / "index.html")

    @app.get("/manifest.json", include_in_schema=False)
    async def manifest() -> FileResponse:
        return FileResponse(web / "manifest.json", media_type="application/manifest+json")

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker() -> FileResponse:
        return FileResponse(web / "sw.js", media_type="application/javascript")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": state["provider"],
            "model": state["model"],
            "uptime_seconds": round(time.monotonic() - started_at),
            "run_in_progress": run_gate.locked(),
            "available_memory_mb": memory_reader(),
        }

    @app.get("/api/sessions")
    async def sessions() -> list[dict[str, Any]]:
        return await asyncio.to_thread(read_sessions, logs, 20)

    def github() -> GitHubService:
        return GitHubService(os.getenv("GITHUB_TOKEN", ""), transport=github_transport)

    @app.get("/api/github/status")
    async def github_status() -> dict[str, Any]:
        try:
            return await github().status()
        except httpx.HTTPError as exc:
            return {
                "authenticated": False,
                "username": "",
                "scopes": [],
                "error": f"GitHub status check failed: {exc}",
            }

    @app.post("/api/github/gist")
    async def create_gist(request: GistRequest) -> dict[str, Any]:
        try:
            return await github().create_gist(request)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            message = exc.response.json().get("message", str(exc)) if exc.response.content else str(exc)
            raise HTTPException(status_code=exc.response.status_code, detail=f"GitHub: {message}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc

    @app.post("/api/github/push")
    async def push_to_github(request: PushRequest) -> dict[str, Any]:
        try:
            return await github().push_file(request)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            message = exc.response.json().get("message", str(exc)) if exc.response.content else str(exc)
            raise HTTPException(status_code=exc.response.status_code, detail=f"GitHub: {message}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}") from exc

    @app.get("/api/arena/roster")
    async def arena_roster() -> dict[str, Any]:
        roster = await default_roster()
        return {"candidates": [item.model_dump() for item in roster]}

    @app.post("/api/run")
    async def run(request: RunRequest) -> StreamingResponse:
        free_mb = memory_reader()
        if free_mb is not None and free_mb < 200:
            raise HTTPException(
                status_code=503,
                detail=f"Only {free_mb:.0f}MB memory available; 200MB is required to start a run.",
            )
        queued = run_gate.locked()
        state.update(provider=request.provider, model=request.model)

        async def stream():
            queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            latest: dict[str, Any] = {"code": "", "critique": "", "turn": 0}

            yield _sse(
                SSEEvent(
                    type="start",
                    data={
                        "prompt": request.prompt,
                        "provider": request.provider,
                        "model": request.model,
                        "queued": queued,
                    },
                )
            )

            def observe(event_type: str, turn_index: int | None, data: dict[str, Any]) -> None:
                if turn_index:
                    latest["turn"] = turn_index
                if event_type == "verifier_done":
                    latest["critique"] = data.get("critique_summary", "")
                event = SSEEvent(type=event_type, turn_index=turn_index, data=data)  # type: ignore[arg-type]
                loop.call_soon_threadsafe(queue.put_nowait, event)

            def execute() -> None:
                try:
                    coder, verifier = adapter_factory(request.provider, request.model)
                    orchestrator = Orchestrator(
                        coder,
                        verifier,
                        max_turns=request.max_turns,
                        allow_filesystem=request.allow_filesystem,
                        allow_network=request.allow_network,
                        log_directory=logs,
                        event_sink=observe,
                    )
                    with SingleInstanceLock(lock_path):
                        result = orchestrator.run(request.prompt)
                    latest["code"] = result.code
                    observe(
                        "complete",
                        result.turns,
                        {
                            "status": "APPROVED",
                            "code": result.code,
                            "turns_used": result.turns,
                            "total_tokens": result.total_tokens,
                            "critique": " ".join(
                                filter(
                                    None,
                                    (
                                        result.verifier.implementation_critique,
                                        result.verifier.test_critique,
                                    ),
                                )
                            ),
                        },
                    )
                except (MaxTurnsExceeded, StagnantLoopException) as exc:
                    latest["code"] = exc.partial_code
                    observe(
                        "complete",
                        latest["turn"] or None,
                        {
                            "status": type(exc).__name__,
                            "code": exc.partial_code,
                            "turns_used": latest["turn"],
                            "total_tokens": 0,
                            "critique": latest["critique"] or str(exc),
                        },
                    )
                except Exception as exc:
                    observe(
                        "error",
                        latest["turn"] or None,
                        {"error_type": type(exc).__name__, "message": str(exc)},
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            async with run_gate:
                worker = asyncio.create_task(asyncio.to_thread(execute))
                active_tasks.add(worker)
                worker.add_done_callback(active_tasks.discard)
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield _sse(event)
                await worker

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/arena")
    async def arena(request: ArenaRequest) -> StreamingResponse:
        free_mb = memory_reader()
        if free_mb is not None and free_mb < 200:
            raise HTTPException(
                status_code=503,
                detail=f"Only {free_mb:.0f}MB memory available; 200MB is required to start a run.",
            )
        roster = request.candidates or await default_roster()
        queued = run_gate.locked()

        async def stream():
            queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            yield _sse(
                SSEEvent(
                    type="start",
                    data={
                        "prompt": request.prompt,
                        "mode": "arena",
                        "queued": queued,
                        "candidates": [item.model_dump() for item in roster],
                    },
                )
            )

            def observe(event_type: str, turn_index: int | None, data: dict[str, Any]) -> None:
                event = SSEEvent(type=event_type, turn_index=turn_index, data=data)  # type: ignore[arg-type]
                loop.call_soon_threadsafe(queue.put_nowait, event)

            async def execute_arena() -> None:
                try:
                    with SingleInstanceLock(lock_path):
                        result = await run_arena(
                            request.prompt,
                            roster,
                            adapter_factory,
                            allow_filesystem=request.allow_filesystem,
                            allow_network=request.allow_network,
                            token_budget=40_000,
                            event_sink=observe,
                            log_directory=logs,
                        )
                    observe("arena_complete", None, result.public_dict())
                except Exception as exc:
                    observe(
                        "error",
                        None,
                        {"error_type": type(exc).__name__, "message": str(exc)},
                    )
                finally:
                    # Use the same thread-safe scheduler as event delivery so the
                    # sentinel cannot overtake judge_done/arena_complete.
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            async with run_gate:
                worker = asyncio.create_task(execute_arena())
                active_tasks.add(worker)
                worker.add_done_callback(active_tasks.discard)
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield _sse(event)
                await worker

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
