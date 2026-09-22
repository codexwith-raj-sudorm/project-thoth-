import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel

from thoth.adapters.base import ModelAdapter
from thoth.schemas import (
    ArenaVerifierOutput,
    CandidateScore,
    CoderOutput,
    Usage,
    VerifierOutput,
)
from thoth.server import create_app, read_sessions


class QueueAdapter(ModelAdapter):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.last_usage = Usage(prompt_tokens=3, completion_tokens=2)

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        raise NotImplementedError

    def generate_structured(self, system_prompt, user_prompt, schema):
        return next(self.outputs)


def factory(provider: str, model: str):
    assert provider == "groq"
    assert model == "test-model"
    return (
        QueueAdapter([
            CoderOutput(
                thought_process="Use a direct deterministic implementation.",
                implementation_code="def add(a: int, b: int) -> int:\n    return a + b\n\nassert add(2, 3) == 5",
            )
        ]),
        QueueAdapter([
            VerifierOutput(
                status="APPROVED",
                implementation_critique="Correct and typed.",
                test_critique="The assertion exercises behavior.",
            )
        ]),
    )


def test_health_and_static_shell(tmp_path: Path):
    (tmp_path / "index.html").write_text("<h1>Thoth</h1>")
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "sw.js").write_text("")
    app = create_app(static_root=tmp_path, log_directory=tmp_path / "logs", memory_reader=lambda: 512)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert client.get("/").text == "<h1>Thoth</h1>"


def test_request_validation_and_memory_guard(tmp_path: Path):
    app = create_app(static_root=tmp_path, memory_reader=lambda: 100)
    with TestClient(app) as client:
        too_short = client.post("/api/run", json={"prompt": "short"})
        assert too_short.status_code == 422
        refused = client.post("/api/run", json={"prompt": "A sufficiently detailed task"})
        assert refused.status_code == 503
        assert "200MB" in refused.json()["detail"]


def test_run_streams_all_major_events_and_writes_history(tmp_path: Path):
    logs = tmp_path / "logs"
    app = create_app(
        adapter_factory=factory,
        static_root=tmp_path,
        log_directory=logs,
        lock_path=tmp_path / "thoth.lock",
        memory_reader=lambda: 512,
    )
    payload = {"prompt": "Write a typed integer addition function", "provider": "groq", "model": "test-model"}
    with TestClient(app) as client:
        response = client.post("/api/run", json=payload)
        assert response.status_code == 200
        events = []
        for line in response.text.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        types = [event["type"] for event in events]
        assert types == [
            "start", "turn_start", "coder_start", "coder_done", "execution_start",
            "execution_done", "verifier_start", "verifier_done", "complete",
        ]
        complete = events[-1]["data"]
        assert complete["status"] == "APPROVED"
        assert "def add" in complete["code"]
        assert client.get("/api/sessions").json()[0]["prompt"].startswith("Write a typed")


def test_read_sessions_ignores_malformed_files(tmp_path: Path):
    (tmp_path / "session_bad.jsonl").write_text("not json")
    good = tmp_path / "session_good.jsonl"
    good.write_text(json.dumps({
        "session_id": "good", "timestamp": "2026-01-01T00:00:00Z", "turn_index": 0,
        "prompt": "x" * 100, "ast_scan_result": "pass", "verifier_status": None,
    }) + "\n" + json.dumps({
        "session_id": "good", "timestamp": "2026-01-01T00:00:01Z", "turn_index": 2,
        "ast_scan_result": "pass", "verifier_status": "APPROVED",
    }))
    sessions = read_sessions(tmp_path)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "good"
    assert len(sessions[0]["prompt"]) == 80
    assert sessions[0]["turns_used"] == 2


class ArenaApiAdapter(ModelAdapter):
    def __init__(self, provider):
        self.provider = provider
        self.last_usage = Usage(prompt_tokens=1, completion_tokens=1)

    def _generate_text(self, system_prompt, user_prompt, schema):
        raise NotImplementedError

    def generate_structured(self, system_prompt, user_prompt, schema):
        if schema is CoderOutput:
            return CoderOutput(
                thought_process=self.provider,
                implementation_code="def answer() -> int:\n    return 42\n\nassert answer() == 42",
            )
        return ArenaVerifierOutput(
            ranking=["A", "B"],
            scores=[
                CandidateScore(candidate_id="A", correctness=3, completeness=3, code_quality=3, total=9, notes="best"),
                CandidateScore(candidate_id="B", correctness=3, completeness=2, code_quality=2, total=7, notes="good"),
            ],
            winner_rationale="A is more complete.",
        )


def test_arena_sse_endpoint(tmp_path: Path):
    app = create_app(
        adapter_factory=lambda provider, model: (ArenaApiAdapter(provider), ArenaApiAdapter("judge")),
        static_root=tmp_path,
        log_directory=tmp_path / "logs",
        lock_path=tmp_path / "arena.lock",
        memory_reader=lambda: 512,
    )
    payload = {
        "prompt": "Build a typed answer function for this test",
        "candidates": [
            {"candidate_id": "A", "provider": "groq", "model": "a"},
            {"candidate_id": "B", "provider": "gemini", "model": "b"},
        ],
    }
    with TestClient(app) as client:
        response = client.post("/api/arena", json=payload)
        assert response.status_code == 200
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
    types = [event["type"] for event in events]
    assert types[0] == "start"
    assert types.count("candidate_done") == 2
    assert "judge_done" in types
    assert types[-1] == "arena_complete"
    assert events[-1]["data"]["winner"] == "A"
