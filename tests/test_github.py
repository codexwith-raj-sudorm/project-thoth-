import base64
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from thoth.github import GitHubService
from thoth.schemas import GistRequest, PushRequest
from thoth.server import create_app


def github_transport(request: httpx.Request) -> httpx.Response:
    headers = {"X-RateLimit-Remaining": "8", "X-OAuth-Scopes": "gist, repo"}
    assert request.headers["Authorization"] == "Bearer test-token"
    if request.url.path == "/user":
        return httpx.Response(200, json={"login": "thoth-user"}, headers=headers)
    if request.url.path == "/gists":
        payload = json.loads(request.content)
        assert payload["files"]["script.py"]["content"] == "print('hello')"
        return httpx.Response(
            201,
            json={"id": "g1", "url": "https://api.github.com/gists/g1", "html_url": "https://gist.github.com/g1"},
            headers=headers,
        )
    if request.url.path == "/repos/acme/tools/contents/scripts/result.py" and request.method == "GET":
        assert request.url.params["ref"] == "main"
        return httpx.Response(200, json={"sha": "old-sha"}, headers=headers)
    if request.url.path == "/repos/acme/tools/contents/scripts/result.py" and request.method == "PUT":
        payload = json.loads(request.content)
        assert payload["sha"] == "old-sha"
        assert base64.b64decode(payload["content"]).decode() == "print('hello')"
        return httpx.Response(
            200,
            json={"commit": {"sha": "commit-sha"}, "content": {"html_url": "https://github.com/acme/tools/blob/main/scripts/result.py"}},
            headers=headers,
        )
    return httpx.Response(404, json={"message": "Not Found"}, headers=headers)


def test_github_service_status_gist_and_push():
    async def exercise():
        transport = httpx.MockTransport(github_transport)
        service = GitHubService("test-token", transport=transport)
        status = await service.status()
        assert status == {
            "authenticated": True,
            "username": "thoth-user",
            "scopes": ["gist", "repo"],
            "rate_limit_remaining": 8,
            "rate_limit_warning": True,
        }
        gist = await service.create_gist(
            GistRequest(code="print('hello')", filename="script.py", is_public=False)
        )
        assert gist["html_url"] == "https://gist.github.com/g1"
        assert gist["rate_limit_warning"] is True
        pushed = await service.push_file(
            PushRequest(
                code="print('hello')",
                filename="scripts/result.py",
                repo="acme/tools",
                branch="main",
                commit_message="Add result",
            )
        )
        assert pushed["commit_sha"] == "commit-sha"

    import asyncio

    asyncio.run(exercise())


def test_github_api_and_missing_token(tmp_path: Path, monkeypatch):
    (tmp_path / "assets").mkdir()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    app = create_app(static_root=tmp_path, github_transport=httpx.MockTransport(github_transport))
    with TestClient(app) as client:
        assert client.get("/api/github/status").json()["authenticated"] is False
        response = client.post(
            "/api/github/gist",
            json={"code": "pass", "filename": "x.py", "description": "", "is_public": False},
        )
        assert response.status_code == 401


def test_github_api_authenticated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    app = create_app(static_root=tmp_path, github_transport=httpx.MockTransport(github_transport))
    with TestClient(app) as client:
        assert client.get("/api/github/status").json()["username"] == "thoth-user"
        response = client.post(
            "/api/github/gist",
            json={
                "code": "print('hello')",
                "filename": "script.py",
                "description": "test",
                "is_public": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["id"] == "g1"


def test_github_paths_reject_traversal():
    import pytest

    with pytest.raises(ValueError):
        PushRequest(code="x", filename="../secret.py", repo="a/b")
    with pytest.raises(ValueError):
        GistRequest(code="x", filename="nested/file.py")
