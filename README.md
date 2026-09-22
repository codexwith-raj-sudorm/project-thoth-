# Project Thoth Mobile

Project Thoth Mobile is a self-hosted, mobile-first PWA for deterministic dual-agent Python synthesis and verification. The FastAPI server, security scanner, subprocess runner, session logs, and interface all run on your Android phone in Termux. Only model inference leaves the device.

The original `thoth` CLI remains available for shell scripts and desktop use.

> **Security notice:** generated code runs inside a temporary directory with a stripped environment after AST screening. On Android, Termux also benefits from Android application isolation. This remains a best-effort safeguard for trusted personal prompts—not a containment system for hostile code. Network access cannot be isolated without root; known network modules are blocked unless explicitly enabled.

## Features

- Coder and Verifier agents coordinated by a deterministic Python mediator
- Live Server-Sent Event progress for every generation, execution, and verification stage
- Strict Pydantic model-output validation with one automatic repair attempt
- Tiered AST security checks and tautological-test detection
- Temporary subprocess execution with a five-second timeout and no API keys in its environment
- Groq and Gemini free-tier adapters, plus optional Ollama support
- Arena mode runs 2–3 providers concurrently and uses a Verifier-as-Judge ranking
- One-tap private Gists and GitHub Contents API repository pushes
- Mobile-first dark interface with copy, native share, and Python download actions
- Local session history, installable manifest, and offline PWA shell
- Single-run queue, process lock, free-memory guard, token budget, and rate limiting

## Termux installation

Install Termux from F-Droid rather than the outdated Play Store package. In a fresh Termux session:

```bash
pkg update
pkg install python git
termux-setup-storage  # optional; Thoth itself does not require shared storage

git clone <your-project-thoth-repository-url>
cd project-thoth-
python -m pip install -e .
```

Configure at least one cloud provider in `~/.profile` or `~/.bashrc`:

```bash
export GROQ_API_KEY="your-groq-key"
export GEMINI_API_KEY="your-gemini-key"
# Optional: enables Gist and repository actions in the result screen.
export GITHUB_TOKEN="your-fine-grained-github-token"
```

Reload the profile without placing secrets in command history again:

```bash
source ~/.profile
```

## Start the mobile app

```bash
./start.sh
```

Open **http://localhost:8000** in Chrome or Firefox on the same phone. In Chrome, use **Add to Home screen** to install the standalone PWA.

`start.sh` checks Python dependencies, warns if no cloud key is configured, requests a Termux wake lock when Termux:API is available, and launches the local-only Uvicorn server. Stop it with Ctrl+C.

Android may terminate background processes under memory or battery pressure. Disable battery optimization for Termux or lock it in the recent-apps screen during long runs. The server refuses new work when `/proc/meminfo` reports less than 200MB available.

## Arena mode

Select **Arena** on the New Task screen to run all available providers concurrently. The default roster is built from configured Groq and Gemini keys; Ollama is added only when its local `/api/tags` endpoint responds. Each candidate is AST-scanned and executed in an independent temporary subprocess. One available provider then judges every completed candidate in a single structured pass.

Arena uses a 40,000-token session ceiling. A failed or offline candidate is shown in the results but does not abort the other candidates. If only one candidate survives, Thoth automatically runs the normal Coder/Verifier workflow rather than presenting a meaningless one-entry ranking.

## GitHub integration

Create a fine-grained GitHub token with **Gists: read/write** for Gist saving and **Contents: read/write** for repository pushes. Keep it only in the Termux environment as `GITHUB_TOKEN`; it is never sent to generated code or stored in browser history.

The PWA checks the token through `/api/github/status` and hides GitHub actions when authentication is unavailable. GitHub rate-limit headers are returned with each action, and the UI warns when fewer than ten requests remain. Gists are private by default.

## API

Health check:

```bash
curl http://localhost:8000/api/health
```

Stream a Groq run:

```bash
curl -N -X POST http://localhost:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a typed function that merges overlapping intervals",
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "allow_filesystem": false,
    "allow_network": false,
    "max_turns": 3
  }'
```

Run an Arena battle using the automatically discovered roster:

```bash
curl -N -X POST http://localhost:8000/api/arena \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Implement a typed TTL cache with deterministic tests"}'
```

Check GitHub authentication or create a private Gist:

```bash
curl http://localhost:8000/api/github/status
curl -X POST http://localhost:8000/api/github/gist \
  -H 'Content-Type: application/json' \
  -d '{"code":"print(42)","filename":"answer.py","description":"Thoth result","is_public":false}'
```

Read the 20 most recent JSONL-backed sessions:

```bash
curl http://localhost:8000/api/sessions
```

Interactive API documentation is available at `http://localhost:8000/docs` while the server is running.

## CLI

```bash
thoth run "Write a typed function that merges overlapping intervals" \
  --provider groq --model llama-3.1-8b-instant -o intervals.py
```

For Gemini, pass `--provider gemini --model gemini-2.5-flash`. For a local Ollama process, pass `--provider ollama --model qwen2.5-coder:7b`.

Permission flags are intentionally explicit:

```bash
thoth run "Parse records from input.csv" --provider groq \
  --model llama-3.1-8b-instant --allow-filesystem
```

Tier 0 facilities such as `eval`, `exec`, `pickle`, `ctypes`, and dynamic imports cannot be enabled by a flag.

## Development and testing

```bash
python -m pip install -e '.[test]'
pytest
```

The API tests use deterministic fake adapters and never consume cloud quota. Real-provider smoke testing requires the corresponding environment key.

Run a short benchmark smoke test or the complete ThothBench-30 suite with:

```bash
python benchmarks/run_bench.py --provider groq --model llama-3.1-8b-instant --limit 3
python benchmarks/run_bench.py --provider groq --model llama-3.1-8b-instant
```

Results are written to the ignored `benchmarks/results.json` file and include first-pass approval and two-turn convergence rates. Do not publish benchmark claims until a complete real-provider run has finished.

## Project layout

```text
server.py                 Termux/Uvicorn launcher
index.html                no-build mobile UI
manifest.json             PWA metadata
sw.js                     offline shell worker
assets/                    icons and local highlighting fallback
src/thoth/server.py       FastAPI application and SSE bridge
src/thoth/orchestrator.py deterministic standard engine
src/thoth/arena.py        parallel synthesis and judge workflow
src/thoth/github.py       direct GitHub REST integration
src/thoth/security.py     AST policy
src/thoth/adapters/       model providers
start.sh                  one-command Termux startup
tests/                    engine and HTTP tests
```

Session event logs are appended under `~/.thoth/logs/`. At least the newest 50 sessions or seven days of logs are retained. The browser separately retains the 20 latest completed results in local storage for immediate offline access.
