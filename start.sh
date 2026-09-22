#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python >/dev/null 2>&1; then
  echo "Python is missing. In Termux run: pkg install python" >&2
  exit 1
fi

if [ -z "${GROQ_API_KEY:-}" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "[Notice] No cloud API key found. Set GROQ_API_KEY or GEMINI_API_KEY in ~/.profile." >&2
  echo "         Ollama remains available if OLLAMA_BASE_URL is configured." >&2
fi

python - <<'PY'
try:
    import fastapi, uvicorn, pydantic
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc.name}. Run: python -m pip install -e .")
PY

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
fi

echo "Project Thoth Mobile: http://localhost:8000"
exec python server.py
