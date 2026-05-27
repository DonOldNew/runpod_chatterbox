"""
AI Phone Call Engine for Global Advance HR.

Orchestrates: Groq Whisper (STT) + DeepSeek V4 Pro (LLM) + Chatterbox Turbo (TTS)
Designed to be called from n8n via HTTP or subprocess.

Usage:
  # As HTTP server (for n8n HTTP Request node):
  python call_engine.py serve --port 5050

  # Single turn (for n8n Execute Command node):
  python call_engine.py turn --audio input.wav --session-id abc123

  # Text input (skip STT):
  python call_engine.py turn --text "Hola, necesito ayuda" --session-id abc123
"""
import os
import sys
import json
import time
import base64
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
CHATTERBOX_URL = "https://api.runpod.ai/v2/chatterbox-turbo"

# Voice cloning reference (set to URL of target person's voice sample)
VOICE_CLONE_URL = os.getenv("VOICE_CLONE_URL", "")

# Session storage directory
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "/tmp/call_sessions"))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Audio output directory
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/tmp/call_audio"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# System prompt — Carolina, HR assistant persona
SYSTEM_PROMPT = """Eres Carolina, la asistente virtual de Recursos Humanos de Global Advance.
Hablas en espanol colombiano natural y calido. Eres amable, profesional y directa. Tuteas a los empleados.

Reglas:
- Responde en 1-3 frases cortas, como en una llamada telefonica real.
- No uses emojis, asteriscos, ni formato especial — solo texto hablado natural.
- Si no tienes la informacion, di que vas a revisar y que le informaras pronto.
- Usa expresiones colombianas naturales: "con gusto", "claro que si", "dale", "listo".
- Nunca digas que eres una IA o un bot. Eres Carolina de Recursos Humanos.
- Si preguntan por datos especificos (salario, vacaciones, etc.), di que revisas en el sistema.
"""


class CallSession:
    """Manages conversation state for a single phone call."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        self.created_at = datetime.now().isoformat()
        self.turn_count = 0
        self.file_path = SESSIONS_DIR / f"{session_id}.json"
        self._load()

    def _load(self):
        if self.file_path.exists():
            data = json.loads(self.file_path.read_text())
            self.history = data.get("history", [])
            self.created_at = data.get("created_at", self.created_at)
            self.turn_count = data.get("turn_count", 0)

    def save(self):
        data = {
            "session_id": self.session_id,
            "history": self.history,
            "created_at": self.created_at,
            "turn_count": self.turn_count,
            "updated_at": datetime.now().isoformat(),
        }
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def add_turn(self, user_text: str, assistant_text: str):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        self.turn_count += 1
        # Keep last 20 turns (40 messages) to avoid token overflow
        if len(self.history) > 40:
            self.history = self.history[-40:]
        self.save()


def stt(audio_path: str, language: str = "es") -> dict:
    """Transcribe audio using Groq Whisper."""
    start = time.time()

    with open(audio_path, "rb") as f:
        resp = requests.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={
                "model": "whisper-large-v3-turbo",
                "language": language,
                "response_format": "verbose_json",
            },
            timeout=30,
        )

    elapsed = time.time() - start

    if resp.status_code != 200:
        return {"error": f"STT failed: {resp.status_code} {resp.text}", "time": elapsed}

    result = resp.json()
    return {
        "text": result.get("text", "").strip(),
        "duration": result.get("duration", 0),
        "time": elapsed,
    }


def llm(user_text: str, session: CallSession) -> dict:
    """Generate response using DeepSeek V4 Pro."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session.history)
    messages.append({"role": "user", "content": user_text})

    start = time.time()

    resp = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-v4-pro",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7,
            "thinking": {"type": "disabled"},
        },
        timeout=30,
    )

    elapsed = time.time() - start

    if resp.status_code != 200:
        return {"error": f"LLM failed: {resp.status_code} {resp.text}", "time": elapsed}

    result = resp.json()
    reply = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})

    return {
        "text": reply,
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "time": elapsed,
    }


def tts(text: str) -> dict:
    """Generate speech using Chatterbox Turbo."""
    payload = {"input": {"prompt": text, "format": "wav"}}

    if VOICE_CLONE_URL:
        payload["input"]["voice_url"] = VOICE_CLONE_URL
    else:
        payload["input"]["voice"] = "lucy"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    start = time.time()

    # Submit job
    resp = requests.post(f"{CHATTERBOX_URL}/run", json=payload, headers=headers, timeout=30)
    data = resp.json()

    if "id" not in data:
        return {"error": f"TTS submit failed: {json.dumps(data)}", "time": time.time() - start}

    job_id = data["id"]

    # Poll for result (max 120s)
    deadline = time.time() + 120
    while time.time() < deadline:
        status_resp = requests.get(f"{CHATTERBOX_URL}/status/{job_id}", headers=headers, timeout=10)
        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            audio_path = str(AUDIO_DIR / f"{job_id}.wav")

            if isinstance(output, dict) and "audio_url" in output:
                audio_resp = requests.get(output["audio_url"], timeout=30)
                with open(audio_path, "wb") as f:
                    f.write(audio_resp.content)
                return {
                    "audio_path": audio_path,
                    "audio_url": output["audio_url"],
                    "time": time.time() - start,
                }
            elif isinstance(output, dict) and "audio_base64" in output:
                audio_bytes = base64.b64decode(output["audio_base64"])
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                return {
                    "audio_path": audio_path,
                    "audio_base64_size": len(output["audio_base64"]),
                    "time": time.time() - start,
                }

            return {"error": f"Unexpected output format: {json.dumps(output)[:200]}", "time": time.time() - start}

        elif status == "FAILED":
            return {"error": f"TTS failed: {json.dumps(status_data)[:200]}", "time": time.time() - start}

        time.sleep(1)

    return {"error": "TTS timeout (120s)", "time": time.time() - start}


def process_turn(
    session_id: str,
    audio_path: str = None,
    audio_base64: str = None,
    text: str = None,
    language: str = "es",
) -> dict:
    """
    Process one conversation turn.

    Input: audio file/base64 OR text
    Output: {
        "user_text": str,
        "assistant_text": str,
        "audio_url": str,
        "audio_path": str,
        "timings": {"stt": float, "llm": float, "tts": float, "total": float},
        "turn_number": int,
    }
    """
    total_start = time.time()
    session = CallSession(session_id)

    # Step 1: STT
    if audio_base64:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(base64.b64decode(audio_base64))
        tmp.close()
        audio_path = tmp.name

    if audio_path:
        stt_result = stt(audio_path, language)
        if "error" in stt_result:
            return {"error": stt_result["error"]}
        user_text = stt_result["text"]
        stt_time = stt_result["time"]
        # Clean up temp file
        if audio_base64:
            os.unlink(audio_path)
    elif text:
        user_text = text
        stt_time = 0
    else:
        return {"error": "No input provided (need audio_path, audio_base64, or text)"}

    if not user_text:
        return {"error": "No speech detected in audio"}

    # Step 2: LLM
    llm_result = llm(user_text, session)
    if "error" in llm_result:
        return {"error": llm_result["error"]}

    assistant_text = llm_result["text"]

    # Step 3: TTS
    tts_result = tts(assistant_text)
    if "error" in tts_result:
        return {"error": tts_result["error"]}

    # Save conversation turn
    session.add_turn(user_text, assistant_text)

    total_time = time.time() - total_start

    return {
        "session_id": session_id,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_url": tts_result.get("audio_url", ""),
        "audio_path": tts_result.get("audio_path", ""),
        "turn_number": session.turn_count,
        "timings": {
            "stt": round(stt_time, 3),
            "llm": round(llm_result["time"], 3),
            "tts": round(tts_result["time"], 3),
            "total": round(total_time, 3),
        },
        "tokens": {
            "in": llm_result.get("tokens_in", 0),
            "out": llm_result.get("tokens_out", 0),
        },
    }


def serve(port: int = 5050):
    """Start HTTP server for n8n integration."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class CallHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._respond(400, {"error": "Invalid JSON"})
                return

            if self.path == "/turn":
                result = process_turn(
                    session_id=data.get("session_id", hashlib.md5(str(time.time()).encode()).hexdigest()[:12]),
                    audio_base64=data.get("audio_base64"),
                    text=data.get("text"),
                    language=data.get("language", "es"),
                )
                self._respond(200, result)

            elif self.path == "/health":
                self._respond(200, {
                    "status": "ok",
                    "components": {
                        "groq": bool(GROQ_API_KEY),
                        "deepseek": bool(DEEPSEEK_API_KEY),
                        "runpod": bool(RUNPOD_API_KEY),
                        "voice_clone": bool(VOICE_CLONE_URL),
                    },
                })

            elif self.path == "/session":
                sid = data.get("session_id", "")
                session = CallSession(sid)
                self._respond(200, {
                    "session_id": sid,
                    "turn_count": session.turn_count,
                    "history": session.history,
                })

            else:
                self._respond(404, {"error": f"Unknown endpoint: {self.path}"})

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "Use POST"})

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

        def log_message(self, format, *args):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")

    server = HTTPServer(("0.0.0.0", port), CallHandler)
    print(f"Call Engine running on http://localhost:{port}")
    print(f"  POST /turn    — Process one conversation turn")
    print(f"  POST /session — Get session history")
    print(f"  GET  /health  — Health check")
    print(f"  Voice clone:  {'ENABLED' if VOICE_CLONE_URL else 'DISABLED (using default voice)'}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python call_engine.py serve [--port 5050]")
        print("  python call_engine.py turn --text 'Hola' --session-id abc123")
        print("  python call_engine.py turn --audio input.wav --session-id abc123")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "serve":
        port = 5050
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        serve(port)

    elif cmd == "turn":
        session_id = "test"
        audio_path = None
        text = None

        if "--session-id" in sys.argv:
            session_id = sys.argv[sys.argv.index("--session-id") + 1]
        if "--audio" in sys.argv:
            audio_path = sys.argv[sys.argv.index("--audio") + 1]
        if "--text" in sys.argv:
            text = sys.argv[sys.argv.index("--text") + 1]

        result = process_turn(session_id=session_id, audio_path=audio_path, text=text)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
