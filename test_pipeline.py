"""
Full pipeline test: Simulates one AI phone call turn.

Flow: User speaks → Groq STT → DeepSeek LLM → Chatterbox TTS → Audio
Cost per turn: ~$0.001 (practically free)

Components:
  1. STT:  Groq Whisper Large V3 Turbo ($0.04/audio hour)
  2. LLM:  DeepSeek V4 Pro ($0.435/1M in, $0.87/1M out)
  3. TTS:  Chatterbox Turbo on RunPod (FREE)
"""
import os
import sys
import json
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

CHATTERBOX_URL = "https://api.runpod.ai/v2/chatterbox-turbo"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# System prompt for the HR assistant persona
SYSTEM_PROMPT = """Eres la asistente virtual de Recursos Humanos de Global Advance.
Tu nombre es Carolina. Hablas español colombiano natural y cálido.
Eres amable, profesional y directa. Tuteas a los empleados.
Respondes en 1-3 frases cortas, como en una llamada telefónica real.
No uses emojis ni formato especial — solo texto hablado natural."""


def step_stt(audio_path: str, language: str = "es") -> dict:
    """Step 1: Speech-to-Text via Groq Whisper."""
    print("\n" + "=" * 60)
    print("STEP 1: STT (Groq Whisper)")
    print("=" * 60)

    if not GROQ_API_KEY:
        print("  ⚠ GROQ_API_KEY not set — skipping STT")
        return {"text": None, "duration": 0, "time": 0}

    file_size = os.path.getsize(audio_path)
    print(f"  Input: {audio_path} ({file_size:,} bytes)")

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
        )

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return {"text": None, "duration": 0, "time": elapsed}

    result = resp.json()
    text = result.get("text", "").strip()
    duration = result.get("duration", 0)

    print(f"  Transcription: '{text}'")
    print(f"  Audio duration: {duration:.1f}s")
    print(f"  Response time: {elapsed:.3f}s")
    print(f"  Cost: ~${duration / 3600 * 0.04:.6f}")

    return {"text": text, "duration": duration, "time": elapsed}


def step_llm(user_text: str, conversation_history: list = None) -> dict:
    """Step 2: LLM response via DeepSeek V4 Pro."""
    print("\n" + "=" * 60)
    print("STEP 2: LLM (DeepSeek V4 Pro)")
    print("=" * 60)

    if not DEEPSEEK_API_KEY:
        # Simulate LLM response for testing
        print("  ⚠ DEEPSEEK_API_KEY not set — using simulated response")
        simulated = (
            "Hola, gracias por llamar. Soy Carolina de Recursos Humanos. "
            "En qué te puedo colaborar hoy?"
        )
        print(f"  Simulated: '{simulated}'")
        return {"text": simulated, "time": 0, "simulated": True}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_text})

    print(f"  User: '{user_text[:80]}'")

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
    )

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return {"text": None, "time": elapsed}

    result = resp.json()
    reply = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})

    print(f"  Reply: '{reply}'")
    print(f"  Tokens: {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out")
    print(f"  Response time: {elapsed:.3f}s")

    in_cost = usage.get("prompt_tokens", 0) / 1_000_000 * 0.435
    out_cost = usage.get("completion_tokens", 0) / 1_000_000 * 0.87
    print(f"  Cost: ~${in_cost + out_cost:.6f}")

    return {"text": reply, "time": elapsed, "simulated": False}


def step_tts(text: str, voice: str = "lucy", voice_url: str = None) -> dict:
    """Step 3: Text-to-Speech via Chatterbox Turbo."""
    print("\n" + "=" * 60)
    print("STEP 3: TTS (Chatterbox Turbo)")
    print("=" * 60)

    if not RUNPOD_API_KEY:
        print("  ⚠ RUNPOD_API_KEY not set — skipping TTS")
        return {"audio_path": None, "time": 0}

    payload = {"input": {"prompt": text, "format": "wav"}}

    if voice_url:
        payload["input"]["voice_url"] = voice_url
        print(f"  Voice: cloned from {voice_url[:50]}...")
    else:
        payload["input"]["voice"] = voice
        print(f"  Voice: {voice} (built-in)")

    print(f"  Text: '{text[:80]}...'")
    print(f"  Cost: $0.00 (FREE)")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    start = time.time()

    # Submit job
    resp = requests.post(f"{CHATTERBOX_URL}/run", json=payload, headers=headers)
    data = resp.json()

    if "id" not in data:
        print(f"  ERROR: {json.dumps(data, indent=2)}")
        return {"audio_path": None, "time": time.time() - start}

    job_id = data["id"]
    print(f"  Job ID: {job_id}")

    # Poll for result
    while True:
        status_resp = requests.get(f"{CHATTERBOX_URL}/status/{job_id}", headers=headers)
        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            audio_path = "pipeline_output.wav"

            if isinstance(output, dict) and "audio_url" in output:
                audio_resp = requests.get(output["audio_url"])
                with open(audio_path, "wb") as f:
                    f.write(audio_resp.content)
                print(f"  Audio URL: {output['audio_url'][:60]}...")
            elif isinstance(output, dict) and "audio_base64" in output:
                audio_bytes = base64.b64decode(output["audio_base64"])
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
            elif isinstance(output, str):
                try:
                    audio_bytes = base64.b64decode(output)
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                except Exception:
                    print(f"  Output: {output[:200]}")
                    return {"audio_path": None, "time": time.time() - start}

            elapsed = time.time() - start
            file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            print(f"  Audio saved: {audio_path} ({file_size:,} bytes)")
            print(f"  Total TTS time: {elapsed:.1f}s")

            return {"audio_path": audio_path, "time": elapsed}

        elif status == "FAILED":
            print(f"  FAILED: {json.dumps(status_data, indent=2)[:300]}")
            return {"audio_path": None, "time": time.time() - start}

        time.sleep(2)


def run_pipeline(
    audio_input: str = None,
    text_input: str = None,
    voice: str = "lucy",
    voice_url: str = None,
):
    """Run the complete pipeline: STT → LLM → TTS."""
    print("\n" + "#" * 60)
    print("  AI PHONE CALL PIPELINE — Full Round Trip")
    print("#" * 60)

    total_start = time.time()

    # Step 1: STT (if audio input provided)
    if audio_input:
        stt_result = step_stt(audio_input)
        user_text = stt_result["text"]
        if not user_text:
            print("\n❌ STT failed — aborting pipeline")
            return
    elif text_input:
        user_text = text_input
        stt_result = {"text": text_input, "duration": 0, "time": 0}
        print(f"\n  (Skipping STT — using text input: '{text_input}')")
    else:
        print("ERROR: Provide either audio_input or text_input")
        return

    # Step 2: LLM
    llm_result = step_llm(user_text)
    if not llm_result["text"]:
        print("\n❌ LLM failed — aborting pipeline")
        return

    # Step 3: TTS
    tts_result = step_tts(llm_result["text"], voice=voice, voice_url=voice_url)

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  User said:  '{user_text}'")
    print(f"  AI replied: '{llm_result['text']}'")
    print(f"  Audio:      {tts_result.get('audio_path', 'N/A')}")
    print()
    print(f"  STT time:   {stt_result['time']:.3f}s")
    print(f"  LLM time:   {llm_result['time']:.3f}s {'(simulated)' if llm_result.get('simulated') else ''}")
    print(f"  TTS time:   {tts_result['time']:.1f}s")
    print(f"  TOTAL:      {total_time:.1f}s")
    print()

    # Verify round-trip: transcribe the output audio
    if tts_result.get("audio_path") and GROQ_API_KEY:
        print("  VERIFICATION: Transcribing output audio...")
        verify = step_stt(tts_result["audio_path"])
        if verify["text"]:
            print(f"\n  ✅ Round-trip verified: '{verify['text']}'")
        else:
            print(f"\n  ⚠ Could not verify round-trip")

    print("\n" + "#" * 60)
    print(f"  Pipeline complete in {total_time:.1f}s")
    print("#" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--text":
        # Text input mode (skip STT)
        text = sys.argv[2] if len(sys.argv) > 2 else "Hola, necesito saber cuántos días de vacaciones me quedan."
        voice_url = None
        if "--voice-url" in sys.argv:
            idx = sys.argv.index("--voice-url")
            voice_url = sys.argv[idx + 1]
        run_pipeline(text_input=text, voice_url=voice_url)
    elif len(sys.argv) > 1:
        # Audio input mode
        run_pipeline(audio_input=sys.argv[1])
    else:
        print("AI Phone Call Pipeline — End-to-End Test")
        print()
        print("Usage:")
        print("  From audio:  python test_pipeline.py input.wav")
        print("  From text:   python test_pipeline.py --text 'Hola, cuantos dias de vacaciones tengo?'")
        print("  With clone:  python test_pipeline.py --text 'Hola' --voice-url https://example.com/ref.wav")
        print()
        print("Running default test (audio → STT → LLM → TTS → verify)...")
        print()

        # Default: use existing test audio
        if os.path.exists("test_turbo_output.wav"):
            run_pipeline(audio_input="test_turbo_output.wav")
        else:
            run_pipeline(text_input="Hola, necesito saber cuántos días de vacaciones me quedan este año.")
