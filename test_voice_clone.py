"""
Voice Cloning Test — Chatterbox Turbo.

Tests the complete voice cloning pipeline:
1. Upload a reference voice sample (WAV, min 6 seconds)
2. Generate speech in the cloned voice
3. Verify with Groq Whisper STT

Usage:
  # With a URL to reference audio:
  python test_voice_clone.py --ref-url https://example.com/carolina.wav

  # With a local file (will be uploaded to a temp hosting):
  python test_voice_clone.py --ref-file carolina.wav

  # Test with multiple Spanish phrases:
  python test_voice_clone.py --ref-url URL --full-test
"""
import os
import sys
import json
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHATTERBOX_URL = "https://api.runpod.ai/v2/chatterbox-turbo"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Test phrases — natural Colombian Spanish HR scenarios
TEST_PHRASES = [
    "Hola, soy Carolina de Recursos Humanos. En que te puedo ayudar hoy?",
    "Claro que si, dejame revisar tu caso en el sistema. Un momentico por favor.",
    "Listo, ya encontre tu informacion. Tienes quince dias de vacaciones pendientes.",
    "Dale, entonces te programo la cita con tu jefe para el viernes a las diez de la manana.",
    "Perfecto, quedo registrada tu solicitud. Te llamo manana para confirmarte. Que tengas buen dia!",
]


def generate_with_voice_clone(text: str, voice_url: str, output_name: str = "clone_output.wav"):
    """Generate TTS with cloned voice."""
    payload = {
        "input": {
            "prompt": text,
            "voice_url": voice_url,
            "format": "wav",
        }
    }

    print(f"  Generating: '{text[:60]}...'")

    resp = requests.post(f"{CHATTERBOX_URL}/run", json=payload, headers=HEADERS)
    data = resp.json()

    if "id" not in data:
        print(f"  ERROR: {json.dumps(data, indent=2)}")
        return None

    job_id = data["id"]
    start = time.time()

    # Poll for result
    while True:
        status_resp = requests.get(f"{CHATTERBOX_URL}/status/{job_id}", headers=HEADERS)
        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            elapsed = time.time() - start

            if isinstance(output, dict) and "audio_url" in output:
                audio_resp = requests.get(output["audio_url"])
                with open(output_name, "wb") as f:
                    f.write(audio_resp.content)
                print(f"  Saved: {output_name} ({len(audio_resp.content):,} bytes, {elapsed:.1f}s)")
                return output_name
            elif isinstance(output, dict) and "audio_base64" in output:
                audio_bytes = base64.b64decode(output["audio_base64"])
                with open(output_name, "wb") as f:
                    f.write(audio_bytes)
                print(f"  Saved: {output_name} ({len(audio_bytes):,} bytes, {elapsed:.1f}s)")
                return output_name

            print(f"  Unexpected output: {json.dumps(output, indent=2)[:200]}")
            return None

        elif status == "FAILED":
            print(f"  FAILED: {json.dumps(status_data, indent=2)[:200]}")
            return None

        time.sleep(2)


def verify_with_stt(audio_path: str, expected_text: str = "") -> str:
    """Verify audio content with Groq Whisper."""
    if not GROQ_API_KEY:
        print("  (STT verification skipped — no GROQ_API_KEY)")
        return ""

    with open(audio_path, "rb") as f:
        resp = requests.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"model": "whisper-large-v3-turbo", "language": "es"},
        )

    if resp.status_code != 200:
        print(f"  STT error: {resp.status_code}")
        return ""

    text = resp.json().get("text", "").strip()
    print(f"  STT result: '{text}'")
    return text


def run_test(voice_url: str, full_test: bool = False):
    """Run voice cloning test."""
    print("=" * 60)
    print("VOICE CLONING TEST — Chatterbox Turbo")
    print("=" * 60)
    print(f"Reference voice: {voice_url}")
    print()

    phrases = TEST_PHRASES if full_test else TEST_PHRASES[:2]

    for i, phrase in enumerate(phrases, 1):
        print(f"\n--- Test {i}/{len(phrases)} ---")
        output_name = f"clone_test_{i}.wav"
        result = generate_with_voice_clone(phrase, voice_url, output_name)

        if result:
            verify_with_stt(result, phrase)

    print(f"\n{'=' * 60}")
    print(f"Voice cloning test complete. {len(phrases)} phrases generated.")
    print(f"Listen to clone_test_*.wav files to evaluate voice quality.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if "--ref-url" in sys.argv:
        idx = sys.argv.index("--ref-url")
        voice_url = sys.argv[idx + 1]
        full = "--full-test" in sys.argv
        run_test(voice_url, full)

    elif "--ref-file" in sys.argv:
        idx = sys.argv.index("--ref-file")
        ref_file = sys.argv[idx + 1]

        if not os.path.exists(ref_file):
            print(f"File not found: {ref_file}")
            sys.exit(1)

        print("Note: Chatterbox Turbo needs a URL to the reference audio.")
        print("You need to host the file somewhere accessible (e.g., Google Drive, Dropbox).")
        print()
        print("Options:")
        print("  1. Upload to Google Drive → Share → Copy link")
        print("  2. Upload to file.io (temp): curl -F 'file=@carolina.wav' https://file.io")
        print("  3. Host on your own server")
        print()
        print(f"File info: {ref_file} ({os.path.getsize(ref_file):,} bytes)")
        sys.exit(0)

    else:
        print("Voice Cloning Test for Chatterbox Turbo")
        print()
        print("Usage:")
        print("  python test_voice_clone.py --ref-url https://example.com/voice.wav")
        print("  python test_voice_clone.py --ref-url URL --full-test")
        print("  python test_voice_clone.py --ref-file local_voice.wav")
        print()
        print("Requirements:")
        print("  - Reference audio: WAV format, minimum 6 seconds")
        print("  - Clear speech, minimal background noise")
        print("  - Same language as target (Spanish preferred)")
        print("  - Single speaker only")
        print()
        print("The reference audio must be hosted at a publicly accessible URL.")
        print("Chatterbox Turbo downloads it during generation.")
