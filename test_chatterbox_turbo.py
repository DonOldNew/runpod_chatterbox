"""
Test script for RunPod's Chatterbox Turbo PUBLIC endpoint.
This is FREE ($0.00/1000 chars) and replaces our custom serverless endpoint for TTS.
Supports voice cloning via voice_url parameter.
"""
import os
import json
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
BASE_URL = "https://api.runpod.ai/v2/chatterbox-turbo"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


def test_tts_default_voice(prompt: str, voice: str = "lucy"):
    """Test TTS with a built-in voice (no cloning)."""
    payload = {
        "input": {
            "prompt": prompt,
            "voice": voice,
            "format": "wav",
        }
    }

    print(f"[Chatterbox Turbo] Sending TTS request with voice '{voice}'...")
    print(f"  Prompt: '{prompt[:80]}...'")
    print(f"  Cost: $0.00 (FREE)")

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
    data = resp.json()

    if "id" not in data:
        print(f"  ERROR: {json.dumps(data, indent=2)}")
        return None

    job_id = data["id"]
    print(f"  Job ID: {job_id}")

    return poll_result(job_id)


def test_tts_voice_clone(prompt: str, voice_url: str):
    """Test TTS with voice cloning via URL to reference audio."""
    payload = {
        "input": {
            "prompt": prompt,
            "voice_url": voice_url,
            "format": "wav",
        }
    }

    print(f"[Chatterbox Turbo] Sending TTS request with voice cloning...")
    print(f"  Prompt: '{prompt[:80]}...'")
    print(f"  Voice URL: {voice_url[:60]}...")
    print(f"  Cost: $0.00 (FREE)")

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
    data = resp.json()

    if "id" not in data:
        print(f"  ERROR: {json.dumps(data, indent=2)}")
        return None

    job_id = data["id"]
    print(f"  Job ID: {job_id}")

    return poll_result(job_id)


def poll_result(job_id: str):
    """Poll for job result."""
    while True:
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        status_data = status_resp.json()
        status = status_data.get("status")
        print(f"  Status: {status}")

        if status == "COMPLETED":
            output = status_data.get("output", {})

            # Check if output is audio data (base64 or URL)
            if isinstance(output, str):
                # Output might be base64 audio directly
                try:
                    audio_bytes = base64.b64decode(output)
                    out_path = "test_turbo_output.wav"
                    with open(out_path, "wb") as f:
                        f.write(audio_bytes)
                    print(f"  Audio saved to {out_path} ({len(audio_bytes)} bytes)")
                    return out_path
                except Exception:
                    pass

            if isinstance(output, dict):
                # Check various output formats
                if "audio_base64" in output:
                    audio_bytes = base64.b64decode(output["audio_base64"])
                    out_path = "test_turbo_output.wav"
                    with open(out_path, "wb") as f:
                        f.write(audio_bytes)
                    print(f"  Audio saved to {out_path} ({len(audio_bytes)} bytes)")
                    return out_path
                elif "audio_url" in output:
                    print(f"  Audio URL: {output['audio_url']}")
                    # Download the audio
                    audio_resp = requests.get(output["audio_url"])
                    out_path = "test_turbo_output.wav"
                    with open(out_path, "wb") as f:
                        f.write(audio_resp.content)
                    print(f"  Audio saved to {out_path} ({len(audio_resp.content)} bytes)")
                    return out_path
                elif "output" in output:
                    # Nested output
                    print(f"  Output: {json.dumps(output, indent=2)[:500]}")
                else:
                    print(f"  Output: {json.dumps(output, indent=2)[:500]}")

            return output

        elif status == "FAILED":
            print(f"  FAILED: {json.dumps(status_data, indent=2)}")
            return None
        else:
            time.sleep(3)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Default voice:  python test_chatterbox_turbo.py 'Hello, this is a test'")
        print("  Voice clone:    python test_chatterbox_turbo.py 'Hello' --voice-url https://example.com/ref.wav")
        print()
        print("Built-in voices: lucy (default)")
        print()

        # Quick test with default voice
        print("Running quick test with default voice...")
        test_tts_default_voice(
            "Hola, soy la asistente virtual de Global Advance. Como puedo ayudarte hoy?",
            voice="lucy"
        )
    else:
        prompt = sys.argv[1]

        if "--voice-url" in sys.argv:
            idx = sys.argv.index("--voice-url")
            voice_url = sys.argv[idx + 1]
            test_tts_voice_clone(prompt, voice_url)
        else:
            voice = "lucy"
            if "--voice" in sys.argv:
                idx = sys.argv.index("--voice")
                voice = sys.argv[idx + 1]
            test_tts_default_voice(prompt, voice)
