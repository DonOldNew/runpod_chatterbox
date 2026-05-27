"""
Test script for RunPod Chatterbox TTS + Whisper STT endpoint.
Sends a TTS request with a reference voice to clone.
"""
import os
import base64
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


def test_tts(prompt: str, audio_ref_path: str):
    """Test TTS voice cloning."""
    print(f"Loading reference audio: {audio_ref_path}")
    with open(audio_ref_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "mode": "tts",
            "prompt": prompt,
            "audio_ref": audio_b64,
        }
    }

    print(f"Sending TTS request: '{prompt[:60]}...'")
    print("This may take 1-3 minutes (cold start + generation)...")

    # Use /run for async, then poll
    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
    data = resp.json()
    job_id = data.get("id")
    print(f"Job ID: {job_id}")

    # Poll for result
    while True:
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        status_data = status_resp.json()
        status = status_data.get("status")
        print(f"  Status: {status}")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            if "audio_base64" in output:
                audio_bytes = base64.b64decode(output["audio_base64"])
                out_path = "test_output.wav"
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"Audio saved to {out_path}")
                print(f"Sample rate: {output.get('sample_rate')}")
            else:
                print(f"Output: {json.dumps(output, indent=2)}")
            break
        elif status == "FAILED":
            print(f"FAILED: {status_data}")
            break
        else:
            time.sleep(5)


def test_stt(audio_path: str, language: str = "es"):
    """Test STT transcription."""
    print(f"Loading audio: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "mode": "stt",
            "audio": audio_b64,
            "language": language,
        }
    }

    print(f"Sending STT request (language: {language})...")

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
    data = resp.json()
    job_id = data.get("id")
    print(f"Job ID: {job_id}")

    while True:
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        status_data = status_resp.json()
        status = status_data.get("status")
        print(f"  Status: {status}")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            print(f"Transcription: {output.get('text')}")
            print(f"Language: {output.get('language')} ({output.get('language_probability')})")
            break
        elif status == "FAILED":
            print(f"FAILED: {status_data}")
            break
        else:
            time.sleep(5)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  TTS: python test_endpoint.py tts 'Hello world' reference_voice.wav")
        print("  STT: python test_endpoint.py stt audio.wav [language]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "tts":
        prompt = sys.argv[2]
        ref_audio = sys.argv[3]
        test_tts(prompt, ref_audio)
    elif mode == "stt":
        audio_file = sys.argv[2]
        lang = sys.argv[3] if len(sys.argv) > 3 else "es"
        test_stt(audio_file, lang)
