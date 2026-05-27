"""
Test script for Groq's Whisper API (Speech-to-Text).
Whisper Large V3 Turbo: $0.04/audio hour, 228x realtime speed.
For 800 calls/month × 10 min speech = ~$5/month.

Get your free API key at: https://console.groq.com/keys
"""
import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("=" * 60)
    print("GROQ_API_KEY not found in .env!")
    print()
    print("To set up Groq (free tier available):")
    print("1. Go to https://console.groq.com")
    print("2. Sign up (free)")
    print("3. Go to API Keys → Create API Key")
    print("4. Add to .env: GROQ_API_KEY=gsk_xxxxx")
    print("=" * 60)
    exit(1)


def test_stt(audio_path: str, language: str = "es"):
    """Test STT transcription with Groq Whisper."""
    print(f"[Groq Whisper] Transcribing: {audio_path}")
    print(f"  Language: {language}")
    print(f"  Model: whisper-large-v3-turbo ($0.04/audio hour)")

    file_size = os.path.getsize(audio_path)
    print(f"  File size: {file_size:,} bytes")

    start_time = time.time()

    with open(audio_path, "rb") as audio_file:
        resp = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            files={
                "file": (os.path.basename(audio_path), audio_file, "audio/wav"),
            },
            data={
                "model": "whisper-large-v3-turbo",
                "language": language,
                "response_format": "verbose_json",
            },
        )

    elapsed = time.time() - start_time

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return None

    result = resp.json()
    text = result.get("text", "")
    duration = result.get("duration", 0)

    print(f"  Transcription: '{text}'")
    print(f"  Audio duration: {duration:.1f}s")
    print(f"  API response time: {elapsed:.2f}s")
    print(f"  Speed factor: {duration / elapsed:.0f}x realtime" if elapsed > 0 else "")

    # Show segments if available
    segments = result.get("segments", [])
    if segments:
        print(f"  Segments: {len(segments)}")
        for seg in segments[:5]:
            print(f"    [{seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s] {seg.get('text', '')}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 test_groq_whisper.py <audio_file> [language]")
        print()
        print("Testing with test_turbo_output.wav (Chatterbox Turbo output)...")

        # Test with the Chatterbox Turbo output
        if os.path.exists("test_turbo_output.wav"):
            test_stt("test_turbo_output.wav", "es")
        elif os.path.exists("test_output.wav"):
            test_stt("test_output.wav", "es")
        else:
            print("No test audio files found!")
    else:
        audio_file = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "es"
        test_stt(audio_file, lang)
