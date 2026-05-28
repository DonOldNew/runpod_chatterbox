"""
Test Chatterbox Streaming TTS on RunPod Serverless.

Tests:
  1. Submit text → poll /stream/{job_id} → verify audio chunks arrive
  2. Measure TTFB (time to first chunk)
  3. Test with voice cloning if --voice-url provided
  4. Verify audio quality by playing or saving

Usage:
  # Test default voice:
  python test_streaming_tts.py

  # Test with voice cloning:
  python test_streaming_tts.py --voice-url https://example.com/reference.wav

  # Full test (3 phrases, save audio):
  python test_streaming_tts.py --full-test --save
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_STREAMING_ENDPOINT")

if not API_KEY:
    print("ERROR: RUNPOD_API_KEY not set in .env")
    sys.exit(1)

if not ENDPOINT_ID:
    print("ERROR: RUNPOD_STREAMING_ENDPOINT not set in .env")
    print("Deploy the streaming endpoint first: ./deploy_streaming_tts.sh <docker-user>")
    sys.exit(1)

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

TEST_PHRASES = [
    "Hola, soy Carolina de Recursos Humanos de Global Advance.",
    "Tu solicitud de vacaciones fue aprobada para la próxima semana.",
    "Necesito que me envíes tu certificado de salud antes del viernes.",
]


def test_streaming(text: str, voice_url: str = None, save: bool = False) -> dict:
    """Submit text and poll /stream/{job_id} for audio chunks."""
    print(f"\n{'='*60}")
    print(f"Text: {text[:80]}...")
    print(f"Voice: {'clone' if voice_url else 'default'}")
    print(f"{'='*60}")

    payload = {
        "input": {
            "text": text,
            "chunk_size": 25,
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
        }
    }
    if voice_url:
        payload["input"]["voice_url"] = voice_url

    # Submit job
    start = time.time()
    print(f"\n[{time.time()-start:.1f}s] Submitting job...")
    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS, timeout=30)
    data = resp.json()

    if "id" not in data:
        print(f"ERROR: Submit failed: {json.dumps(data, indent=2)}")
        return {"error": "submit_failed"}

    job_id = data["id"]
    print(f"[{time.time()-start:.1f}s] Job ID: {job_id}")

    # Poll /stream/{job_id}
    chunks_received = 0
    total_audio_bytes = 0
    all_audio = b""
    ttfb = None
    first_chunk_time = None

    for poll in range(200):  # max ~100s
        try:
            resp = requests.get(
                f"{BASE_URL}/stream/{job_id}", headers=HEADERS, timeout=10
            )
            stream_data = resp.json()
        except Exception as e:
            print(f"[{time.time()-start:.1f}s] Poll error: {e}")
            time.sleep(0.5)
            continue

        for chunk in stream_data.get("stream", []):
            output = chunk.get("output", chunk)

            if "error" in output:
                print(f"ERROR: {output['error']}")
                return {"error": output["error"]}

            if output.get("done"):
                elapsed = time.time() - start
                print(f"\n[{elapsed:.1f}s] COMPLETE!")
                print(f"  Chunks: {output.get('total_chunks', chunks_received)}")
                print(f"  Audio duration: {output.get('total_audio_duration', 0):.1f}s")
                print(f"  Generation time: {output.get('total_generation_time', 0):.1f}s")
                print(f"  RTF: {output.get('rtf', 0):.2f}")
                print(f"  TTFB: {ttfb:.3f}s" if ttfb else "  TTFB: N/A")
                print(f"  Total audio bytes: {total_audio_bytes:,}")

                if save and all_audio:
                    filename = f"streaming_test_{int(time.time())}.wav"
                    with open(filename, "wb") as f:
                        f.write(all_audio)
                    print(f"  Saved: {filename}")

                return {
                    "ttfb": ttfb,
                    "total_time": elapsed,
                    "chunks": chunks_received,
                    "audio_duration": output.get("total_audio_duration", 0),
                    "audio_bytes": total_audio_bytes,
                }

            if "audio_base64" in output:
                wav_data = base64.b64decode(output["audio_base64"])
                total_audio_bytes += len(wav_data)
                chunks_received += 1

                if chunks_received == 1:
                    first_chunk_time = time.time() - start
                    ttfb = output.get("ttfb", first_chunk_time)
                    print(f"[{first_chunk_time:.1f}s] First chunk! TTFB={ttfb:.3f}s, {len(wav_data):,} bytes")
                else:
                    print(f"[{time.time()-start:.1f}s] Chunk {chunks_received}: {len(wav_data):,} bytes ({output.get('chunk_duration', 0):.1f}s)")

                if save:
                    all_audio += wav_data

        status = stream_data.get("status")
        if status == "FAILED":
            print(f"ERROR: Job failed: {json.dumps(stream_data)[:200]}")
            return {"error": "job_failed"}

        time.sleep(0.5)

    print("ERROR: Timeout waiting for stream")
    return {"error": "timeout"}


def main():
    parser = argparse.ArgumentParser(description="Test Streaming TTS")
    parser.add_argument("--voice-url", help="Voice cloning reference audio URL")
    parser.add_argument("--text", help="Custom text to synthesize")
    parser.add_argument("--full-test", action="store_true", help="Test all 3 phrases")
    parser.add_argument("--save", action="store_true", help="Save audio files")
    args = parser.parse_args()

    print(f"Endpoint: {ENDPOINT_ID}")
    print(f"Base URL: {BASE_URL}")

    if args.text:
        phrases = [args.text]
    elif args.full_test:
        phrases = TEST_PHRASES
    else:
        phrases = [TEST_PHRASES[0]]

    results = []
    for phrase in phrases:
        result = test_streaming(phrase, voice_url=args.voice_url, save=args.save)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if successful:
        avg_ttfb = sum(r["ttfb"] for r in successful if r.get("ttfb")) / len(successful)
        avg_time = sum(r["total_time"] for r in successful) / len(successful)
        print(f"  Successful: {len(successful)}/{len(results)}")
        print(f"  Avg TTFB: {avg_ttfb:.3f}s")
        print(f"  Avg total: {avg_time:.1f}s")

    if failed:
        print(f"  Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"    Error: {r['error']}")


if __name__ == "__main__":
    main()
