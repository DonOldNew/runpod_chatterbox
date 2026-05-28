"""
RunPod Serverless Generator Handler — Chatterbox Streaming TTS.

Streams base64-encoded WAV audio chunks via RunPod's generator pattern.
Supports zero-shot voice cloning via voice_url parameter.

Endpoint: POST /run or /runsync → then GET /stream/{job_id} to poll chunks.

Input:
  {
    "text": "Hola, soy Carolina de Recursos Humanos.",
    "voice_url": "https://example.com/reference.wav",  # optional
    "chunk_size": 25,        # tokens per chunk (lower = faster TTFB)
    "exaggeration": 0.5,     # emotion intensity 0.0-1.0+
    "cfg_weight": 0.5,       # pace/guidance
    "temperature": 0.8,      # sampling randomness
  }

Each streamed chunk:
  {
    "chunk_index": 0,
    "audio_base64": "UklGRi...",   # base64 WAV (~1 second)
    "chunk_duration": 1.023,
    "ttfb": 0.472,                  # only on first chunk
  }

Final chunk:
  {
    "done": true,
    "total_chunks": 6,
    "total_audio_duration": 5.8,
    "total_generation_time": 2.9,
    "sample_rate": 24000,
  }

GPU: 16-24 GB VRAM (A5000, L4, RTX 4090 class)
TTFB: ~472ms on RTX 4090 with chunk_size=25
"""
import os
import io
import base64
import tempfile
import time

import requests
import torch
import torchaudio
import runpod

from chatterbox.tts import ChatterboxTTS

# ──────────────────────────────────────────────
# Global model — loaded once at cold start
# ──────────────────────────────────────────────

MODEL: ChatterboxTTS = None
SAMPLE_RATE: int = 24000


def load_model():
    """Load ChatterboxTTS at worker startup."""
    global MODEL, SAMPLE_RATE
    print("Loading ChatterboxTTS model...")
    start = time.time()

    MODEL = ChatterboxTTS.from_pretrained(device="cuda")
    SAMPLE_RATE = MODEL.sr  # 24000

    elapsed = time.time() - start
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"Model loaded in {elapsed:.1f}s | GPU: {gpu_name} | VRAM: {vram:.1f} GB")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def download_voice_ref(url: str) -> str:
    """Download voice reference audio to temp file. Returns path."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    ext = ".wav"
    ct = resp.headers.get("Content-Type", "").lower()
    if "mp3" in ct or url.lower().endswith(".mp3"):
        ext = ".mp3"
    elif "ogg" in ct or url.lower().endswith(".ogg"):
        ext = ".ogg"

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(resp.content)
    tmp.flush()
    tmp.close()
    return tmp.name


def tensor_to_base64_wav(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert [1, samples] tensor to base64 WAV string."""
    buf = io.BytesIO()
    torchaudio.save(buf, audio_tensor.cpu(), sample_rate, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ──────────────────────────────────────────────
# Handler (generator = streaming)
# ──────────────────────────────────────────────

def handler(job):
    """RunPod generator handler — yields audio chunks as they're generated."""
    inp = job["input"]
    text = inp.get("text", "").strip()
    voice_url = inp.get("voice_url")
    chunk_size = inp.get("chunk_size", 25)
    exaggeration = inp.get("exaggeration", 0.5)
    cfg_weight = inp.get("cfg_weight", 0.5)
    temperature = inp.get("temperature", 0.8)
    context_window = inp.get("context_window", 50)

    if not text:
        yield {"error": "No text provided"}
        return

    # Download voice reference if provided
    voice_ref_path = None
    try:
        if voice_url:
            voice_ref_path = download_voice_ref(voice_url)
            print(f"Voice ref downloaded: {voice_ref_path}")
    except Exception as e:
        yield {"error": f"Failed to download voice reference: {e}"}
        return

    # Stream audio chunks
    start_time = time.time()
    chunk_index = 0
    total_audio_duration = 0.0
    ttfb = None

    try:
        for audio_chunk, metrics in MODEL.generate_stream(
            text=text,
            audio_prompt_path=voice_ref_path,
            chunk_size=chunk_size,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            context_window=context_window,
            print_metrics=False,
        ):
            chunk_duration = audio_chunk.shape[-1] / SAMPLE_RATE
            total_audio_duration += chunk_duration

            if chunk_index == 0:
                ttfb = time.time() - start_time
                print(f"TTFB: {ttfb:.3f}s")

            audio_b64 = tensor_to_base64_wav(audio_chunk, SAMPLE_RATE)

            yield {
                "chunk_index": chunk_index,
                "audio_base64": audio_b64,
                "chunk_duration": round(chunk_duration, 3),
                "ttfb": round(ttfb, 3) if chunk_index == 0 else None,
            }
            chunk_index += 1

    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        yield {"error": f"Generation failed: {e}"}
        return
    finally:
        if voice_ref_path and os.path.exists(voice_ref_path):
            os.remove(voice_ref_path)

    total_time = time.time() - start_time
    rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0

    print(f"Done: {chunk_index} chunks | {total_audio_duration:.1f}s audio | RTF {rtf:.2f}")

    yield {
        "done": True,
        "total_chunks": chunk_index,
        "total_audio_duration": round(total_audio_duration, 3),
        "total_generation_time": round(total_time, 3),
        "rtf": round(rtf, 3),
        "sample_rate": SAMPLE_RATE,
    }


# ──────────────────────────────────────────────
# Non-streaming fallback (for /runsync without streaming)
# ──────────────────────────────────────────────

def handler_batch(job):
    """Non-streaming fallback: returns complete audio as single base64 WAV."""
    inp = job["input"]
    text = inp.get("text", "").strip()
    voice_url = inp.get("voice_url")
    exaggeration = inp.get("exaggeration", 0.5)
    cfg_weight = inp.get("cfg_weight", 0.5)
    temperature = inp.get("temperature", 0.8)
    mode = inp.get("mode", "stream")

    if mode == "batch":
        # Full generation, no streaming
        if not text:
            return {"error": "No text provided"}

        voice_ref_path = None
        try:
            if voice_url:
                voice_ref_path = download_voice_ref(voice_url)

            start = time.time()
            wav = MODEL.generate(
                text=text,
                audio_prompt_path=voice_ref_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            elapsed = time.time() - start

            audio_b64 = tensor_to_base64_wav(wav, SAMPLE_RATE)
            duration = wav.shape[-1] / SAMPLE_RATE

            return {
                "audio_base64": audio_b64,
                "duration": round(duration, 3),
                "generation_time": round(elapsed, 3),
                "sample_rate": SAMPLE_RATE,
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            if voice_ref_path and os.path.exists(voice_ref_path):
                os.remove(voice_ref_path)

    # Default: use streaming handler
    return handler(job)


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

load_model()

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
