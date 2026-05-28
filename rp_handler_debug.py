"""
Debug handler — tests infrastructure without model loading.
Echoes back text + reports system info.
"""
import os
import sys
import time
import runpod

print("=== DEBUG HANDLER STARTING ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Working dir: {os.getcwd()}", flush=True)

# Test imports one by one
errors = []

try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
except Exception as e:
    errors.append(f"torch: {e}")
    print(f"ERROR importing torch: {e}", flush=True)

try:
    import torchaudio
    print(f"torchaudio: {torchaudio.__version__}", flush=True)
except Exception as e:
    errors.append(f"torchaudio: {e}")
    print(f"ERROR importing torchaudio: {e}", flush=True)

try:
    import transformers
    print(f"transformers: {transformers.__version__}", flush=True)
except Exception as e:
    errors.append(f"transformers: {e}")
    print(f"ERROR importing transformers: {e}", flush=True)

try:
    from chatterbox.tts import ChatterboxTTS
    print("chatterbox.tts: OK", flush=True)
except Exception as e:
    errors.append(f"chatterbox: {e}")
    print(f"ERROR importing chatterbox: {e}", flush=True)

# Try model loading
MODEL = None
load_error = None
try:
    print("Loading model...", flush=True)
    start = time.time()
    MODEL = ChatterboxTTS.from_pretrained(device="cuda")
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s", flush=True)
except Exception as e:
    load_error = str(e)
    import traceback
    print(f"MODEL LOAD ERROR: {e}", flush=True)
    traceback.print_exc()

SYSTEM_INFO = {
    "python": sys.version,
    "import_errors": errors,
    "model_loaded": MODEL is not None,
    "load_error": load_error,
}

print(f"System info: {SYSTEM_INFO}", flush=True)
print("=== STARTING RUNPOD HANDLER ===", flush=True)


def handler(job):
    inp = job["input"]
    text = inp.get("text", "no text")

    yield {
        "status": "alive",
        "system_info": SYSTEM_INFO,
        "echo": text,
        "model_ready": MODEL is not None,
    }

    if MODEL is not None and text:
        try:
            print(f"Generating: {text}", flush=True)
            wav = MODEL.generate(text=text)
            duration = wav.shape[-1] / MODEL.sr
            yield {
                "generated": True,
                "duration": round(duration, 3),
                "samples": wav.shape[-1],
            }
        except Exception as e:
            yield {"error": f"Generation failed: {e}"}


runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
