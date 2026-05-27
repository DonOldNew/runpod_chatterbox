"""
RunPod Serverless Handler — Whisper STT only.
TTS is handled by RunPod's free Chatterbox Turbo public endpoint.
"""
import runpod
import os
import tempfile
import base64
import torch
from faster_whisper import WhisperModel

stt_model = None


def handler(event):
    input_data = event["input"]
    mode = input_data.get("mode", "stt")

    if mode == "stt":
        return handle_stt(input_data)
    else:
        return {"error": f"Unknown mode: {mode}. This endpoint only supports 'stt'. Use Chatterbox Turbo public endpoint for TTS."}


def handle_stt(input_data):
    audio_b64 = input_data.get("audio")
    language = input_data.get("language", "es")

    if not audio_b64:
        return {"error": "Missing 'audio' (base64 audio data)"}

    print(f"STT request. Language: {language}")

    try:
        audio_path = save_base64_audio(audio_b64, suffix=".wav")

        segments, info = stt_model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(segment.text.strip() for segment in segments)

        os.unlink(audio_path)

        return {
            "status": "success",
            "text": text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }

    except Exception as e:
        print(f"STT error: {e}")
        return {"error": str(e)}


def save_base64_audio(b64_data, suffix=".wav"):
    audio_bytes = base64.b64decode(b64_data)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def initialize_models():
    global stt_model

    try:
        print("Loading Whisper Large V3...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        stt_model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )
        print("Whisper loaded.")
    except Exception as e:
        print(f"FATAL: Whisper load failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    initialize_models()
    runpod.serverless.start({"handler": handler})
