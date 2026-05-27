import runpod
import torchaudio
import os
import tempfile
import base64
import torch
from chatterbox.tts import ChatterboxTTS
from faster_whisper import WhisperModel

tts_model = None
stt_model = None

def handler(event):
    input_data = event["input"]
    mode = input_data.get("mode", "tts")

    if mode == "tts":
        return handle_tts(input_data)
    elif mode == "stt":
        return handle_stt(input_data)
    else:
        return {"error": f"Unknown mode: {mode}. Use 'tts' or 'stt'."}


def handle_tts(input_data):
    prompt = input_data.get("prompt")
    audio_ref_b64 = input_data.get("audio_ref")
    exaggeration = input_data.get("exaggeration", 0.5)
    cfg_weight = input_data.get("cfg_weight", 0.5)

    if not prompt:
        return {"error": "Missing 'prompt'"}
    if not audio_ref_b64:
        return {"error": "Missing 'audio_ref' (base64 WAV of reference voice)"}

    print(f"TTS request. Prompt: {prompt[:80]}...")

    try:
        ref_path = save_base64_audio(audio_ref_b64, suffix=".wav")

        audio_tensor = tts_model.generate(
            prompt,
            audio_prompt_path=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        audio_b64 = tensor_to_base64(audio_tensor, tts_model.sr)

        os.unlink(ref_path)

        return {
            "status": "success",
            "audio_base64": audio_b64,
            "sample_rate": tts_model.sr,
        }

    except Exception as e:
        print(f"TTS error: {e}")
        return {"error": str(e)}


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


def tensor_to_base64(audio_tensor, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, audio_tensor, sample_rate)
        with open(tmp.name, "rb") as f:
            data = f.read()
        os.unlink(tmp.name)
        return base64.b64encode(data).decode("utf-8")


def initialize_models():
    global tts_model, stt_model

    print("Loading Chatterbox TTS...")
    tts_model = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"Chatterbox loaded. Sample rate: {tts_model.sr}")

    print("Loading Whisper Large V3...")
    stt_model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )
    print("Whisper loaded.")


if __name__ == "__main__":
    initialize_models()
    runpod.serverless.start({"handler": handler})
