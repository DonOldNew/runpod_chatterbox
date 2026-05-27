FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg

RUN python -m pip install --no-deps chatterbox-tts

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# Pre-download models into cache (no GPU needed, just downloading files)
# Chatterbox (~5GB) - uses default HF cache so from_pretrained() finds it
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox'); print('Chatterbox weights cached')"
# Whisper Large V3 (~3GB) - faster-whisper can download on CPU
RUN python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('large-v3', device='cpu', compute_type='int8'); del m; print('Whisper model cached')"

COPY rp_handler.py /

CMD ["python3", "-u", "rp_handler.py"]
