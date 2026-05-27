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
COPY rp_handler.py /

# Pre-download model weights only (no CUDA needed)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox', local_dir='/root/.cache/huggingface/hub/chatterbox')"
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3')"

CMD ["python3", "-u", "rp_handler.py"]
