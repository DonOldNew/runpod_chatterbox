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

# Pre-download Chatterbox model
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"

# Pre-download Whisper Large V3
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='float32')"

CMD ["python3", "-u", "rp_handler.py"]
