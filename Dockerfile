FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper Large V3 (~3GB) into cache
RUN python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('large-v3', device='cpu', compute_type='int8'); del m; print('Whisper model cached')"

COPY rp_handler.py /

CMD ["python3", "-u", "rp_handler.py"]
