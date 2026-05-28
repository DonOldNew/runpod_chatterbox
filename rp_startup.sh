#!/bin/bash
# ──────────────────────────────────────────────
# Runtime startup: install chatterbox + start handler
# ──────────────────────────────────────────────
set -e

echo "=== STARTUP: Installing chatterbox-streaming ==="

# Uninstall old torch, install 2.6.0 compatible version
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install chatterbox-streaming and deps
pip install --no-cache-dir --no-deps "chatterbox-streaming"
pip install --no-cache-dir \
    "numpy~=1.26.0" \
    "resampy==0.4.3" \
    "librosa==0.10.0" \
    "s3tokenizer" \
    "transformers==4.46.3" \
    "diffusers==0.29.0" \
    "resemble-perth==1.0.1" \
    "omegaconf==2.3.0" \
    "conformer==0.3.2" \
    "huggingface_hub"

echo "=== STARTUP: Packages installed, starting handler ==="

# Start the handler (model loads inside Python)
exec python -u /app/rp_handler_streaming.py
