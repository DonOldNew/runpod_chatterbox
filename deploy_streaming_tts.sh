#!/bin/bash
# ──────────────────────────────────────────────
# Deploy Chatterbox Streaming TTS to RunPod Serverless
# ──────────────────────────────────────────────
#
# Prerequisites:
#   1. Docker installed and running
#   2. Docker Hub account (or GHCR)
#   3. RunPod account with credits (~$10-20/month)
#
# Usage:
#   ./deploy_streaming_tts.sh <docker-hub-username>
#
# After deploy:
#   1. Go to runpod.io/console/serverless → Endpoints → your new endpoint
#   2. Copy the Endpoint ID
#   3. Add to .env: RUNPOD_STREAMING_ENDPOINT=<endpoint-id>
#   4. Restart realtime_server.py
# ──────────────────────────────────────────────

set -e

DOCKER_USER="${1:-}"
IMAGE_NAME="chatterbox-streaming"
TAG="latest"

if [ -z "$DOCKER_USER" ]; then
    echo "Usage: ./deploy_streaming_tts.sh <docker-hub-username>"
    echo ""
    echo "Example: ./deploy_streaming_tts.sh donoldnew"
    echo ""
    echo "This will:"
    echo "  1. Build Docker image with Chatterbox Streaming TTS"
    echo "  2. Push to Docker Hub as <username>/chatterbox-streaming:latest"
    echo "  3. Print instructions for RunPod Serverless setup"
    echo ""
    echo "Requirements: Docker running, Docker Hub login"
    exit 1
fi

FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

echo "═══════════════════════════════════════════════"
echo " Chatterbox Streaming TTS — RunPod Deploy"
echo "═══════════════════════════════════════════════"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Step 1: Build
echo "▸ Step 1/3: Building Docker image..."
echo "  (This downloads ~1.5 GB of model weights — first build takes 10-15 min)"
echo "  (Building for linux/amd64 — RunPod runs x86 GPUs)"
echo ""
docker build --platform linux/amd64 -f Dockerfile.streaming -t "${FULL_IMAGE}" .
echo ""
echo "✓ Image built: ${FULL_IMAGE}"
echo ""

# Step 2: Push
echo "▸ Step 2/3: Pushing to Docker Hub..."
docker push "${FULL_IMAGE}"
echo ""
echo "✓ Image pushed: ${FULL_IMAGE}"
echo ""

# Step 3: Instructions
echo "▸ Step 3/3: RunPod Serverless Setup"
echo ""
echo "═══════════════════════════════════════════════"
echo " MANUELLE SCHRITTE (2 Minuten):"
echo "═══════════════════════════════════════════════"
echo ""
echo "1. Öffne: https://www.runpod.io/console/serverless"
echo ""
echo "2. Klick: 'New Endpoint'"
echo ""
echo "3. Einstellungen:"
echo "   - Name: chatterbox-streaming"
echo "   - Docker Image: ${FULL_IMAGE}"
echo "   - GPU: 24 GB (A5000, L4, oder RTX 4090)"
echo "   - Min Workers: 0 (spart Geld — kaltet Start ~3 Min)"
echo "   - Max Workers: 1 (für ~800 Calls/Monat reicht 1)"
echo "   - Idle Timeout: 60 Sekunden"
echo "   - FlashBoot: AN (wenn verfügbar)"
echo ""
echo "4. Klick: 'Create Endpoint'"
echo ""
echo "5. Kopiere die Endpoint ID (z.B. 'abc123def456')"
echo ""
echo "6. Füge in .env ein:"
echo "   RUNPOD_STREAMING_ENDPOINT=<deine-endpoint-id>"
echo ""
echo "7. Starte realtime_server.py neu"
echo ""
echo "═══════════════════════════════════════════════"
echo " Kosten: ~\$0.00076/Sekunde GPU (24GB)"
echo " Bei 800 Calls × 5 Min = ~\$10-20/Monat"
echo "═══════════════════════════════════════════════"
