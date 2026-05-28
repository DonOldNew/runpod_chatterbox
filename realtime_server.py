"""
Real-time AI Phone Call Server — Pipecat + SmallWebRTC.

Employees click a link → browser opens → real-time voice call with Carolina (AI HR).

Architecture:
  Browser (WebRTC) ←→ FastAPI + SmallWebRTC ←→ Pipecat Pipeline
    ├─ STT: Groq Whisper Large V3 Turbo (0.4s)
    ├─ LLM: DeepSeek V4 Pro (1.9s, thinking OFF)
    └─ TTS: Chatterbox Turbo (5-70s batch — WILL be replaced with streaming)

Usage:
  source venv/bin/activate
  python realtime_server.py [--port 7860]

  Then open http://localhost:7860/client in browser.
"""
import argparse
import asyncio
import os
import sys
import json
import base64
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TTSAudioRawFrame,
    TextFrame,
    EndFrame,
    LLMRunFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
VOICE_CLONE_URL = os.getenv("VOICE_CLONE_URL", "")

# Load persona
PERSONA_FILE = Path(__file__).parent / "persona.txt"
SYSTEM_PROMPT = PERSONA_FILE.read_text(encoding="utf-8").strip() if PERSONA_FILE.exists() else "Eres Carolina de Recursos Humanos."


# ──────────────────────────────────────────────
# Custom TTS: Chatterbox Turbo (non-streaming)
# ──────────────────────────────────────────────

class ChatterboxTurboTTS(FrameProcessor):
    """Wraps Chatterbox Turbo public endpoint as a Pipecat TTS processor.

    LIMITATION: This is batch (non-streaming) TTS with 5-70s latency.
    For production, replace with streaming Chatterbox or Fish Speech.
    """

    def __init__(self, api_key: str, voice_url: str = "", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.voice_url = voice_url
        self._base_url = "https://api.runpod.ai/v2/chatterbox-turbo"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text and frame.text.strip():
            text = frame.text.strip()
            logger.info(f"[TTS] Generating: '{text[:60]}...'")
            pcm_data = await self._generate(text)

            if pcm_data:
                audio_frame = TTSAudioRawFrame(
                    audio=pcm_data,
                    sample_rate=24000,
                    num_channels=1,
                )
                await self.push_frame(audio_frame)
                logger.info(f"[TTS] Audio sent ({len(pcm_data):,} bytes)")
            else:
                logger.error("[TTS] Generation failed")
        else:
            await self.push_frame(frame, direction)

    async def _generate(self, text: str) -> bytes | None:
        payload = {"input": {"prompt": text, "format": "wav"}}
        if self.voice_url:
            payload["input"]["voice_url"] = self.voice_url
        else:
            payload["input"]["voice"] = "lucy"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/run", json=payload, headers=self._headers
                ) as resp:
                    data = await resp.json()

                if "id" not in data:
                    logger.error(f"[TTS] Submit error: {data}")
                    return None

                job_id = data["id"]

                for _ in range(60):
                    async with session.get(
                        f"{self._base_url}/status/{job_id}", headers=self._headers
                    ) as resp:
                        status_data = await resp.json()

                    status = status_data.get("status")
                    if status == "COMPLETED":
                        output = status_data.get("output", {})
                        if isinstance(output, dict) and "audio_url" in output:
                            async with session.get(output["audio_url"]) as audio_resp:
                                wav_data = await audio_resp.read()
                                return wav_data[44:] if len(wav_data) > 44 else wav_data
                        return None
                    elif status == "FAILED":
                        logger.error(f"[TTS] Failed: {status_data}")
                        return None

                    await asyncio.sleep(2)

                logger.error("[TTS] Timeout")
                return None
        except Exception as e:
            logger.error(f"[TTS] Error: {e}")
            return None


# ──────────────────────────────────────────────
# Bot pipeline (one per call)
# ──────────────────────────────────────────────

async def run_bot(webrtc_connection):
    """Run the AI HR assistant pipeline for one WebRTC call."""
    logger.info("=== New call connected ===")

    # Transport
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    # STT — Groq Whisper
    stt = GroqSTTService(
        api_key=GROQ_API_KEY,
        model="whisper-large-v3-turbo",
        language="es",
    )

    # LLM — DeepSeek V4 Pro
    llm = DeepSeekLLMService(
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-v4-pro",
        params=DeepSeekLLMService.InputParams(
            temperature=0.7,
            max_tokens=150,
            extra={"thinking": {"type": "disabled"}},
        ),
    )

    # TTS — Chatterbox Turbo
    tts = ChatterboxTurboTTS(
        api_key=RUNPOD_API_KEY,
        voice_url=VOICE_CLONE_URL,
    )

    # Conversation context
    context = LLMContext(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
    )

    # Context aggregators (handle multi-turn conversation)
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Pipeline: audio in → STT → context → LLM → TTS → audio out
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info(f"Client connected: {client}")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info("=== Call ended ===")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

small_webrtc_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount prebuilt WebRTC UI
app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/")
async def root():
    """Redirect to call UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/client")


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC SDP offer from browser."""
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, connection)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handle ICE candidate trickle."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "services": {
            "stt": "groq-whisper" if GROQ_API_KEY else "missing",
            "llm": "deepseek-v4-pro" if DEEPSEEK_API_KEY else "missing",
            "tts": f"chatterbox-turbo{'(clone)' if VOICE_CLONE_URL else '(lucy)'}",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Phone Call Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    logger.info(f"Starting AI Call Server on http://{args.host}:{args.port}")
    logger.info(f"  Open http://{args.host}:{args.port}/client to start a call")
    logger.info(f"  STT: Groq Whisper | LLM: DeepSeek V4 Pro | TTS: Chatterbox Turbo")

    uvicorn.run(app, host=args.host, port=args.port)
