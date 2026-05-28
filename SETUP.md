# AI Phone System — Global Advance HR

## Architecture

```
User (WhatsApp) → n8n Webhook → Call Engine (localhost:5050)
                                    ├── Groq Whisper STT (0.4s)
                                    ├── DeepSeek V4 Pro LLM (1.9s)
                                    └── Chatterbox Turbo TTS (5s, FREE)
                                ← Audio URL returned to WhatsApp
```

## Monthly Cost: ~$7 (800 calls)

| Component | Service | Cost |
|-----------|---------|------|
| STT | Groq Whisper Large V3 Turbo | ~$5/mo |
| LLM | DeepSeek V4 Pro (thinking OFF) | ~$2/mo |
| TTS | Chatterbox Turbo (RunPod Public) | $0/mo |

## Quick Start

```bash
# 1. Set up API keys in .env
cp .env.example .env
# Edit .env with your keys

# 2. Install dependencies
pip install requests python-dotenv

# 3. Start the call engine
./start_engine.sh

# 4. Test a turn
curl -X POST http://localhost:5050/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "text": "Hola, necesito ayuda con mis vacaciones"}'

# 5. Import n8n workflow
# Import n8n_workflow_ai_calls.json into your n8n instance
```

## API Keys Required

| Key | Where to get it |
|-----|----------------|
| RUNPOD_API_KEY | https://runpod.io/console/user/settings |
| GROQ_API_KEY | https://console.groq.com/keys |
| DEEPSEEK_API_KEY | https://platform.deepseek.com/api_keys |

## Voice Cloning

```bash
# Test with a reference voice URL
python test_voice_clone.py --ref-url https://your-url/carolina.wav

# Full test (5 phrases)
python test_voice_clone.py --ref-url URL --full-test

# Enable in production: set VOICE_CLONE_URL in .env
```

## Files

| File | Purpose |
|------|---------|
| call_engine.py | Main engine — HTTP server for n8n |
| test_pipeline.py | End-to-end pipeline test |
| test_voice_clone.py | Voice cloning test |
| test_chatterbox_turbo.py | TTS-only test |
| test_groq_whisper.py | STT-only test |
| test_endpoint.py | RunPod fallback STT test |
| rp_handler.py | RunPod serverless handler (Whisper fallback) |
| start_engine.sh | Start script |
| n8n_workflow_ai_calls.json | n8n workflow blueprint |
