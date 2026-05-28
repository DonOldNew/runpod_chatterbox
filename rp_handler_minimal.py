"""Minimal handler — no model, no imports. Just echo."""
import runpod
import sys
import os

print(f"MINIMAL HANDLER | Python {sys.version}", flush=True)

def handler(job):
    text = job["input"].get("text", "empty")
    yield {"echo": text, "status": "alive", "python": sys.version}

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
