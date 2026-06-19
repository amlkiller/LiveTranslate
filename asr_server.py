"""
Remote ASR server for LiveTranslate, using faster-whisper.

Run this on a machine with a GPU, then point LiveTranslate's "Remote Whisper"
engine at it (Settings -> VAD/ASR -> Remote ASR Server URL). The client
(asr_remote.py) POSTs raw float32 PCM (16 kHz mono) to /transcribe and gets
back the transcription as JSON.

    pip install faster-whisper fastapi uvicorn numpy
    python asr_server.py --host 0.0.0.0 --port 8765 --model large-v3 --device cuda --compute-type float16

Notes:
- For CUDA, faster-whisper/CTranslate2 needs the CUDA 12 cuBLAS and cuDNN 9
  libraries on the library path (e.g. `pip install nvidia-cublas-cu12
  nvidia-cudnn-cu12`, or a system CUDA install).
- The model is downloaded from Hugging Face on first run; set the HF_ENDPOINT
  env var to a mirror if direct access is slow.
"""

import argparse
import logging
import time

import numpy as np
from fastapi import FastAPI, Request
from faster_whisper import WhisperModel
import uvicorn

log = logging.getLogger("ASR-Server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="Remote ASR Server")
_model: WhisperModel = None


@app.on_event("startup")
def load_model():
    global _model
    args = app.state.args
    log.info(f"Loading model: {args.model} on {args.device} ({args.compute_type})")
    _model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    log.info(f"Model ready: {args.model}")


@app.post("/transcribe")
async def transcribe(request: Request):
    """Accept raw float32 PCM audio at 16kHz mono. Return transcription."""
    import struct

    request_body = await request.body()

    # Parse header: first 8 bytes = language string length (uint32) + language string
    # Then rest is audio data
    if len(request_body) < 4:
        return {"error": "empty request"}

    lang_len = struct.unpack("<I", request_body[:4])[0]
    language = request_body[4 : 4 + lang_len].decode("utf-8") if lang_len > 0 else None
    if language == "auto" or language == "":
        language = None
    audio_bytes = request_body[4 + lang_len :]

    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    duration = len(audio) / 16000

    t0 = time.time()
    segments, info = _model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    text_parts = []
    for seg in segments:
        text_parts.append(seg.text.strip())

    full_text = " ".join(text_parts).strip()
    elapsed = time.time() - t0

    log.info(
        f"Transcribed {duration:.1f}s audio in {elapsed:.2f}s: "
        f"[{info.language}] {full_text[:80]}"
    )

    if not full_text:
        return {"text": None, "language": info.language, "elapsed": elapsed}

    return {
        "text": full_text,
        "language": info.language,
        "language_name": info.language,
        "elapsed": elapsed,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": app.state.args.model}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--model", default="medium", help="Whisper model size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--compute-type", default="float16", help="Compute type")
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
