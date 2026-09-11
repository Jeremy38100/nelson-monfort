# NelsonMonfort

Local web application for automatic speech recognition (ASR/STT) and text translation. It does not perform TTS: it transcribes microphone audio and translates finalized text.

All inference runs locally: on Apple Silicon Mac with MLX/Metal, and on Windows/Linux with CUDA via `faster-whisper`. Ollama runs the translation model. No API key, paid cloud service, or remote audio processing is required.

## Requirements

- **macOS** (Apple Silicon, macOS 14+) OR **Windows 10/11** (NVIDIA GPU recommended) / Linux.
- Python 3.10+.
- Node.js 20+.
- [Ollama](https://ollama.com/download).

No additional system audio package is needed. The browser sends mono 16 kHz PCM directly to the backend.

## Installation

From a clean clone:

**macOS / Linux:**
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
npm ci
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r server/requirements.txt
npm install
```

Pull Ollama models:
```sh
ollama pull qwen3:0.6b
# Optional:
ollama pull translategemma:4b

# MLX Model (MacOS only)
ollama pull qwen3.5:0.8b-mlx
```

The default ASR model is `large-v3-turbo` (`mlx-community/whisper-large-v3-turbo` on macOS MLX, or `Systran/faster-whisper-large-v3-turbo` on Windows/CUDA). It is downloaded locally on the first backend start and stored in the Hugging Face cache. This public model does not require a Hugging Face token.

## Models

### ASR

The UI provides two MLX ASR models that support French, English, and Japanese:

| Model | Use |
| --- | --- |
| `small` | Benchmarking or minimum ASR latency |
| `large-v3-turbo` | Default, best multilingual quality |

Whisper detects the source language independently for each finalized segment. No session-wide source language is forced, so a speaker can switch between French, English, and Japanese.

### Translation

| Preset | Default model | Use |
| --- | --- | --- |
| `Fast` | `translategemma:4b` | Default translation quality and latency balance |
| `Quality` | `translategemma:12b` | Higher quality, more memory use and latency |
| `Qwen 3.5 MLX` | `qwen3.5:0.8b-mlx` | Lowest latency, lower translation quality |
| `Qwen 3` | `qwen3:0.6b` | Lowest latency, lower translation quality |

Translation only starts for a finalized ASR segment. The detected source language is displayed directly, so no unnecessary `fr -> fr`, `en -> en`, or `ja -> ja` Ollama request is made. Ollama thinking is explicitly disabled and models stay loaded for 30 minutes after use.

## Development

Start Ollama if it is not already running:

```sh
ollama serve
```

In one terminal:

```sh
source .venv/bin/activate
uvicorn server.app:app --reload
```

In another terminal:

```sh
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`, and grant microphone access. The Vite proxy forwards `/ws/transcribe` to FastAPI.

## Production Build

```sh
npm run build
source .venv/bin/activate
uvicorn server.app:app
```

The backend then serves `dist/` at `http://127.0.0.1:8000`.

## Configuration

The defaults are tuned automatically for Apple Silicon Mac (`mlx`) or Windows/Linux (`faster-whisper`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASR_BACKEND` | `auto` (`mlx` on macOS, `faster-whisper` on Windows/Linux) | Local ASR backend. |
| `WHISPER_MODEL` | `large-v3-turbo` | ASR model loaded at backend startup. |
| `OLLAMA_MODEL` | `translategemma:4b` | `Fast` translation preset model. |
| `OLLAMA_QUALITY_MODEL` | `translategemma:12b` | `Quality` translation preset model. |
| `OLLAMA_QWEN_MLX_MODEL` | `qwen3.5:0.8b-mlx` | `Qwen 3.5 MLX` preset model. |
| `OLLAMA_QWEN_MODEL` | `qwen3:0.6b` | `Qwen 3` preset model. |
| `TRANSLATION_CONCURRENCY` | `1` | Concurrent Ollama requests. Keep `1` to avoid contention on one local model. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama server URL. |
| `LOG_LEVEL` | `INFO` | Latency log level. |

Example:

```sh
WHISPER_MODEL=small OLLAMA_MODEL=translategemma:4b uvicorn server.app:app --reload
```

## Architecture

```text
Microphone getUserMedia
  -> AudioWorklet (mono PCM 16 kHz, RMS noise gate)
  -> WebSocket
  -> WebRTC VAD (20 ms frames, 500 ms end-of-utterance pause)
  -> Whisper Large v3 Turbo / MLX / Metal
  -> transcript.partial or transcript.final
  -> TranslateGemma or Qwen / Ollama for finals only
  -> WebSocket
  -> UI
```

The live UI has one fixed block per selected target language. Each block keeps its language order, concatenates recent text, scrolls to the latest content, and retains the configured number of finalized segments. The live footer exposes the microphone RMS level and an adjustable noise threshold. Audio below that threshold is sent as silence so VAD can finish speech cleanly instead of transcribing low-level noise.

The VAD keeps 300 ms of pre-roll. Continuous speech is capped at eight seconds and preserves a short overlap with exact deduplication, limiting latency without cutting words. Each client ASR queue is bounded to three jobs: excess partials are dropped, while finals apply backpressure instead of growing memory without limit.

## WebSocket Messages

```json
{"type":"transcript.partial","segmentId":"3","text":"I want to go","language":"en","isFinal":false}
{"type":"transcript.final","segmentId":"3","text":"I want to go to Japan.","language":"en","isFinal":true}
{"type":"translation","segmentId":"3","sourceLanguage":"en","targetLanguage":"ja","text":"日本に行きたいです。","isFinal":true}
```

Translations use the same `segmentId`, so the frontend updates a segment instead of adding a new line for each partial.

## Metal And Latency

Verify MLX/Metal:

```sh
source .venv/bin/activate
python -c "import mlx.core as mx; print('Metal:', mx.metal.is_available()); print('Device:', mx.default_device())"
```

The expected result is `Metal: True` and a `gpu` device. The backend logs, for example:

```text
ASR backend=mlx model=large-v3-turbo warmed in 4.21s
ASR segment=3 audio=2.30s processing=0.31s RTF=0.13 language=ja final=True
Translation segment=3 target=fr model=qwen3:0.6b queue=0.00s request=0.06s ollama_total=0.06s load=0.00s prompt=0.02s generation=0.03s tokens=24/7
End-to-end segment=3 target=fr latency=0.61s
```

`RTF` is processing time divided by audio duration; below `1` means ASR is faster than real time. Translation logs separate semaphore wait time (`queue`), HTTP time (`request`), model loading (`load`), prompt evaluation (`prompt`), and generation (`generation`). They identify whether the bottleneck is ASR, a cold Ollama model, token generation, or excess concurrent work.

## ASR Benchmark

The benchmark compares both ASR models on the same mono PCM16 16 kHz WAV file:

```sh
source .venv/bin/activate
python -m server.benchmark path/to/16khz-mono-sample.wav
```

It reports audio duration, transcription duration, RTF, and detected language. The `ASREngine` abstraction in `server/asr.py` isolates `WhisperMLXEngine`, so a future `Qwen3ASREngine` can reuse the VAD/WebSocket pipeline without changing the rest of the backend.

## Checks

```sh
source .venv/bin/activate
python server/app.py --check
python -m unittest server.test_pipeline
npm run build
```

The tests cover multiple targets, skipping translation into the source language, changing source language between segments, partial/final message shapes, VAD, deduplication, ASR model configuration, and Ollama failures.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| Microphone is denied | Grant the browser microphone permission, then reload the page. |
| Ollama is unavailable | Install Ollama, then run `ollama serve`. |
| An Ollama model is missing | Run the matching `ollama pull` command from Installation. |
| Whisper model is missing | Let the first backend startup finish downloading the MLX model; check the initial network connection and free disk space. |
| MLX or Metal is unavailable | Use native ARM64 Python on Apple Silicon, then run the Metal verification command above. |
| WebSocket is unreachable | Check that Uvicorn is running on port 8000 and Vite was launched from this project, or use the production build on port 8000. |
