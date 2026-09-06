import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

try:
    from server.asr import ASR_MODELS, WhisperMLXEngine
    from server.streaming import FRAME_BYTES, AudioJob, VoiceSegmenter, trim_overlap
except ModuleNotFoundError:  # Supports `python server/app.py --check` too.
    from asr import ASR_MODELS, WhisperMLXEngine
    from streaming import FRAME_BYTES, AudioJob, VoiceSegmenter, trim_overlap

LOG = logging.getLogger("parole")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

LANGUAGES = {
    "fr": "French", "en": "English", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "ru": "Russian",
}
ASR_BACKEND = os.getenv("ASR_BACKEND", "mlx")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "translategemma:4b")
OLLAMA_QUALITY_MODEL = os.getenv("OLLAMA_QUALITY_MODEL", "translategemma:12b")
OLLAMA_QWEN_MLX_MODEL = os.getenv("OLLAMA_QWEN_MLX_MODEL", "qwen3.5:0.8b-mlx")
OLLAMA_QWEN_MODEL = os.getenv("OLLAMA_QWEN_MODEL", "qwen3:0.6b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TRANSLATION_CONCURRENCY = int(os.getenv("TRANSLATION_CONCURRENCY", "1"))
TRANSLATION_PRESETS = {
    "fast": {"label": "Fast", "model": OLLAMA_MODEL},
    "quality": {"label": "Quality", "model": OLLAMA_QUALITY_MODEL},
    "qwen-mlx": {"label": "Qwen 3.5 MLX", "model": OLLAMA_QWEN_MLX_MODEL},
    "qwen": {"label": "Qwen 3", "model": OLLAMA_QWEN_MODEL},
}


class TranslationError(Exception):
    pass


def translation_targets(source: str, targets: list[str]) -> list[str]:
    return [target for target in targets if target != source]


def translation_prompt(text: str, source: str, target: str) -> str:
    japanese_instruction = " Use Japanese script, never romanization." if target == "ja" else ""
    return (
        f"Translate the following text from {LANGUAGES.get(source, source)} to {LANGUAGES[target]}. "
        "Return only the translated text. Preserve meaning, tone, names, numbers, and punctuation. "
        "Do not explain, comment, add Markdown, or add a preamble."
        f"{japanese_instruction}\n\nText:\n{text}"
    )


def transcript_message(job: AudioJob, text: str, language: str) -> dict:
    return {
        "type": "transcript.final" if job.is_final else "transcript.partial",
        "segmentId": job.segment_id,
        "text": text,
        "language": language,
        "isFinal": job.is_final,
    }


def translation_message(segment_id: str, source: str, target: str, text: str) -> dict:
    return {
        "type": "translation",
        "segmentId": segment_id,
        "sourceLanguage": source,
        "targetLanguage": target,
        "text": text,
        "isFinal": True,
    }


async def ollama_models(client: httpx.AsyncClient) -> set[str]:
    try:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        return {model["name"] for model in response.json().get("models", [])}
    except httpx.HTTPError as error:
        raise TranslationError("Ollama est inaccessible. Lancez `ollama serve`.") from error


async def translate(client: httpx.AsyncClient, text: str, source: str, target: str, model: str) -> tuple[str, dict]:
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": translation_prompt(text, source, target),
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()
        payload = response.json()
        translated = payload.get("response", "").strip()
    except httpx.ConnectError as error:
        raise TranslationError("Ollama est inaccessible. Lancez `ollama serve`.") from error
    except httpx.TimeoutException as error:
        raise TranslationError("La traduction Ollama a expire.") from error
    except httpx.HTTPStatusError as error:
        raise TranslationError(f"Le modele Ollama `{model}` est absent ou indisponible.") from error
    except httpx.HTTPError as error:
        raise TranslationError("La traduction Ollama a echoue.") from error
    if not translated:
        raise TranslationError("Ollama a retourne une traduction vide.")
    return translated, payload


async def select_asr(app: FastAPI, model_id: str) -> None:
    if ASR_BACKEND != "mlx":
        raise ValueError("Seul ASR_BACKEND=mlx est disponible dans cette version.")
    async with app.state.asr_lock:
        if app.state.asr.model_id == model_id:
            return
        engine = WhisperMLXEngine(model_id)
        await asyncio.to_thread(engine.warmup)
        app.state.asr = engine


async def transcribe_job(app: FastAPI, job: AudioJob):
    async with app.state.asr_lock:
        return await asyncio.to_thread(app.state.asr.transcribe, job.pcm)


class LiveSession:
    def __init__(self, app: FastAPI, websocket: WebSocket):
        self.app = app
        self.websocket = websocket
        self.segmenter = VoiceSegmenter()
        self.targets: list[str] = []
        self.translation_model = OLLAMA_MODEL
        self.jobs: asyncio.Queue[AudioJob] = asyncio.Queue(maxsize=3)
        self.send_lock = asyncio.Lock()
        self.previous_final = ""
        self.translation_tasks: set[asyncio.Task] = set()

    async def send(self, payload: dict) -> None:
        async with self.send_lock:
            await self.websocket.send_json(payload)

    def track_translation(self, task: asyncio.Task) -> None:
        self.translation_tasks.add(task)
        task.add_done_callback(self.translation_tasks.discard)

    async def configure(self, command: dict) -> None:
        requested_model = command.get("model", WHISPER_MODEL)
        preset = command.get("translationPreset", "fast")
        if preset not in TRANSLATION_PRESETS:
            raise ValueError("Preset de traduction non pris en charge.")
        self.targets = [code for code in command.get("targets", []) if code in LANGUAGES]
        await select_asr(self.app, requested_model)
        self.translation_model = TRANSLATION_PRESETS[preset]["model"]
        await self.send({
            "type": "session.ready",
            "asrBackend": ASR_BACKEND,
            "asrModel": requested_model,
            "translationModel": self.translation_model,
        })

    async def enqueue(self, job: AudioJob) -> None:
        if job.is_final:
            await self.jobs.put(job)  # Bounded backpressure beats an unbounded audio queue.
        elif not self.jobs.full():
            self.jobs.put_nowait(job)

    async def worker(self) -> None:
        while True:
            job = await self.jobs.get()
            try:
                started = time.perf_counter()
                result = await transcribe_job(self.app, job)
                elapsed = time.perf_counter() - started
                duration = len(job.pcm) / 32_000
                rtf = elapsed / duration if duration else 0
                text = result.text
                if job.is_final and job.overlap:
                    text = trim_overlap(self.previous_final, text)
                if not text:
                    continue
                if job.is_final:
                    self.previous_final = text
                LOG.info(
                    "ASR segment=%s audio=%.2fs processing=%.2fs RTF=%.2f language=%s final=%s",
                    job.segment_id, duration, elapsed, rtf, result.language, job.is_final,
                )
                await self.send(transcript_message(job, text, result.language))
                if job.is_final:
                    self.track_translation(asyncio.create_task(self.translate_final(job, text, result.language)))
            except Exception as error:
                LOG.exception("ASR failed")
                await self.send({"type": "error", "scope": "asr", "message": f"Transcription impossible : {error}"})
            finally:
                self.jobs.task_done()

    async def translate_final(self, job: AudioJob, text: str, source: str) -> None:
        async def one(target: str) -> None:
            try:
                queued_at = time.perf_counter()
                async with self.app.state.translation_semaphore:
                    started = time.perf_counter()
                    translated, stats = await translate(self.app.state.ollama, text, source, target, self.translation_model)
                duration = time.perf_counter() - started
                queue_wait = started - queued_at
                LOG.info(
                    "Translation segment=%s target=%s model=%s queue=%.2fs request=%.2fs "
                    "ollama_total=%.2fs load=%.2fs prompt=%.2fs generation=%.2fs tokens=%s/%s",
                    job.segment_id, target, self.translation_model, queue_wait, duration,
                    stats.get("total_duration", 0) / 1e9, stats.get("load_duration", 0) / 1e9,
                    stats.get("prompt_eval_duration", 0) / 1e9, stats.get("eval_duration", 0) / 1e9,
                    stats.get("prompt_eval_count", 0), stats.get("eval_count", 0),
                )
                await self.send(translation_message(job.segment_id, source, target, translated))
                LOG.info("End-to-end segment=%s target=%s latency=%.2fs", job.segment_id, target, time.perf_counter() - job.created_at)
            except TranslationError as error:
                await self.send({"type": "error", "scope": "translation", "segmentId": job.segment_id, "message": str(error)})

        await asyncio.gather(*(one(target) for target in translation_targets(source, self.targets)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ASR_BACKEND != "mlx":
        raise RuntimeError("ASR_BACKEND doit etre `mlx` sur Apple Silicon.")
    app.state.asr_lock = asyncio.Lock()
    app.state.asr = WhisperMLXEngine(WHISPER_MODEL)
    app.state.translation_semaphore = asyncio.Semaphore(TRANSLATION_CONCURRENCY)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=3)) as client:
        app.state.ollama = client
        started = time.perf_counter()
        await asyncio.to_thread(app.state.asr.warmup)
        LOG.info("ASR backend=mlx model=%s warmed in %.2fs", WHISPER_MODEL, time.perf_counter() - started)
        try:
            installed = await ollama_models(client)
            if OLLAMA_MODEL not in installed:
                LOG.warning("Ollama model missing: run `ollama pull %s`", OLLAMA_MODEL)
        except TranslationError as error:
            LOG.warning("%s", error)
        yield


app = FastAPI(title="Parole Locale", lifespan=lifespan)


@app.get("/api/health")
async def health():
    try:
        installed = await ollama_models(app.state.ollama)
        ollama = {preset: details["model"] in installed for preset, details in TRANSLATION_PRESETS.items()}
    except TranslationError:
        ollama = {preset: False for preset in TRANSLATION_PRESETS}
    return {"asrBackend": ASR_BACKEND, "asrModel": app.state.asr.model_id, "ollama": ollama}


@app.get("/api/models")
async def models():
    return {
        "default": WHISPER_MODEL,
        "models": [{key: value for key, value in model.items() if key != "repository"} for model in ASR_MODELS],
        "translationPresets": [{"id": key, **value} for key, value in TRANSLATION_PRESETS.items()],
    }


@app.websocket("/ws/transcribe")
async def transcribe_socket(websocket: WebSocket):
    await websocket.accept()
    session = LiveSession(app, websocket)
    worker = asyncio.create_task(session.worker())
    try:
        while True:
            message = await websocket.receive()
            if text := message.get("text"):
                try:
                    command = json.loads(text)
                    if command.get("type") == "configure":
                        await session.configure(command)
                except (json.JSONDecodeError, ValueError) as error:
                    await session.send({"type": "error", "scope": "configuration", "message": str(error)})
                continue
            pcm = message.get("bytes")
            if not pcm or not session.targets:
                continue
            if len(pcm) > FRAME_BYTES * 5:
                await session.send({"type": "error", "scope": "audio", "message": "Paquet audio trop grand."})
                continue
            for job in session.segmenter.feed(pcm):
                await session.enqueue(job)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        worker.cancel()
        for task in tuple(session.translation_tasks):
            task.cancel()


dist = Path(__file__).parent.parent / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


def self_check():
    assert translation_targets("fr", ["fr", "en", "ja"]) == ["en", "ja"]
    assert "Return only the translated text" in translation_prompt("Bonjour", "fr", "en")
    assert trim_overlap("I want to go to", "to Japan tomorrow") == "Japan tomorrow"


if __name__ == "__main__" and "--check" in sys.argv:
    self_check()
    print("ok")
