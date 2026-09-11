from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import site
import sys

import numpy as np

# Set up Windows DLL search paths for CUDA if installed in site-packages
if os.name == "nt":
    for p in site.getsitepackages():
        for sub in ("cublas", "cudnn"):
            bin_p = os.path.join(p, "nvidia", sub, "bin")
            if os.path.isdir(bin_p):
                try:
                    os.add_dll_directory(bin_p)
                except Exception:
                    pass

try:
    from faster_whisper import WhisperModel
    import ctranslate2
except ImportError:
    WhisperModel = None
    ctranslate2 = None

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

try:
    from mlx_audio.stt import load as mlx_audio_load
except ImportError:
    mlx_audio_load = None

SAMPLE_RATE = 16_000
ASR_MODELS = [
    {
        "id": "small",
        "label": "Whisper Small",
        "repository": "mlx-community/whisper-small",
        "description": "Plus rapide",
        "engine": "whisper",
    },
    {
        "id": "large-v3-turbo",
        "label": "Whisper Large v3 Turbo",
        "repository": "mlx-community/whisper-large-v3-turbo",
        "description": "Meilleure qualite multilingue",
        "engine": "whisper",
    },
    {
        "id": "large-v3",
        "label": "Whisper Large v3",
        "repository": "mlx-community/whisper-large-v3-mlx",
        "description": "Plus precis, plus lent",
        "engine": "whisper",
    },
    {
        "id": "qwen3-asr-1.7b",
        "label": "Qwen3 ASR 1.7B",
        "repository": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "description": "Qualite multilingue recente",
        "engine": "qwen",
    },
]
QWEN_LANGUAGE_CODES = {
    "Arabic": "ar",
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Dutch": "nl",
}

HALLUCINATIONS = {
    "thank you",
    "thank you.",
    "thank you very much",
    "thank you very much.",
    "thanks for watching",
    "thanks for watching.",
    "thanks",
    "merci",
    "merci.",
    "merci beaucoup",
    "merci beaucoup.",
    "merci d'avoir regardé",
    "bye",
    "goodbye",
    "au revoir",
    "you",
    "sous-titrage",
    "sous-titres réalisés par",
    "transcription",
}


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str


class ASREngine(ABC):
    """Small seam for trying another local ASR engine later, such as Qwen3-ASR."""

    model_id: str

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def transcribe(self, pcm: bytes) -> ASRResult: ...


def asr_model(model_id: str) -> dict:
    model = next((model for model in ASR_MODELS if model["id"] == model_id), None)
    if model is None:
        raise ValueError(f"Modele ASR non pris en charge : {model_id}")
    return model


class WhisperFasterEngine(ASREngine):
    def __init__(self, model_id: str, device: str = None, compute_type: str = None):
        if asr_model(model_id)["engine"] != "whisper":
            raise ValueError(f"Modele ASR incompatible avec faster-whisper : {model_id}")
        if WhisperModel is None:
            raise RuntimeError("faster-whisper n'est pas installe.")
        self.model_id = model_id

        if device is None:
            if ctranslate2 and ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if compute_type is None:
            compute_type = "float16" if self.device == "cuda" else "int8"
        self.compute_type = compute_type

        self.model = WhisperModel(
            self.model_id, device=self.device, compute_type=self.compute_type
        )

    def warmup(self) -> None:
        silent_audio = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
        list(self.model.transcribe(silent_audio, language=None, beam_size=1)[0])

    def transcribe(self, pcm: bytes) -> ASRResult:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio,
            language=None,
            task="transcribe",
            beam_size=1,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=200),
        )
        text = "".join(segment.text for segment in segments).strip()
        if getattr(info, "no_speech_prob", 0) > 0.6:
            return ASRResult(text="", language=info.language or "fr")

        duration = len(pcm) / 32_000
        clean_text = text.lower().strip().rstrip(".,!?")
        if clean_text in HALLUCINATIONS and duration < 2.0:
            return ASRResult(text="", language=info.language or "fr")

        return ASRResult(text=text, language=info.language or "fr")


class WhisperMLXEngine(ASREngine):
    def __init__(self, model_id: str):
        model = asr_model(model_id)
        if model["engine"] != "whisper":
            raise ValueError(f"Modele ASR incompatible avec mlx-whisper : {model_id}")
        if mlx_whisper is None:
            raise RuntimeError("mlx-whisper n'est pas disponible sur ce systeme.")
        self.model_id = model_id
        self.repository = model["repository"]

    def warmup(self) -> None:
        # mlx-whisper owns a process-wide model cache keyed by this repository.
        # A silent sample loads the weights once without keeping an extra model copy.
        mlx_whisper.transcribe(
            np.zeros(SAMPLE_RATE // 10, dtype=np.float32),
            path_or_hf_repo=self.repository,
            language=None,
            task="transcribe",
            verbose=None,
            fp16=True,
            condition_on_previous_text=False,
        )

    def transcribe(self, pcm: bytes) -> ASRResult:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.repository,
            language=None,
            task="transcribe",
            verbose=None,
            fp16=True,
            condition_on_previous_text=False,
            temperature=0,
        )
        text = result["text"].strip()
        duration = len(pcm) / 32_000
        clean_text = text.lower().strip().rstrip(".,!?")
        if clean_text in HALLUCINATIONS and duration < 2.0:
            return ASRResult(text="", language=result.get("language") or "fr")
        return ASRResult(text=text, language=result["language"])


class QwenMLXEngine(ASREngine):
    def __init__(self, model_id: str):
        model = asr_model(model_id)
        if model["engine"] != "qwen":
            raise ValueError(f"Modele ASR incompatible avec Qwen : {model_id}")
        if mlx_audio_load is None:
            raise RuntimeError("mlx-audio n'est pas disponible sur ce systeme.")
        self.model_id = model_id
        self.repository = model["repository"]
        self.model = None

    def warmup(self) -> None:
        self.model = mlx_audio_load(self.repository)
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes())

    def transcribe(self, pcm: bytes) -> ASRResult:
        if self.model is None:
            self.model = mlx_audio_load(self.repository)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.model.generate(audio, language=None, max_tokens=256)
        language = result.language[0] if isinstance(result.language, list) else result.language
        text = result.text.strip()
        duration = len(pcm) / 32_000
        clean_text = text.lower().strip().rstrip(".,!?")
        if clean_text in HALLUCINATIONS and duration < 2.0:
            return ASRResult(text="", language=QWEN_LANGUAGE_CODES.get(str(language).title(), "fr"))
        return ASRResult(text=text, language=QWEN_LANGUAGE_CODES.get(str(language).title(), "fr"))


def create_asr_engine(backend: str, model_id: str) -> ASREngine:
    model = asr_model(model_id)
    if backend in ("faster-whisper", "faster"):
        return WhisperFasterEngine(model_id)
    elif backend == "mlx":
        return QwenMLXEngine(model_id) if model["engine"] == "qwen" else WhisperMLXEngine(model_id)
    else:
        raise ValueError(f"Backend ASR inconnu ou non pris en charge : {backend}")
