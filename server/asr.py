from abc import ABC, abstractmethod
from dataclasses import dataclass

import mlx_whisper
import numpy as np

SAMPLE_RATE = 16_000
ASR_MODELS = [
    {"id": "small", "label": "Whisper Small", "repository": "mlx-community/whisper-small", "description": "Plus rapide"},
    {"id": "large-v3-turbo", "label": "Whisper Large v3 Turbo", "repository": "mlx-community/whisper-large-v3-turbo", "description": "Meilleure qualite multilingue"},
]
ASR_MODEL_IDS = {model["id"] for model in ASR_MODELS}


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


class WhisperMLXEngine(ASREngine):
    def __init__(self, model_id: str):
        if model_id not in ASR_MODEL_IDS:
            raise ValueError(f"Modele ASR non pris en charge : {model_id}")
        self.model_id = model_id
        self.repository = next(model["repository"] for model in ASR_MODELS if model["id"] == model_id)

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
            condition_on_previous_text=True,
            temperature=0,
        )
        return ASRResult(text=result["text"].strip(), language=result["language"])
