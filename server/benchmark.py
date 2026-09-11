import argparse
import time
import wave

from server.asr import SAMPLE_RATE, create_asr_engine


def pcm_from_wav(path: str) -> bytes:
    with wave.open(path, "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != SAMPLE_RATE:
            raise ValueError("Le fichier doit etre un WAV mono PCM16 a 16 kHz.")
        return audio.readframes(audio.getnframes())


def benchmark(path: str, models: list[str]) -> None:
    pcm = pcm_from_wav(path)
    duration = len(pcm) / (SAMPLE_RATE * 2)
    for model in models:
        engine = create_asr_engine("mlx", model)
        engine.warmup()
        started = time.perf_counter()
        result = engine.transcribe(pcm)
        elapsed = time.perf_counter() - started
        print(
            f"model={model} audio={duration:.2f}s processing={elapsed:.2f}s "
            f"RTF={elapsed / duration:.2f} language={result.language}\n{result.text}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare local MLX ASR models on one WAV file.")
    parser.add_argument("audio", help="WAV mono PCM16 16 kHz")
    parser.add_argument("--models", default="large-v3-turbo,large-v3,qwen3-asr-1.7b", help="Comma-separated ASR model IDs")
    args = parser.parse_args()
    benchmark(args.audio, args.models.split(","))
