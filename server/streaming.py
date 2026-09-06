from collections import deque
from dataclasses import dataclass
import time

import webrtcvad

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass(frozen=True)
class AudioJob:
    segment_id: str
    pcm: bytes
    is_final: bool
    overlap: bool
    created_at: float


MIN_SPEECH_FRAMES = 6  # At least 120 ms of cumulative speech required in an utterance.


class VoiceSegmenter:
    """Turns 20 ms PCM frames into short ASR jobs without unbounded buffering."""

    def __init__(self, vad=None):
        self.vad = vad or webrtcvad.Vad(3)
        self.pending = b""
        self.preroll = deque(maxlen=15)  # 300 ms protects the first phoneme.
        self.utterance = bytearray()
        self.active = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.frames_since_partial = 0
        self.segment_number = 0

    def feed(self, pcm: bytes) -> list[AudioJob]:
        self.pending += pcm
        jobs = []
        while len(self.pending) >= FRAME_BYTES:
            frame, self.pending = self.pending[:FRAME_BYTES], self.pending[FRAME_BYTES:]
            jobs.extend(self._frame(frame))
        return jobs

    def _frame(self, frame: bytes) -> list[AudioJob]:
        speech = self.vad.is_speech(frame, SAMPLE_RATE)
        jobs = []
        self.preroll.append(frame)
        if not self.active:
            if not speech:
                return jobs
            self.active = True
            self.utterance = bytearray().join(self.preroll)
            self.silence_frames = 0
            self.speech_frames = 1
            self.frames_since_partial = 0
        else:
            self.utterance.extend(frame)
            if speech:
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                self.silence_frames += 1

        self.frames_since_partial += 1
        if self.frames_since_partial >= 50:  # One partial per second at most.
            if self.speech_frames >= MIN_SPEECH_FRAMES:
                jobs.append(self._job(is_final=False, overlap=False))
            self.frames_since_partial = 0

        if self.silence_frames >= 25:  # 500 ms of silence finalizes the utterance.
            if self.speech_frames >= MIN_SPEECH_FRAMES:
                jobs.append(self._finish(overlap=False))
            else:
                self._reset()
        elif (
            len(self.utterance) >= SAMPLE_RATE * 2 * 8
        ):  # Bound latency during continuous speech.
            if self.speech_frames >= MIN_SPEECH_FRAMES:
                overlap_pcm = b"".join(self.preroll)
                jobs.append(self._finish(overlap=True))
                self.active = True
                self.utterance = bytearray(overlap_pcm)
                self.silence_frames = 0
                self.speech_frames = 0
                self.frames_since_partial = 0
            else:
                self._reset()
        return jobs

    def _job(self, is_final: bool, overlap: bool) -> AudioJob:
        return AudioJob(
            segment_id=str(self.segment_number),
            pcm=bytes(self.utterance),
            is_final=is_final,
            overlap=overlap,
            created_at=time.perf_counter(),
        )

    def _reset(self) -> None:
        self.preroll.clear()
        self.utterance = bytearray()
        self.active = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.frames_since_partial = 0

    def _finish(self, overlap: bool) -> AudioJob:
        job = self._job(is_final=True, overlap=overlap)
        self.segment_number += 1
        self._reset()
        return job


def trim_overlap(previous: str, current: str) -> str:
    """Remove only an exact word overlap created when an 8-second window rolls over."""
    before = previous.split()
    after = current.split()
    for size in range(min(12, len(before), len(after)), 0, -1):
        if [word.lower() for word in before[-size:]] == [
            word.lower() for word in after[:size]
        ]:
            return " ".join(after[size:])
    return current
