import asyncio
import unittest

import httpx

from server.app import (
    TranslationError,
    LiveSession,
    transcript_message,
    translate,
    translation_message,
    translation_prompt,
    translation_targets,
)
from server.asr import WhisperMLXEngine, create_asr_engine
from server.streaming import FRAME_BYTES, AudioJob, VoiceSegmenter, trim_overlap


class FakeVad:
    def __init__(self, states):
        self.states = iter(states)

    def is_speech(self, frame, sample_rate):
        return next(self.states)


class FailingOllama:
    async def post(self, *args, **kwargs):
        raise httpx.ConnectError("offline")


class SuccessfulResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "Hello", "eval_count": 1}


class SuccessfulOllama:
    def __init__(self):
        self.payload = None

    async def post(self, url, json):
        self.payload = json
        return SuccessfulResponse()


class RecordingSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


class PipelineTests(unittest.TestCase):
    def test_target_selection_skips_source(self):
        self.assertEqual(translation_targets("fr", ["fr", "en", "ja"]), ["en", "ja"])
        self.assertEqual(translation_targets("ja", ["fr", "en", "ja"]), ["fr", "en"])

    def test_partial_and_final_messages_share_segment(self):
        partial = AudioJob("42", b"", False, False, 0)
        final = AudioJob("42", b"", True, False, 0)
        self.assertEqual(
            transcript_message(partial, "I want", "en"),
            {
                "type": "transcript.partial",
                "segmentId": "42",
                "text": "I want",
                "language": "en",
                "isFinal": False,
            },
        )
        self.assertEqual(
            transcript_message(final, "I want to go", "en")["type"], "transcript.final"
        )
        self.assertEqual(
            translation_message("42", "en", "ja", "日本に行きたいです。", False),
            {
                "type": "translation",
                "segmentId": "42",
                "sourceLanguage": "en",
                "targetLanguage": "ja",
                "text": "日本に行きたいです。",
                "isFinal": False,
            },
        )

    def test_new_translation_cancels_the_previous_segment_request(self):
        async def run():
            session = LiveSession(None, None)
            first = asyncio.create_task(asyncio.sleep(60))
            session.track_translation("42", first)
            second = asyncio.create_task(asyncio.sleep(0))
            session.track_translation("42", second)
            await asyncio.sleep(0)
            self.assertTrue(first.cancelled())
            self.assertIs(session.translation_tasks_by_segment["42"], second)
            await second

        asyncio.run(run())

    def test_stale_translation_is_not_sent(self):
        async def run():
            socket = RecordingSocket()
            session = LiveSession(None, socket)
            session.translation_revisions["42"] = 2
            sent = await session.send_translation(
                AudioJob("42", b"", False, False, 0),
                "en",
                "ja",
                "日本に行きたいです。",
                1,
            )
            self.assertFalse(sent)
            self.assertEqual(socket.messages, [])

        asyncio.run(run())

    def test_source_language_can_change_between_segments(self):
        first = transcript_message(AudioJob("1", b"", True, False, 0), "Bonjour", "fr")
        second = transcript_message(
            AudioJob("2", b"", True, False, 0), "明日は東京に行きます。", "ja"
        )
        self.assertEqual((first["language"], second["language"]), ("fr", "ja"))

    def test_vad_emits_partial_then_final(self):
        frames = [False] * 5 + [True] * 50 + [False] * 25
        segmenter = VoiceSegmenter(vad=FakeVad(frames))
        jobs = segmenter.feed(b"\0" * FRAME_BYTES * len(frames))
        self.assertEqual([job.is_final for job in jobs], [False, True])
        self.assertEqual({job.segment_id for job in jobs}, {"0"})

    def test_overlap_deduplication(self):
        self.assertEqual(
            trim_overlap("I want to go to", "to Japan tomorrow"), "Japan tomorrow"
        )
        self.assertEqual(trim_overlap("Bonjour", "Actually, hello"), "Actually, hello")

    def test_translation_prompt_is_restrictive_and_japanese_is_scripted(self):
        prompt = translation_prompt("Bonjour", "fr", "ja")
        self.assertIn("Return only the translated text", prompt)
        self.assertIn("never romanization", prompt)

    def test_ollama_connection_error_is_clean(self):
        with self.assertRaises(TranslationError):
            asyncio.run(
                translate(FailingOllama(), "Bonjour", "fr", "en", "translategemma:4b")
            )

    def test_translation_disables_thinking_and_keeps_model_warm(self):
        client = SuccessfulOllama()
        text, stats = asyncio.run(
            translate(client, "Bonjour", "fr", "en", "qwen3:0.6b")
        )
        self.assertEqual((text, stats["eval_count"]), ("Hello", 1))
        self.assertEqual(client.payload["think"], False)
        self.assertEqual(client.payload["keep_alive"], "5m")

    def test_model_configuration_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            WhisperMLXEngine("unknown")
        with self.assertRaises(ValueError):
            create_asr_engine("unknown_backend", "small")
        with self.assertRaises(ValueError):
            create_asr_engine("faster-whisper", "unknown_model")


if __name__ == "__main__":
    unittest.main()
