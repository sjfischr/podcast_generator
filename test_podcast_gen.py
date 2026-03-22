#!/usr/bin/env python3
"""
test_podcast_gen.py
-------------------
Unit and smoke tests for generate_podcast.py.

Tests are split into two categories:

  Unit tests  — run with no API key, no network, no ffmpeg required.
                These test the parser, batch builder, audio helpers, etc.

  Smoke tests — require ELEVENLABS_API_KEY.  They make real, short API calls
                to verify the ElevenLabs v3 endpoints actually work.

Usage:
    # Run all tests (unit only if no API key set):
    python -m pytest test_podcast_gen.py -v

    # Run only unit tests:
    python -m pytest test_podcast_gen.py -v -m unit

    # Run smoke tests (also runs units):
    ELEVENLABS_API_KEY=sk-... python -m pytest test_podcast_gen.py -v -m smoke

    # Quick self-contained run without pytest:
    python test_podcast_gen.py
"""

import hashlib
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make sure the sibling module is importable regardless of CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import generate_podcast as gp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_SCRIPT = textwrap.dedent("""\
    HOST: Hello and welcome to the show.
    STEVE: Thanks for having me.
    [PAUSE_SHORT]
    HOST: Let's dive right in.
    STEVE: Absolutely.
    [MUSIC_OUTRO]
""")

MULTI_SPEAKER_SCRIPT = textwrap.dedent("""\
    # This is a comment — should be ignored
    HOST: First line.
    STEVE: Second line.
    HOST: Third line.
    [PAUSE_LONG]
    STEVE: After the pause.
""")


def write_temp_script(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


# ===========================================================================
# Unit tests
# ===========================================================================

class TestParseScript(unittest.TestCase):

    def test_basic_parsing(self):
        p = write_temp_script(MINIMAL_SCRIPT)
        try:
            turns = gp.parse_script(str(p))
        finally:
            p.unlink()

        kinds = [t.kind for t in turns]
        self.assertEqual(kinds, ["HOST", "STEVE", "DIRECTION", "HOST", "STEVE", "DIRECTION"])

    def test_comments_and_blanks_ignored(self):
        p = write_temp_script(MULTI_SPEAKER_SCRIPT)
        try:
            turns = gp.parse_script(str(p))
        finally:
            p.unlink()

        # comment line and blank lines must not appear
        self.assertNotIn("DIRECTION", [t.text for t in turns if t.text.startswith("#")])
        speech = [t for t in turns if t.kind != "DIRECTION"]
        self.assertEqual(len(speech), 4)

    def test_direction_text_extracted_correctly(self):
        p = write_temp_script(MINIMAL_SCRIPT)
        try:
            turns = gp.parse_script(str(p))
        finally:
            p.unlink()

        directions = [t.text for t in turns if t.kind == "DIRECTION"]
        self.assertIn("PAUSE_SHORT", directions)
        self.assertIn("MUSIC_OUTRO", directions)

    def test_missing_file_exits(self):
        with self.assertRaises(SystemExit):
            gp.parse_script("/this/path/does/not/exist/script.md")

    def test_speaker_text_stripped(self):
        script = "HOST:   leading and trailing spaces   \n"
        p = write_temp_script(script)
        try:
            turns = gp.parse_script(str(p))
        finally:
            p.unlink()

        self.assertEqual(turns[0].text, "leading and trailing spaces")

    def test_v3_audio_tags_pass_through(self):
        """Inline v3 tags like [laughing] must survive unparsed as text content."""
        script = "HOST: [laughing] That's hilarious!\n"
        p = write_temp_script(script)
        try:
            turns = gp.parse_script(str(p))
        finally:
            p.unlink()

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].kind, "HOST")
        self.assertIn("[laughing]", turns[0].text)


class TestBuildBatches(unittest.TestCase):

    VOICE_MAP = {"HOST": "voice_host_id", "STEVE": "voice_steve_id"}

    def _turns_from(self, script: str):
        p = write_temp_script(script)
        try:
            return gp.parse_script(str(p))
        finally:
            p.unlink()

    def test_directions_split_batches(self):
        turns = self._turns_from(MINIMAL_SCRIPT)
        items = gp.build_batches(turns, self.VOICE_MAP)

        # Expect: batch, batch, DIRECTION(PAUSE_SHORT), batch, batch, DIRECTION(MUSIC_OUTRO)
        batch_count = sum(1 for i in items if isinstance(i, gp.DialogueBatch))
        dir_count   = sum(1 for i in items if isinstance(i, gp.Turn) and i.kind == "DIRECTION")
        self.assertEqual(dir_count, 2)
        self.assertGreaterEqual(batch_count, 2)

    def test_batch_inputs_have_correct_voice_ids(self):
        turns = self._turns_from("HOST: A\nSTEVE: B\n")
        items = gp.build_batches(turns, self.VOICE_MAP)

        batches = [i for i in items if isinstance(i, gp.DialogueBatch)]
        self.assertEqual(len(batches), 1)
        b = batches[0]
        self.assertEqual(b.inputs[0]["voice_id"], "voice_host_id")
        self.assertEqual(b.inputs[1]["voice_id"], "voice_steve_id")

    def test_batch_splits_at_char_limit(self):
        long_text = "x" * (gp.CONFIG["MAX_CHARS_PER_BATCH"] - 10)
        script = f"HOST: {long_text}\nSTEVE: {long_text}\n"
        turns = self._turns_from(script)
        items = gp.build_batches(turns, self.VOICE_MAP)

        batches = [i for i in items if isinstance(i, gp.DialogueBatch)]
        # Two turns each near the limit must produce two separate batches
        self.assertEqual(len(batches), 2)

    def test_empty_script_produces_no_batches(self):
        turns = self._turns_from("# just a comment\n")
        items = gp.build_batches(turns, self.VOICE_MAP)
        self.assertEqual(items, [])

    def test_total_chars_property(self):
        batch = gp.DialogueBatch(index=0)
        batch.inputs = [{"text": "hello", "voice_id": "v"}, {"text": "world!", "voice_id": "v"}]
        self.assertEqual(batch.total_chars, 11)


class TestDirectionToSilence(unittest.TestCase):

    def test_music_intro_duration(self):
        seg = gp.direction_to_silence("MUSIC_INTRO")
        self.assertEqual(len(seg), gp.CONFIG["MUSIC_PLACEHOLDER_MS"])

    def test_pause_short_duration(self):
        seg = gp.direction_to_silence("PAUSE_SHORT")
        self.assertEqual(len(seg), gp.CONFIG["PAUSE_SHORT_MS"])

    def test_pause_long_duration(self):
        seg = gp.direction_to_silence("PAUSE_LONG")
        self.assertEqual(len(seg), gp.CONFIG["PAUSE_LONG_MS"])

    def test_unknown_tag_fallback(self):
        seg = gp.direction_to_silence("SOMETHING_WEIRD")
        self.assertEqual(len(seg), gp.CONFIG["PAUSE_SHORT_MS"])

    def test_case_insensitive(self):
        seg_lower = gp.direction_to_silence("pause_short")
        seg_upper = gp.direction_to_silence("PAUSE_SHORT")
        self.assertEqual(len(seg_lower), len(seg_upper))


class TestBatchCachePath(unittest.TestCase):

    def test_stable_hash_same_inputs(self):
        batch = gp.DialogueBatch(index=0)
        batch.inputs = [{"text": "hello", "voice_id": "abc"}]
        p1 = gp.batch_cache_path(Path("/tmp"), batch)
        p2 = gp.batch_cache_path(Path("/tmp"), batch)
        self.assertEqual(p1, p2)

    def test_different_inputs_different_path(self):
        b1 = gp.DialogueBatch(index=0)
        b1.inputs = [{"text": "hello", "voice_id": "aaa"}]
        b2 = gp.DialogueBatch(index=0)
        b2.inputs = [{"text": "world", "voice_id": "bbb"}]
        self.assertNotEqual(
            gp.batch_cache_path(Path("/tmp"), b1),
            gp.batch_cache_path(Path("/tmp"), b2),
        )

    def test_filename_contains_batch_index(self):
        batch = gp.DialogueBatch(index=7)
        batch.inputs = [{"text": "test", "voice_id": "v"}]
        path = gp.batch_cache_path(Path("/tmp"), batch)
        self.assertIn("007", path.name)


class TestConfig(unittest.TestCase):

    def test_model_is_eleven_v3(self):
        self.assertEqual(gp.CONFIG["MODEL_ID"], "eleven_v3")

    def test_output_format_is_high_quality(self):
        self.assertIn("192", gp.CONFIG["OUTPUT_FORMAT"])

    def test_char_limit_within_v3_bounds(self):
        # eleven_v3 hard limit is 5,000 chars per input item
        self.assertLessEqual(gp.CONFIG["MAX_CHARS_PER_BATCH"], 5000)


# ===========================================================================
# Smoke tests  (require ELEVENLABS_API_KEY, skipped otherwise)
# ===========================================================================

SKIP_SMOKE = not os.getenv("ELEVENLABS_API_KEY")
SKIP_MSG   = "ELEVENLABS_API_KEY not set — skipping smoke tests"


class TestSmokeDialogueAPI(unittest.TestCase):
    """Live API calls — short texts to minimise credit usage."""

    @unittest.skipIf(SKIP_SMOKE, SKIP_MSG)
    def test_dialogue_api_returns_mp3_bytes(self):
        """
        Calls POST /v1/text-to-dialogue with two short turns and verifies
        we receive a non-empty MP3 byte-string back.
        """
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        batch = gp.DialogueBatch(index=0)
        batch.inputs = [
            {"text": "Hello, welcome to the show.", "voice_id": gp.CONFIG["HOST_VOICE_ID"]},
            {"text": "Thanks for having me.", "voice_id": gp.CONFIG["STEVE_VOICE_ID"]},
        ]
        batch.turns = []  # not needed for the API call

        mp3_bytes = gp.generate_dialogue_batch(client, batch)

        self.assertIsInstance(mp3_bytes, bytes)
        self.assertGreater(len(mp3_bytes), 1_000, "MP3 output suspiciously small")
        # MP3 files start with 0xff 0xfb, 0xff 0xf3, 0xff 0xf2, or ID3 header
        self.assertTrue(
            mp3_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") or
            mp3_bytes[:3] == b"ID3",
            "Response does not look like a valid MP3 file",
        )
        print(f"\n  [smoke] dialogue API returned {len(mp3_bytes):,} bytes — OK")

    @unittest.skipIf(SKIP_SMOKE, SKIP_MSG)
    def test_tts_api_returns_mp3_bytes(self):
        """
        Calls POST /v1/text-to-speech/{voice_id} with a single short turn.
        Tests the --tts-mode fallback path.
        """
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        mp3_bytes = gp.generate_speech_single(
            client,
            voice_id=gp.CONFIG["HOST_VOICE_ID"],
            text="This is a test of the single-turn fallback.",
        )

        self.assertIsInstance(mp3_bytes, bytes)
        self.assertGreater(len(mp3_bytes), 1_000)
        print(f"\n  [smoke] TTS API returned {len(mp3_bytes):,} bytes — OK")

    @unittest.skipIf(SKIP_SMOKE, SKIP_MSG)
    def test_full_pipeline_dialogue_mode(self):
        """
        Runs main() end-to-end with a tiny 4-line script in dialogue mode,
        verifying an MP3 file is produced with non-trivial size.
        """
        from elevenlabs import ElevenLabs

        mini_script = textwrap.dedent("""\
            HOST: Hello.
            STEVE: Hi there.
            [PAUSE_SHORT]
            HOST: Thanks for coming.
            STEVE: My pleasure.
        """)

        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "mini.md"
            output_path = Path(tmp) / "mini.mp3"
            script_path.write_text(mini_script, encoding="utf-8")

            # Override output dir so we don't pollute the workspace cache
            original_dir = gp.CONFIG["OUTPUT_DIR"]
            gp.CONFIG["OUTPUT_DIR"] = str(Path(tmp) / "segments")
            try:
                gp.main(str(script_path), str(output_path), tts_mode=False)
            finally:
                gp.CONFIG["OUTPUT_DIR"] = original_dir

            self.assertTrue(output_path.exists(), "Output MP3 was not created")
            size = output_path.stat().st_size
            self.assertGreater(size, 5_000, f"Output MP3 is suspiciously small: {size} bytes")
            print(f"\n  [smoke] full pipeline produced {size:,}-byte MP3 — OK")


# ===========================================================================
# Entry point for running without pytest
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("podcast_gen test suite")
    print("=" * 60)
    if SKIP_SMOKE:
        print("NOTE: Smoke tests SKIPPED (no ELEVENLABS_API_KEY in env)\n")
    else:
        print("Smoke tests ENABLED (ELEVENLABS_API_KEY detected)\n")

    unittest.main(verbosity=2)
