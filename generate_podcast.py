#!/usr/bin/env python3
"""
generate_podcast.py
-------------------
Parses podcast_script.md and generates a stitched MP3 using the ElevenLabs
Eleven v3 Text to Dialogue API (/v1/text-to-dialogue).

The Text to Dialogue API is the v3 flagship for multi-speaker content.
Instead of issuing one TTS call per speaker turn, consecutive speech turns
between direction tags are sent together as a single dialogue request.  The
model sees the full conversational context and produces naturally flowing,
emotionally appropriate audio in one pass — no manual stitching artifacts.

Direction tags [MUSIC_INTRO], [MUSIC_OUTRO], [PAUSE_SHORT], [PAUSE_LONG]
separate dialogue batches and are rendered as silence placeholders so you
can overdub real music / SFX in your DAW.

v3 Audio-tag support (embed directly in speaker lines):
    [laughing]   [whispering]   [sighing]   [groaning]
    [cheerfully] [cautiously]   [elated]    [sad]
    [applause]   [gentle footsteps]          etc.

Requirements:
    pip install "elevenlabs>=1.50.0" pydub python-dotenv

    Python 3.13+ only: pydub requires audioop which was removed from stdlib.
    Install the compatibility shim: pip install audioop-lts

ffmpeg must be on PATH for pydub to write MP3:
    macOS:   brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg
    Windows: winget install ffmpeg   (or: choco install ffmpeg)

Usage:
    # Set your key (or create a .env file with ELEVENLABS_API_KEY=...)
    export ELEVENLABS_API_KEY=your_key_here

    # Run with defaults (reads podcast_script.md, writes episode.mp3)
    python generate_podcast.py

    # Custom script / output path
    python generate_podcast.py --script my_script.md --output my_episode.mp3

    # Force re-generation (ignore cached batches)
    python generate_podcast.py --no-cache

    # Fall back to per-turn TTS instead of the Dialogue API
    python generate_podcast.py --tts-mode
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — swap in your chosen voice IDs from ElevenLabs Voice Library
# ---------------------------------------------------------------------------

CONFIG = {
    # Host voice — warm, conversational, curious
    # Browse voices: https://elevenlabs.io/voice-library
    # e.g. "Rachel" = 21m00Tcm4TlvDq8ikWAM | "Adam" = pNInz6obpgDQGcFmaJgB
    "HOST_VOICE_ID": os.getenv("HOST_VOICE_ID", "Qin2NRfiKVQMJLxnoZaY"),

    # Steve voice — confident, technical, dry humor
    # e.g. "Josh" = TxGEqnHWrfWFTfGW9XjX | "Antoni" = ErXwobaYiN019PkySvjV
    "STEVE_VOICE_ID": os.getenv("STEVE_VOICE_ID", "LsL94tCOGlpxzIXa4Irh"),

    # eleven_v3         — flagship, 70+ languages, 5,000 char/input limit,
    #                     emotionally expressive, powers the Dialogue API
    # eleven_flash_v2_5 — ultra-fast (~75 ms), 32 languages, great for
    #                     cheaper iteration (does NOT support Dialogue API)
    "MODEL_ID": "eleven_v3",

    # Max characters accumulated per dialogue batch before splitting.
    # eleven_v3 supports 5,000 chars per input item; stay comfortably under.
    "MAX_CHARS_PER_BATCH": 4500,

    # Output audio format.  192 kbps requires Creator tier or above.
    # Other options: mp3_44100_128 (default/free), pcm_44100 (Pro, lossless)
    "OUTPUT_FORMAT": "mp3_44100_192",

    # Silence durations in milliseconds
    "PAUSE_BETWEEN_BATCHES_MS": 300,   # brief gap between dialogue batches
    "PAUSE_SHORT_MS": 1_500,           # [PAUSE_SHORT] direction tag
    "PAUSE_LONG_MS": 3_000,            # [PAUSE_LONG] direction tag
    "MUSIC_PLACEHOLDER_MS": 5_000,     # [MUSIC_INTRO]/[MUSIC_OUTRO] — overdub in DAW

    # Paths
    "SCRIPT_PATH": "podcast_script.md",
    "OUTPUT_DIR": "audio_segments",    # cached batch MP3s land here
    "FINAL_OUTPUT": "episode.mp3",

    # Retry / rate-limit handling
    "MAX_RETRIES": 3,
    "RETRY_DELAY_SEC": 5,
}

# VoiceSettings used only in --tts-mode (per-turn TTS fallback).
# The Dialogue API uses the model's own contextual delivery — influence it
# via the text itself and v3 audio tags like [laughing], [softly], etc.
TTS_VOICE_SETTINGS = VoiceSettings(
    stability=0.45,
    similarity_boost=0.80,
    style=0.30,
    use_speaker_boost=True,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    kind: Literal["HOST", "STEVE", "DIRECTION"]
    text: str
    line_number: int


@dataclass
class DialogueBatch:
    """A sequence of speech turns to send as one dialogue request."""
    index: int
    inputs: list[dict] = field(default_factory=list)  # [{"text":..., "voice_id":...}]
    turns: list[Turn] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(item["text"]) for item in self.inputs)


# ---------------------------------------------------------------------------
# Script parser
# ---------------------------------------------------------------------------

def parse_script(filepath: str) -> list[Turn]:
    """
    Parse a podcast Markdown script into a list of Turn objects.

    Recognized line formats:
        HOST: some spoken text
        STEVE: some spoken text
        [DIRECTION_TAG]    — e.g. [PAUSE_SHORT], [PAUSE_LONG],
                             [MUSIC_INTRO], [MUSIC_OUTRO]
        # comment / blank  — ignored

    v3 audio tags embedded in speaker text are passed through verbatim,
    e.g.:  HOST: [laughing] Oh, come on — you're serious?
    """
    turns = []
    path = Path(filepath)

    if not path.exists():
        print(f"ERROR: Script not found at {filepath}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                turns.append(Turn(kind="DIRECTION", text=line[1:-1], line_number=lineno))

            elif line.upper().startswith("HOST:"):
                text = line[5:].strip()
                if text:
                    turns.append(Turn(kind="HOST", text=text, line_number=lineno))

            elif line.upper().startswith("STEVE:"):
                text = line[6:].strip()
                if text:
                    turns.append(Turn(kind="STEVE", text=text, line_number=lineno))

            else:
                print(f"  [WARN] Line {lineno} not recognized, skipping: {line[:60]}")

    return turns


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_batches(
    turns: list[Turn],
    voice_map: dict[str, str],
) -> list:
    """
    Group consecutive speech turns into DialogueBatch objects.

    A new batch is started when:
      1. A direction tag is encountered  (directions stay as bare Turn objects)
      2. Adding the next turn would exceed MAX_CHARS_PER_BATCH

    Returns a flat list of DialogueBatch and Turn (direction) objects.
    """
    items: list = []
    batch_index = 0
    current: DialogueBatch | None = None

    def close_batch() -> None:
        nonlocal current
        if current and current.inputs:
            items.append(current)
        current = None

    for turn in turns:
        if turn.kind == "DIRECTION":
            close_batch()
            items.append(turn)
            continue

        voice_id = voice_map[turn.kind]

        if current is None:
            current = DialogueBatch(index=batch_index)
            batch_index += 1

        # Split if this turn would overflow the character budget
        if current.inputs and (current.total_chars + len(turn.text)) > CONFIG["MAX_CHARS_PER_BATCH"]:
            close_batch()
            current = DialogueBatch(index=batch_index)
            batch_index += 1

        current.inputs.append({"text": turn.text, "voice_id": voice_id})
        current.turns.append(turn)

    close_batch()
    return items


# ---------------------------------------------------------------------------
# ElevenLabs API helpers
# ---------------------------------------------------------------------------

def _retry_call(fn, label: str) -> bytes:
    """Execute fn(), retrying up to MAX_RETRIES times on rate-limit errors."""
    for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
        try:
            result = fn()
            return b"".join(result)
        except Exception as exc:
            err = str(exc).lower()
            if "rate" in err or "429" in err:
                wait = CONFIG["RETRY_DELAY_SEC"] * attempt
                print(f"    Rate limited. Waiting {wait}s (retry {attempt}/{CONFIG['MAX_RETRIES']})…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"API call failed after {CONFIG['MAX_RETRIES']} retries: {label}")


def generate_dialogue_batch(client: ElevenLabs, batch: DialogueBatch) -> bytes:
    """
    Call POST /v1/text-to-dialogue with a full batch of speaker turns.

    The Dialogue API processes all turns together, giving the model the full
    conversational context for emotionally appropriate, naturally flowing audio.

    Docs: https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert
    """
    speakers_preview = ", ".join(
        f"{t.kind}({len(t.text)}c)" for t in batch.turns[:4]
    ) + ("…" if len(batch.turns) > 4 else "")
    print(f"    → {len(batch.inputs)} turns, {batch.total_chars} chars: {speakers_preview}")

    def call():
        return client.text_to_dialogue.convert(
            inputs=batch.inputs,
            model_id=CONFIG["MODEL_ID"],
            output_format=CONFIG["OUTPUT_FORMAT"],
        )

    return _retry_call(call, f"batch {batch.index}")


def generate_speech_single(client: ElevenLabs, voice_id: str, text: str) -> bytes:
    """
    Per-turn TTS via POST /v1/text-to-speech/{voice_id} with eleven_v3.
    Used only in --tts-mode.
    """
    def call():
        return client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=CONFIG["MODEL_ID"],
            voice_settings=TTS_VOICE_SETTINGS,
            output_format=CONFIG["OUTPUT_FORMAT"],
        )

    return _retry_call(call, f"TTS '{text[:40]}'")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def silence(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms)


def load_mp3_bytes(data: bytes) -> AudioSegment:
    return AudioSegment.from_file(io.BytesIO(data), format="mp3")


def direction_to_silence(tag: str) -> AudioSegment:
    """Convert a direction tag string to a silence AudioSegment."""
    tag_upper = tag.upper()
    if "MUSIC" in tag_upper:
        ms = CONFIG["MUSIC_PLACEHOLDER_MS"]
        print(f"    [{tag}] — {ms}ms silence placeholder (overdub music in DAW)")
    elif tag_upper == "PAUSE_LONG":
        ms = CONFIG["PAUSE_LONG_MS"]
        print(f"    [{tag}] — {ms}ms pause")
    elif tag_upper == "PAUSE_SHORT":
        ms = CONFIG["PAUSE_SHORT_MS"]
        print(f"    [{tag}] — {ms}ms pause")
    else:
        ms = CONFIG["PAUSE_SHORT_MS"]
        print(f"    [{tag}] — unknown direction tag, using {ms}ms silence")
    return silence(ms)


def batch_cache_path(seg_dir: Path, batch: DialogueBatch) -> Path:
    """Return a stable cache path keyed by the batch's content hash."""
    key = hashlib.sha1(
        json.dumps(batch.inputs, sort_keys=True).encode()
    ).hexdigest()[:12]
    return seg_dir / f"batch_{batch.index:03d}_{key}.mp3"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(script_path: str, output_path: str, tts_mode: bool = False) -> None:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set. Export it or add it to a .env file.")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)

    mode_label = "TTS per-turn (--tts-mode)" if tts_mode else "Text to Dialogue API"
    print(f"\nModel  : {CONFIG['MODEL_ID']}")
    print(f"Mode   : {mode_label}")
    print(f"Format : {CONFIG['OUTPUT_FORMAT']}")
    print(f"\nParsing script: {script_path}")

    turns = parse_script(script_path)
    speech_turns = [t for t in turns if t.kind != "DIRECTION"]
    dir_turns    = [t for t in turns if t.kind == "DIRECTION"]
    print(
        f"Found {len(turns)} lines → {len(speech_turns)} speech turns, "
        f"{len(dir_turns)} direction tags\n"
    )

    voice_map = {
        "HOST":  CONFIG["HOST_VOICE_ID"],
        "STEVE": CONFIG["STEVE_VOICE_ID"],
    }

    seg_dir = Path(CONFIG["OUTPUT_DIR"])
    seg_dir.mkdir(exist_ok=True)

    # Build the work list: DialogueBatch objects interleaved with direction Turns
    if tts_mode:
        work_items = turns  # every turn processed individually
    else:
        work_items = build_batches(turns, voice_map)
        n_batches = sum(1 for w in work_items if isinstance(w, DialogueBatch))
        print(f"Grouped into {n_batches} dialogue batches\n")

    segments: list[AudioSegment] = []
    batch_gap = silence(CONFIG["PAUSE_BETWEEN_BATCHES_MS"])

    for item in work_items:

        # ── Direction tag ─────────────────────────────────────────────────
        if isinstance(item, Turn) and item.kind == "DIRECTION":
            segments.append(direction_to_silence(item.text))
            continue

        # ── Dialogue batch (default mode) ─────────────────────────────────
        if isinstance(item, DialogueBatch):
            label = f"[batch {item.index:03d}]"
            cache_path = batch_cache_path(seg_dir, item)

            if cache_path.exists():
                print(f"{label} [cached] {cache_path.name}")
                audio = AudioSegment.from_mp3(str(cache_path))
            else:
                print(f"{label} Generating {len(item.inputs)}-turn dialogue…")
                mp3_bytes = generate_dialogue_batch(client, item)
                cache_path.write_bytes(mp3_bytes)
                audio = load_mp3_bytes(mp3_bytes)
                print(f"    ✓ {len(audio) / 1000:.1f}s saved → {cache_path.name}")

            segments.append(audio)
            segments.append(batch_gap)
            continue

        # ── Per-turn TTS fallback (--tts-mode) ───────────────────────────
        if isinstance(item, Turn) and item.kind != "DIRECTION":
            speaker  = item.kind
            voice_id = voice_map[speaker]
            preview  = item.text[:70] + ("…" if len(item.text) > 70 else "")
            print(f"  {speaker}: {preview}")

            cache_path = seg_dir / f"tts_{item.line_number:04d}_{speaker.lower()}.mp3"
            if cache_path.exists():
                print(f"    [cached] {cache_path.name}")
                audio = AudioSegment.from_mp3(str(cache_path))
            else:
                mp3_bytes = generate_speech_single(client, voice_id, item.text)
                cache_path.write_bytes(mp3_bytes)
                audio = load_mp3_bytes(mp3_bytes)
                print(f"    ✓ {len(audio) / 1000:.1f}s → {cache_path.name}")

            segments.append(audio)
            segments.append(batch_gap)

    if not segments:
        print("ERROR: No audio segments were generated.")
        sys.exit(1)

    print(f"\nStitching {len(segments)} segments…")
    episode = segments[0]
    for seg in segments[1:]:
        episode = episode + seg

    duration_min = len(episode) / 1_000 / 60
    print(f"Total duration: {duration_min:.1f} minutes")

    output = Path(output_path)
    print(f"Exporting to {output}…")
    episode.export(str(output), format="mp3", bitrate="192k", tags={
        "title":   "Welcome to the Party, Pal",
        "artist":  "The Evaluation Layer",
        "album":   "The Evaluation Layer Podcast",
        "genre":   "Technology",
        "comment": f"Generated with ElevenLabs {CONFIG['MODEL_ID']}",
    })

    music_tags = [t for t in turns if t.kind == "DIRECTION" and "MUSIC" in t.text.upper()]
    print(f"\nDone! Episode saved to: {output.resolve()}")
    if music_tags:
        print(
            f"\nNOTE: {len(music_tags)} music placeholder gap(s) "
            f"({CONFIG['MUSIC_PLACEHOLDER_MS']}ms each) are silent. "
            "Import into your DAW and overdub intro/outro music at those markers."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate podcast audio from a speaker-labeled Markdown script "
            "using ElevenLabs Eleven v3 and the Text to Dialogue API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--script",
        default=CONFIG["SCRIPT_PATH"],
        help=f"Path to the Markdown script (default: {CONFIG['SCRIPT_PATH']})",
    )
    parser.add_argument(
        "--output",
        default=CONFIG["FINAL_OUTPUT"],
        help=f"Output MP3 filename (default: {CONFIG['FINAL_OUTPUT']})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Delete cached audio segments and regenerate everything from scratch",
    )
    parser.add_argument(
        "--tts-mode",
        action="store_true",
        help=(
            "Use per-turn Text to Speech API (POST /v1/text-to-speech/{voice_id}) "
            "instead of the Text to Dialogue API."
        ),
    )
    args = parser.parse_args()

    if args.no_cache:
        import shutil
        seg_dir = Path(CONFIG["OUTPUT_DIR"])
        if seg_dir.exists():
            shutil.rmtree(seg_dir)
            print(f"Cache cleared: {seg_dir}/")

    main(args.script, args.output, tts_mode=args.tts_mode)
