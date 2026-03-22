#!/usr/bin/env python3
"""
generate_intro.py
-----------------
Generates a bespoke podcast intro sting for "The Evaluation Layer" using the
ElevenLabs Sound Effects API (POST /v1/sound-generation).

The output is a raw audio file you can layer under your music bed in a DAW,
or stitch directly into episode.mp3 where the [MUSIC_INTRO] silence gap sits.

The Sound Effects API takes a natural-language description and optional
duration/prompt_influence controls and returns an MP3 (or WAV for non-looping).

API docs: https://elevenlabs.io/docs/overview/capabilities/sound-effects

Usage:
    python generate_intro.py                  # writes intro_sting.mp3
    python generate_intro.py --duration 10    # 10-second version
    python generate_intro.py --list           # preview all candidate prompts
    python generate_intro.py --prompt-index 2 # generate a specific variant
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv()

# ---------------------------------------------------------------------------
# Candidate intro prompts  — pick the one that fits your show
# ---------------------------------------------------------------------------
# Tips from the Sound Effects prompting guide:
#   - Use audio terminology: braam, drone, whoosh, glitch, one-shot
#   - Describe the sequence of events for complex multi-part sounds
#   - Specify BPM or key for musical elements
#   - "One-shot" = non-repeating, which is what you want for an intro sting
# ---------------------------------------------------------------------------

PROMPTS = [
    # 0 — default: synthetic intelligence, forward momentum
    (
        "Futuristic tech podcast intro sting, one-shot. Opens with a sharp digital "
        "glitch burst and rising synthesizer arpeggio. Mid-section: pulsing "
        "low-frequency drone with subtle modulation and crisp hi-hat groove at 120 BPM. "
        "Resolves into a clean cinematic braam with atmospheric reverb tail. "
        "Intelligent, modern, slightly cinematic. Not aggressive — focused."
    ),
    # 1 — warmer, more narrative podcast feel
    (
        "Podcast intro sting, one-shot. Warm analog synth pad swells in, layered with "
        "a soft electronic pulse and occasional data-transmission glitch texture. "
        "A bright melodic motif plays over the top — optimistic and curious. "
        "Fades on a clean held chord. Feels like the beginning of something interesting."
    ),
    # 2 — more dramatic / cinematic
    (
        "Cinematic podcast intro, one-shot. Deep bass impact hit followed by a rising "
        "digital tone cluster with whoosh movement. Layered electronic percussion loop "
        "at 110 BPM builds tension. Sharp, decisive outro hit. Feels like a trailer "
        "for ideas — intelligent and a little ominous."
    ),
    # 3 — minimal / understated
    (
        "Minimal tech podcast intro sting, one-shot. Single clean synthesizer note "
        "with slow attack, surrounded by subtle ambient texture and gentle electronic "
        "rhythm. A soft glitch accent mid-way. Resolves quietly. Understated and "
        "thoughtful. Background-friendly — sits under a voiceover naturally."
    ),
]

DEFAULT_PROMPT_INDEX = 0
DEFAULT_DURATION = 8.0   # seconds — Sound Effects API max is 30.0
DEFAULT_PROMPT_INFLUENCE = 0.4  # 0.0 = max creative freedom, 1.0 = literal


def generate_intro(
    prompt: str,
    duration: float,
    prompt_influence: float,
    output_path: Path,
) -> None:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set. Export it or add it to a .env file.")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)

    print(f"\nGenerating intro sting via Sound Effects API")
    print(f"Duration      : {duration}s")
    print(f"Prompt infl.  : {prompt_influence}")
    print(f"Prompt preview: {prompt[:80]}…\n")

    # POST /v1/sound-generation
    # SDK: client.text_to_sound_effects.convert()
    result = client.text_to_sound_effects.convert(
        text=prompt,
        duration_seconds=duration,
        prompt_influence=prompt_influence,
    )

    # SDK returns a generator of bytes chunks
    audio_bytes = b"".join(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    print(f"Saved {len(audio_bytes):,} bytes → {output_path.resolve()}")
    print("\nNext steps:")
    print("  • Import intro_sting.mp3 into your DAW")
    print(f"  • Layer it at the [MUSIC_INTRO] gap in episode.mp3")
    print(f"  • Try different variants: python generate_intro.py --prompt-index 1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a podcast intro sting using the ElevenLabs Sound Effects API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default="intro_sting.mp3",
        help="Output filename (default: intro_sting.mp3)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Duration in seconds, 0.1–30.0 (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--prompt-influence", "-p",
        type=float,
        default=DEFAULT_PROMPT_INFLUENCE,
        dest="prompt_influence",
        help=(
            f"How literally the model follows the prompt, 0.0–1.0. "
            f"Lower = more creative. (default: {DEFAULT_PROMPT_INFLUENCE})"
        ),
    )
    parser.add_argument(
        "--prompt-index", "-i",
        type=int,
        default=DEFAULT_PROMPT_INDEX,
        dest="prompt_index",
        choices=range(len(PROMPTS)),
        help=f"Which prompt variant to use, 0–{len(PROMPTS) - 1} (default: {DEFAULT_PROMPT_INDEX})",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Print all prompt variants and exit without generating",
    )
    args = parser.parse_args()

    if args.list:
        for i, p in enumerate(PROMPTS):
            marker = " ← default" if i == DEFAULT_PROMPT_INDEX else ""
            print(f"\n[{i}]{marker}\n{p}")
        sys.exit(0)

    if not (0.1 <= args.duration <= 30.0):
        print("ERROR: --duration must be between 0.1 and 30.0")
        sys.exit(1)

    if not (0.0 <= args.prompt_influence <= 1.0):
        print("ERROR: --prompt-influence must be between 0.0 and 1.0")
        sys.exit(1)

    generate_intro(
        prompt=PROMPTS[args.prompt_index],
        duration=args.duration,
        prompt_influence=args.prompt_influence,
        output_path=Path(args.output),
    )
