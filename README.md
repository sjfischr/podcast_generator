# podcast_gen

Generate a full-length podcast MP3 from a plain Markdown script using
**ElevenLabs Eleven v3** and the **Text to Dialogue API**.

---

## Why v3 + Text to Dialogue?

Traditional TTS pipelines make one API call per speaker turn and stitch the
audio together afterward. The result sounds like two people reading into
separate microphones — tonally flat and contextually unaware.

The **ElevenLabs Eleven v3 Text to Dialogue API** (`POST /v1/text-to-dialogue`)
changes this fundamentally. You send the entire multi-speaker script as a
batch of `{text, voice_id}` pairs and the model generates the full exchange
in one pass. Because it sees the whole conversation, it can:

- Respond to the _emotional register_ of the previous turn
- Produce natural interruptions, overlapping energy, and trailing sentences
- Apply delivery cues from inline **v3 audio tags** like `[laughing]`,
  `[whispering]`, `[sighing]`, `[elated]`, `[cautiously]`, etc.

The result sounds like a real conversation, not stitched narrator tracks.

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.11+ | Uses `match`-style `|` union types |
| `elevenlabs >= 1.50.0` | Must include `text_to_dialogue` SDK namespace |
| `pydub` | Audio stitching and export |
| `python-dotenv` | `.env` file support |
| **ffmpeg** | Required by pydub to write MP3 |

### Install Python packages

```bash
pip install "elevenlabs>=1.50.0" pydub python-dotenv pytest

# Python 3.13+ only: audioop was removed from stdlib; pydub needs this shim
pip install audioop-lts
```

### Install ffmpeg

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| Windows | `winget install ffmpeg` or `choco install ffmpeg` |

---

## Quick start

### 1 — Get your API key

Sign up at [elevenlabs.io](https://elevenlabs.io) and copy your key from
**Profile → API Keys**.  Creator tier or above is required for the
`mp3_44100_192` output format (192 kbps).

### 2 — Set the key

```bash
# Option A: environment variable
export ELEVENLABS_API_KEY=sk-...

# Option B: .env file in the project directory
echo "ELEVENLABS_API_KEY=sk-..." > .env
```

### 3 — Configure voices

Open `generate_podcast.py` and update `CONFIG` with your chosen voice IDs.
Browse the [Voice Library](https://elevenlabs.io/voice-library) to find voices
that suit your hosts.

```python
CONFIG = {
    "HOST_VOICE_ID":  os.getenv("HOST_VOICE_ID",  "Qin2NRfiKVQMJLxnoZaY"),
    "STEVE_VOICE_ID": os.getenv("STEVE_VOICE_ID", "LsL94tCOGlpxzIXa4Irh"),
    ...
}
```

You can also pass them as environment variables:

```bash
export HOST_VOICE_ID=21m00Tcm4TlvDq8ikWAM   # Rachel
export STEVE_VOICE_ID=TxGEqnHWrfWFTfGW9XjX  # Josh
```

### 4 — Write your script

Create (or edit) `podcast_script.md`:

```markdown
HOST: Welcome to the show. Today we're talking about something wild.
STEVE: [laughing] You have no idea how wild it gets.

[MUSIC_INTRO]

HOST: Okay, so walk me through the origin story.
STEVE: [sighing] It started at 2am on a Tuesday...
```

**Script format rules:**

| Line format | Meaning |
|---|---|
| `HOST: text` | Host speaker turn |
| `STEVE: text` | Steve speaker turn |
| `[MUSIC_INTRO]` | 5-second silence placeholder — overdub music in your DAW |
| `[MUSIC_OUTRO]` | Same as above |
| `[PAUSE_SHORT]` | 1.5-second pause |
| `[PAUSE_LONG]` | 3-second pause |
| Lines starting with `#` | Comments (ignored) |
| Blank lines | Ignored |

**v3 audio tags** can be embedded inline in any speaker line:

```
HOST: [laughs] Okay but that's actually genius.
STEVE: [whispers] Don't tell anyone I told you this.
HOST: [excited] That's the best outcome I could have imagined!
STEVE: [mischievously] Well, I may have exaggerated slightly.
HOST: [sighs] Of course you did.
```

The Eleven v3 model interprets these tags natively to shape emotional delivery.

| Tag | Effect |
|---|---|
| `[laughs]` · `[laughs harder]` · `[starts laughing]` | Laughter, escalating |
| `[whispers]` | Hushed delivery |
| `[sighs]` · `[exhales]` | Breath-based beats |
| `[sarcastic]` · `[mischievously]` | Tonal colour |
| `[curious]` · `[excited]` | Question / enthusiasm energy |
| `[snorts]` · `[wheezing]` | Involuntary reactions |
| `[crying]` | Emotional break |

### 5 — Generate the episode

```bash
# Default: reads podcast_script.md, writes episode.mp3
python generate_podcast.py

# Custom paths
python generate_podcast.py --script my_script.md --output my_episode.mp3

# Clear cache and regenerate everything
python generate_podcast.py --no-cache

# Use per-turn TTS instead of the Dialogue API
python generate_podcast.py --tts-mode
```

---

## How it works

```
podcast_script.md
      │
      ▼
 parse_script()          Turns the file into Turn objects (HOST/STEVE/DIRECTION)
      │
      ▼
 build_batches()         Groups consecutive speech turns between direction tags
      │                  into DialogueBatch objects (max 4,500 chars/batch)
      ▼
 For each batch:
   text_to_dialogue      POST /v1/text-to-dialogue — single MP3 for the whole
   .convert()            multi-speaker exchange, with full conversational context
      │
      ▼
 Direction tags          Converted to silence AudioSegments (PAUSE/MUSIC gaps)
      │
      ▼
 Stitch + export         pydub concatenates all segments → episode.mp3
```

### Caching

Each dialogue batch is cached as an MP3 in `audio_segments/` using a SHA-1
hash of the batch content.  Re-running the script skips batches that haven't
changed, which makes iteration fast when you're tweaking later sections of a
long script.

Use `--no-cache` to force a full regeneration.

---

## Model reference

| Model | Use case | Char limit | Cost |
|---|---|---|---|
| `eleven_v3` (**default**) | Expressive, contextual dialogue, 70+ languages | 5,000/input | Standard |
| `eleven_flash_v2_5` | Ultra-fast (~75 ms), 32 languages | 40,000 | 50% cheaper |
| `eleven_multilingual_v2` | Stable, consistent, 29 languages | 10,000 | Standard |

> **Note:** Only `eleven_v3` supports the Text to Dialogue API.
> `eleven_flash_v2_5` and `eleven_multilingual_v2` require `--tts-mode`.

---

## Output format

The default output format is `mp3_44100_192` (44.1 kHz, 192 kbps stereo MP3).
This requires **Creator tier** or above.

| Format | Tier required |
|---|---|
| `mp3_44100_128` | Free |
| `mp3_44100_192` | Creator+ |
| `pcm_44100` | Pro+ (lossless, uncompressed) |

Change `OUTPUT_FORMAT` in `CONFIG` to switch formats.

---

## Generating an intro sting

`generate_intro.py` uses the **ElevenLabs Sound Effects API** (`POST /v1/sound-generation`)
to generate a short audio sting you can layer at the `[MUSIC_INTRO]` gap.

```bash
# Generate with the default prompt (8 seconds, futuristic glitch + synth)
python generate_intro.py

# List all four built-in prompt variants
python generate_intro.py --list

# Try a different variant
python generate_intro.py --prompt-index 1 --output intro_warm.mp3

# Tune duration and prompt influence
python generate_intro.py --duration 6.0 --prompt-influence 0.6
```

| Flag | Default | Description |
|---|---|---|
| `--output` | `intro_sting.mp3` | Output filename |
| `--duration` | `8.0` | Length in seconds (0.1–30) |
| `--prompt-influence` | `0.4` | How closely to follow the text prompt (0.0–1.0) |
| `--prompt-index` | `0` | Which built-in prompt to use (0–3) |
| `--list` | — | Print all prompt variants and exit |

---

## Music and post-production

`[MUSIC_INTRO]` and `[MUSIC_OUTRO]` tags insert **5-second silence gaps** in
the final MP3.  The intended workflow is:

1. Generate `episode.mp3`
2. Import it into your DAW (Reaper, Logic, Audacity, etc.)
3. Locate the silence gaps at the start / end
4. Drop your intro / outro music onto a second track at those positions
5. Mix and export your final episode

---

## Running the tests

```bash
# Install test dependencies (just pytest)
pip install pytest

# Unit tests only (no API key needed)
python -m pytest test_podcast_gen.py -v -m unit

# All tests including live API smoke tests
ELEVENLABS_API_KEY=sk-... python -m pytest test_podcast_gen.py -v

# Without pytest
python test_podcast_gen.py
```

The smoke tests make a few short real API calls to verify the v3 endpoints
respond correctly. They use the voice IDs configured in `CONFIG` and consume
a small number of characters.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | _(required)_ | Your ElevenLabs API key |
| `HOST_VOICE_ID` | `Qin2NRfiKVQMJLxnoZaY` | Voice ID for HOST speaker |
| `STEVE_VOICE_ID` | `LsL94tCOGlpxzIXa4Irh` | Voice ID for STEVE speaker |

---

## Project structure

```
podcast_gen/
├── generate_podcast.py   # Main script — v3 Dialogue API pipeline
├── generate_intro.py     # Intro sting generator — Sound Effects API
├── podcast_script.md     # Your podcast script (edit this)
├── test_podcast_gen.py   # Unit + smoke tests
├── README.md             # This file
├── .env                  # API key (create this, not committed)
└── audio_segments/       # Cached batch MP3s (auto-created, safe to delete)
```

---

## API reference

- [ElevenLabs API introduction](https://elevenlabs.io/docs/api-reference/introduction)
- [Text to Dialogue API](https://elevenlabs.io/docs/api-reference/text-to-dialogue/convert)
- [Text to Speech API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
- [Eleven v3 model](https://elevenlabs.io/docs/overview/models#eleven-v3)
- [v3 audio tags and prompting guide](https://elevenlabs.io/docs/overview/capabilities/text-to-dialogue#prompting)
- [Voice Library](https://elevenlabs.io/voice-library)
