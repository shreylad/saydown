# open-flow

Hold a key, talk, release. Cleaned-up text is pasted at your cursor, in any app.

A local, private take on Wispr Flow for Apple Silicon Macs. Your audio and text never leave your machine: speech is transcribed by Whisper and tidied by a small language model, both running on your Mac.

## What it does

- **Push to talk.** Hold the right Option key, speak, release.
- **Cleans up speech.** Removes filler words, fixes punctuation, and drops what you took back: "Can you please write down, sorry, please stop" becomes "Please stop."
- **Keeps your voice.** Complete sentences are never deleted, and casual words (gonna, kinda) stay as spoken.
- **Personal dictionary.** Names and jargon you list are spelled your way.
- **On-screen indicator.** A small pill shows "Listening" and "Writing…" so you know it heard you.
- **Never loses a dictation.** If the cleanup model is unavailable, the raw transcript is pasted instead.

## Requirements

- An Apple Silicon Mac (16 GB of memory is enough)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com), running

## Setup

```sh
ollama pull gemma4:e4b
uv run dictate.py
```

The first run downloads the Whisper model (about 1.6 GB). uv installs the Python dependencies on its own.

macOS will ask the app you launch it from (Terminal, iTerm, ...) for three permissions under System Settings → Privacy & Security. Grant all three, then start it again:

| Permission | Needed for |
|---|---|
| Microphone | Recording your voice |
| Input Monitoring | Seeing the hotkey |
| Accessibility | Pasting at your cursor |

## Usage

Click into any text field, hold **right Option (⌥)**, say a sentence, and release. Presses shorter than 0.3 seconds are ignored as accidental taps.

The terminal logs every dictation with the raw transcription beside the cleaned text, so you can see which step made a mistake.

## Dictionary

Create `dictionary.txt` next to `dictate.py`, one term per line:

```
Reykjavík
Siobhan
Kubernetes
```

The terms are given to Whisper as spelling hints and to the cleanup model as corrections for misheard words. The file is read on every dictation, so edits apply without a restart. It is git-ignored because it is personal.

## Configuration

| Setting | How | Default |
|---|---|---|
| Cleanup model | `DICTATE_MODEL` environment variable | `gemma4:e4b` |
| Hotkey | `HOTKEY` at the top of `dictate.py` | right Option |
| Whisper model | `WHISPER` at the top of `dictate.py` | `whisper-large-v3-turbo` |

On a 16 GB M5 MacBook Air, measured on the cleanup step alone:

| Model | Speed | An 82-second dictation | Notes |
|---|---|---|---|
| `gemma4:e2b` | ~115 tokens/s | 2.0s | Rewrites more than asked |
| `gemma4:e4b` | ~68 tokens/s | 3.4s | Faithful; the default |
| `gemma4:12b` | ~16 tokens/s | 14.5s | Faithful, too slow for long dictations |

## How it works

1. `pynput` watches for the hotkey and `sounddevice` records while it is held.
2. `mlx-whisper` transcribes the audio on the GPU.
3. Ollama rewrites the transcript using the prompt in `dictate.py`.
4. The result goes to the clipboard and is pasted with Cmd+V.

Everything is in one file, `dictate.py`.

## Testing the cleanup prompt

```sh
uv run dictate.py --test
```

This runs a list of raw transcripts through the cleanup model and checks each result. When the tool gets one of your dictations wrong, add it to the `cases` list at the bottom of `dictate.py` so a later prompt change cannot quietly break it again.

## Known limits

- Dictating overwrites your clipboard.
- The first dictation after about 30 idle minutes is slower while the model reloads.
- Text appears when you release the key, not as you speak.
- macOS on Apple Silicon only.

## Licence

MIT. See `LICENSE`.
