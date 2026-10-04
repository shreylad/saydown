# Saydown

Hold a key, talk, release. Cleaned-up text is pasted at your cursor, in any app.

A local, private take on Wispr Flow for Apple Silicon Macs. Your audio and text never leave your machine: speech is transcribed by Whisper and tidied by a small language model, both running on your Mac.

## What it does

- **Push to talk.** Hold the right Option key, speak, release.
- **Hands-free.** Double-tap right Option to keep listening, and tap once more to stop.
- **Or click to talk.** A small, mostly see-through pill stays on screen; click it to start, click again to stop.
- **Rewrite in a style.** Turn any dictation into a grammar-corrected, polite, or sugarcoated version, or a style you define yourself.
- **A window to manage it.** The window opens with the app and holds your dictation history and stats, the dictionary, and settings. Closing it only hides it; click its Dock icon or right-click the pill to bring it back.
- **Cleans up speech.** Removes filler words, fixes punctuation, and drops what you took back: "Can you please write down, sorry, please stop" becomes "Please stop."
- **Keeps your voice.** Complete sentences are never deleted, and casual words (gonna, kinda) stay as spoken.
- **Personal dictionary.** Names and jargon you list are spelled your way.
- **On-screen indicator.** The pill turns solid and shows "Listening" and "Writing…" so you know it heard you.
- **Never loses a dictation.** If the cleanup model is unavailable, the raw transcript is pasted instead.

## Install

For an Apple Silicon Mac running macOS 14 or later. Nothing else needs to be installed first.

1. Download `Saydown.dmg` from the [latest release](https://github.com/shreylad/saydown/releases/latest), open it, and drag Saydown into Applications.
2. Open Saydown. macOS will say it cannot verify the app, because it is not signed with an Apple developer account. Go to System Settings → Privacy & Security, scroll down, and click **Open Anyway**.
3. Follow the **Setting up Saydown** card in the window. It has an Open Settings button for each permission (Accessibility and Input Monitoring) and restarts Saydown once both are on. macOS also asks for the Microphone; click Allow.
4. Leave it open while it finishes setting itself up. It downloads the speech model (about 1.6 GB), then the AI engine and model (about 7 GB), showing progress on the same card. You can dictate as soon as the speech model is ready.

Hold right Option, talk, release.

Nothing else needs installing. The AI features (cleanup of self-corrections, Rewrite, Ask, voice commands) run on [Ollama](https://ollama.com): if you already have it, Saydown uses yours; if not, it downloads a private copy into its own folder, checks that it is signed by Ollama, and runs it only while Saydown is open. If a download fails, dictation still works and Saydown tries again the next time it opens.

Your dictations, dictionary and settings are kept in `~/Library/Application Support/Saydown`.

## Run from source

Requirements: an Apple Silicon Mac (16 GB of memory is enough), [uv](https://docs.astral.sh/uv/), and optionally [Ollama](https://ollama.com).


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

### As an app, while developing

```sh
./install-app.sh
```

This builds `Saydown.app` in your Applications folder, so you can start it from Launchpad or Spotlight with no terminal. Its window opens with it and it stays in the Dock; opening it again, or clicking its Dock icon, brings the window back. Quit from the Dock, with ⌘Q, or from the Settings tab. Add it under System Settings → General → Login Items to start it when you log in.

The app has its own identity, so macOS asks for Microphone, Accessibility, and Input Monitoring again, this time for Saydown; grant them, quit, and reopen. It is a small launcher tied to this folder and your `uv` install, not a self-contained download: run the script again if you move the folder. It needs Apple's Command Line Tools for the compiler. The log is at `~/Library/Logs/saydown.log`.

### Building the download

```sh
./install-app.sh --release
```

This builds `dist/Saydown.app` and `dist/Saydown.dmg` with Python and every package inside, so it runs on a Mac with nothing installed. It is ad-hoc signed; signing with an Apple Developer ID and notarising it would remove the "cannot verify" step for people who download it.

## Usage

Click into any text field, then either:

- hold **right Option (⌥)**, say a sentence, and release,
- double-tap **right Option** to record hands-free, then tap it once more to stop, or
- click the pill at the bottom of the screen, speak for as long as you like, and click it again.

Clicking the pill never takes focus from the app you are typing in. Recordings shorter than 0.3 seconds are ignored as accidental taps.

The terminal logs every dictation with the raw transcription beside the cleaned text, so you can see which step made a mistake.

## Rewrite styles

In the window, each dictation has a Rewrite button. Pick a style and the result appears in a card with Copy and Regenerate; nothing is generated until you pick one. Eleven styles are built in (Grammar, Polite, Sugarcoated, Shorter, Bullet points, Summary, Tasks, Email, Daily update, Meeting notes, Thought dump). Add, edit, and delete your own in the Styles tab: a name of up to 30 characters and an instruction of up to 300. Custom styles are saved in `styles.json`.

Every style uses the same prompt, which tells the model to keep your facts, names, and numbers, and to treat the dictation as text to rewrite, never as instructions.

The same panel has a box for a one-off instruction, such as "make it more confident" or "rewrite for my manager", and shows your original words when cleanup changed them.

## Ask

The Ask tab answers questions from your saved dictations, such as "what did I say about pricing?", and lists the dictations it cited. It finds the dictations closest in meaning to your question, so "how much will we charge" finds what you said about pricing. This needs one extra model, `ollama pull embeddinggemma`; without it Ask matches keywords instead. Questions about a time, such as "what did I say yesterday", are not found by date.

## Notes

Record a note, in the window, records without typing anything: stop it from the same button, the pill, or your shortcut, and the transcript is saved in the list. Use it for meetings and thinking out loud, then open Rewrite and pick Meeting notes, Thought dump, Tasks or Summary. A note is one voice with no speaker labels, and the styles see about 30 minutes of speech at most.

## How text is cleaned

Most dictations are tidied by rules, with no model call: filler sounds (um, uh, hmm) and stutters are dropped, and words that are nearly a dictionary term are respelled. The cleanup model runs only when you correct yourself ("no wait", "sorry", "I mean", "actually", "scratch that"), because working out what you meant is the one job the rules cannot do. This saves about a second on a typical dictation.

Pasting uses the clipboard, and whatever you had copied is put back straight afterwards.

While a password field has the keyboard, macOS hides key presses from every app, so a shortcut such as ⌘B cannot be seen. The pill says "Shortcut paused" until you leave the field; clicking the pill still works.

## Voice commands

Off by default; turn it on in Settings. Within two minutes of a dictation, a short remark such as "make that shorter", "fix the grammar", "delete the last sentence" or "undo that" changes the text that was just typed instead of being typed itself. An edit has to point at the text with "that", "this", "it", or a part of it such as "the last sentence"; anything else, and any question, is typed as usual.

With voice commands on, opening with a style name types the rest in that style: "Daily update. Yesterday I fixed the login bug, today I am on billing, no blockers." It works by pressing ⌘Z and pasting the new text, so it assumes you have not typed anything since, and it does nothing useful in apps without undo, such as a terminal.

## Dictionary

Add terms in the Dictionary tab of the window, or edit `dictionary.txt` in `~/Library/Application Support/Saydown`, one term per line:

```
Reykjavík
Siobhan
Kubernetes
```

The terms are given to Whisper as spelling hints and to the cleanup model as corrections for misheard words. The file is read on every dictation, so edits apply without a restart. It is git-ignored because it is personal.

A line written as `heard -> written` always replaces one with the other, whatever the model does:

```
x ai -> xAI
```

## Configuration

| Setting | How | Default |
|---|---|---|
| Cleanup model | Settings tab, or `DICTATE_MODEL` environment variable | `gemma4:e4b` |
| Shortcut | Settings tab: click Change, then press any single modifier key, a function key, or a combination such as ⌘E | right Option |
| Microphone | Settings tab | system default |
| Dictation language | Settings tab | auto-detect |
| Idle pill visibility | Settings tab | 30% |
| Save history | Settings tab | on |
| Whisper model | `WHISPER` at the top of `dictate.py` | `whisper-large-v3-turbo` |

On a 16 GB M5 MacBook Air, measured on the cleanup step alone:

| Model | Speed | An 82-second dictation | Notes |
|---|---|---|---|
| `gemma4:e2b` | ~115 tokens/s | 2.0s | Rewrites more than asked |
| `gemma4:e4b` | ~68 tokens/s | 3.4s | Faithful; the default |
| `gemma4:12b` | ~16 tokens/s | 14.5s | Faithful, too slow for long dictations |

## How it works

1. A native macOS key monitor watches for the hotkey and `sounddevice` records while it is held.
2. `mlx-whisper` transcribes the audio on the GPU.
3. Ollama rewrites the transcript using the prompt in `dictate.py`.
4. The result goes to the clipboard and is pasted with Cmd+V.

The pill is a small AppKit panel drawn with PyObjC. The window is `ui.html` shown in a WebKit view; it talks to the dictation process over a localhost-only server that rejects requests from any other website.

Your history, dictionary and settings are plain files in `~/Library/Application Support/Saydown` (`history.jsonl`, `dictionary.txt`, `settings.json`, `styles.json`).

The logic is in `dictate.py`; the window's layout is in `ui.html`.

## Testing the cleanup prompt

```sh
uv run dictate.py --test
```

This runs a list of raw transcripts through the cleanup model and checks each result. When the tool gets one of your dictations wrong, add it to the `cases` list at the bottom of `dictate.py` so a later prompt change cannot quietly break it again.

## Known limits

- Auto-detect can mistake a short non-English dictation for English; pick the language in Settings if you mostly dictate in one.
- A microphone plugged in after launch appears in Settings only after a restart.
- Dictating overwrites your clipboard.
- The first dictation after about 30 idle minutes is slower while the model reloads.
- Text appears when you release the key, not as you speak.
- macOS on Apple Silicon only.

## Licence

MIT. See `LICENSE`.
