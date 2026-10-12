# Saydown

Hold a key, talk, release. Cleaned-up text is pasted at your cursor, in any app.

A personal dictation app for Apple Silicon Macs. By default, speech is transcribed by Whisper and tidied by a small language model on your Mac. You can also choose paid [OpenAI](#paid-speech-through-openai) or [OpenRouter](#paid-speech-through-openrouter) transcription in Settings; those options upload audio and dictionary hints.

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
4. Leave it open while it finishes setting itself up. It downloads the speech model (about 1.6 GB), then the AI models (about 6 GB), showing progress on the same card. You can dictate as soon as the speech model is ready.

Hold right Option, talk, release.

Nothing else needs installing. The AI features (cleanup of self-corrections, Rewrite, Ask, voice commands) run inside Saydown itself, on a model it downloads from Hugging Face the same way as the speech model. If a download fails, dictation still works and Saydown tries again the next time it opens.

Your dictations, dictionary and settings are kept in `~/Library/Application Support/Saydown`.

## Run from source

Requirements: an Apple Silicon Mac (16 GB of memory is enough), and [uv](https://docs.astral.sh/uv/).

```sh
uv run dictate.py
```

The first run downloads the Whisper model (about 1.6 GB) and the AI models (about 6 GB). uv installs the Python dependencies on its own.

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

The Ask tab answers questions from your saved dictations, such as "what did I say about pricing?", and lists the dictations it cited. It finds the dictations closest in meaning to your question, so "how much will we charge" finds what you said about pricing. This uses a second, small model (0.2 GB) that Saydown downloads with the first; until it arrives Ask matches keywords instead. Questions about a time, such as "what did I say yesterday", are not found by date.

## Notes

Record a note, in the window, records without typing anything: stop it from the same button, the pill, or your shortcut, and the transcript is saved in the list. Use it for meetings and thinking out loud, then open Rewrite and pick Meeting notes, Thought dump, Tasks or Summary. A note is one voice with no speaker labels, and the styles see about 30 minutes of speech at most.

## How text is cleaned

Most dictations are tidied by rules, with no model call: filler sounds (um, uh, hmm) and stutters are dropped, and words that are nearly a dictionary term are respelled. The cleanup model runs only when you correct yourself ("no wait", "sorry", "I mean", "actually", "scratch that"), because working out what you meant is the one job the rules cannot do. This saves about a second on a typical dictation.

Complete courtesy sentences such as "Sorry I am late." and "Sorry for the delay."
stay on the rule path. Ambiguous apologies and any additional correction cue
still use the model. With voice commands enabled, ordinary references such as
"Please send that file to Alex" are typed directly; an editing cue is required
before the command classifier runs.

For lower local transcription latency when dictating in English, choose
**Settings → Dictation language → English**. Auto-detect runs an extra language
detection pass. Keep Auto-detect if you need to switch between languages.
Recording already runs separately from transcription, so you can start the next
recording while the previous dictation is processed. Two simultaneous Whisper
model processes compete for the GPU and can increase individual request latency.

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
| Shortcut | Settings tab: click Change, then press any single modifier key, a function key, or a combination such as ⌘E | right Option |
| Microphone | Settings tab | system default |
| Dictation language | Settings tab | auto-detect |
| Transcription service | Settings → Speech-to-text | Local Whisper (existing OpenAI opt-ins are preserved) |
| API speech model | Settings → Speech-to-text | GPT Transcribe |
| OpenAI API key | Settings → Speech-to-text; stored in macOS Keychain | none |
| OpenRouter speech model | Settings → Speech-to-text | Whisper Large V3 Turbo |
| OpenRouter API key | Settings → Speech-to-text; separate macOS Keychain entry | none |
| Idle pill visibility | Settings tab | 30% |
| Save history | Settings tab | on |
| Whisper model | `WHISPER` at the top of `dictate.py` | `whisper-large-v3-turbo` |

On a 16 GB M5 MacBook Air, scored with `bench.py` (42 checks of cleanup, rewriting, editing, voice commands and Ask):

| Model | Score | A typical cleanup | Notes |
|---|---|---|---|
| `gemma-4-e4b-it-5bit` | 42/42 | 0.6s | The default; 5.7 GB |
| `gemma-4-e4b-it-4bit` | 40/42 | 0.6s | 4.8 GB; missed one self-correction and one voice command |

## Paid speech through OpenAI

In your existing Saydown window, open **Settings → Speech-to-text**:

1. Select **OpenAI · paid API**.
2. Choose **GPT Transcribe**, **GPT-4o Mini Transcribe**, **GPT-4o Transcribe**, or **Whisper API**.
3. Paste your OpenAI API key and click **Save key**. It goes directly to macOS Keychain, not settings, logs, or process arguments. The app never sends the saved key back to the settings page.
4. Dictate normally with your existing shortcut or pill. Changes apply to the next dictation without a restart.

API use is billed to your OpenAI API account separately from a ChatGPT subscription. OpenAI receives your recording and dictionary hints. Cleanup, Ask, rewriting, and history stay local.

Select **Local Whisper** to keep speech entirely on your Mac, even with a saved API key. **Remove key** deletes the Saydown key from Keychain. Saving a key alone does not switch a locally configured app to OpenAI.

If the key is missing or rejected, the network fails, a rate limit is reached, or a recording exceeds the upload limit, local Whisper takes over. The Settings card reports the service actually used and any fallback. No automatic paid retry is made. API requests have a 30-second timeout; a fallback after a network problem can therefore take longer than a normal dictation.

Existing key-only OpenAI configurations retain `gpt-4o-mini-transcribe` until you make a new selection. New setups should prefer `gpt-transcribe`; OpenAI has scheduled the three legacy models for removal on February 26, 2027. See [file transcription](https://developers.openai.com/api/docs/guides/speech-to-text) and [deprecations](https://developers.openai.com/api/docs/deprecations).

For a focused offline regression suite (with Saydown's Python dependencies installed):

```sh
python -m unittest test_speech -v
```

The suite uses temporary settings and mocked API/Keychain calls; it does not record, upload audio, use credits, or change your saved key.

## Paid speech through OpenRouter

Open **Settings → Speech-to-text**, select **OpenRouter · paid API**, choose a
speech model, paste your **OpenRouter API key**, and click **Save key**.

The available models are Whisper Large V3 Turbo, Whisper Large V3, GPT Transcribe,
GPT-4o Mini Transcribe, and GPT-4o Transcribe. Availability and price depend on
your OpenRouter account and its current catalog. Requests go to OpenRouter's
`/api/v1/audio/transcriptions` endpoint with the selected namespaced model ID.
See [OpenRouter's transcription documentation](https://openrouter.ai/blog/tutorials/transcription-on-openrouter/).

OpenRouter and its upstream transcription provider receive your recording and
dictionary hints. Usage is billed to your OpenRouter account. The key is stored
as `saydown-openrouter` in macOS Keychain, separately from `saydown-openai`.
Selecting or removing one provider's key does not change the other's key or
model selection. Local Whisper makes no API request even when keys are saved.
Saydown reads keys directly through macOS Keychain in its own process and reuses
them in memory while running; it does not invoke the `security` command to read
secrets. Saving or removing a key immediately updates that session cache. If
Keychain denies access, Saydown falls back locally without repeating the access
prompt on every dictation. Save the key again in Settings or restart Saydown to
retry. Settings labels old transcription results as **Last dictation**, and a
successful key save clears the earlier failure message.
The same local fallback applies if the key is missing, the request fails, or
the upload exceeds the limit. Cleanup, Ask, rewriting, and history remain local.

## How it works

1. A native macOS key monitor watches for the hotkey and `sounddevice` records while it is held.
2. `mlx-whisper` transcribes the audio on the GPU.
3. The cleanup model rewrites the transcript using the prompt in `dictate.py`.
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
