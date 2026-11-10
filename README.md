# Saydown

Hold a key, talk, release. Cleaned-up text is pasted at your cursor, in any app.

A personal dictation app for Apple Silicon Macs. By default, speech is transcribed by Whisper and tidied by a small language model on your Mac. You can also choose paid [OpenAI](#paid-speech-through-openai), [OpenRouter](#paid-speech-through-openrouter) or [DeepInfra](#paid-speech-and-ai-features-through-deepinfra) transcription in Settings; those options upload audio and dictionary hints. Separately, the AI features can [run on a paid API](#ai-features-on-a-paid-api), which uploads the text of your dictations.

## Stack

| Layer | What |
|---|---|
| Language | Python 3.11 or later (the app bundles 3.12), a single-file [uv script](https://docs.astral.sh/uv/guides/scripts/) with inline dependencies |
| Speech to text | [mlx-whisper](https://github.com/ml-explore/mlx-examples) running Whisper Large V3 Turbo on the Apple GPU |
| Text cleanup, Rewrite, Ask | [mlx-lm](https://github.com/ml-explore/mlx-lm) running Gemma 4 E4B, plus mlx-embeddings for search by meaning |
| Hotkey, pill, paste | AppKit and Quartz through PyObjC |
| Window | Plain HTML and JavaScript in `ui.html`, shown in a WebKit view, talking to Python over a localhost-only server |
| Packaging | `install-app.sh` builds a signed `.app` and `.dmg` with Python bundled; `launcher.m` is the native entry point |
| Tests | `unittest`; `bench_text_model.py` scores a text model on 42 real dictations |

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

For an Apple Silicon Mac running macOS 14 or later. Nothing else needs to be installed first. On Windows? See [Saydown for Windows](https://github.com/shreylad/saydown-windows).

1. Download `Saydown.dmg` from the [latest release](https://github.com/shreylad/saydown/releases/latest), open it, and drag Saydown into Applications.
2. Open Saydown. If macOS says it cannot verify the app, the download was built without an Apple developer account: go to System Settings → Privacy & Security, scroll down, and click **Open Anyway**.
3. Follow the **Setting up Saydown** card in the window. It has an Open Settings button for each permission (Accessibility and Input Monitoring) and restarts Saydown once both are on. macOS also asks for the Microphone; click Allow.
4. Leave it open while it finishes setting itself up. It downloads the speech model (about 1.6 GB), showing progress on the same card, and you can dictate as soon as it is ready. The AI features (Rewrite, Ask, voice commands, fixing self-corrections) are optional: turn them on in Settings or on the setup card, and Saydown downloads their models (about 6 GB).

Hold right Option, talk, release.

Nothing else needs installing. The AI features (cleanup of self-corrections, Rewrite, Ask, voice commands) run inside Saydown itself, on a model it downloads from Hugging Face the same way as the speech model. If a download fails, dictation still works and Saydown tries again the next time it opens.

Your dictations, dictionary and settings are kept in `~/Library/Application Support/Saydown`. Delete a single dictation from its row, export them all to a Markdown file in Downloads, or set Settings → Keep history to delete old ones automatically. The log records timings and word counts, not what you said, unless you turn on Write dictations to the log.

## Run from source

Requirements: an Apple Silicon Mac (16 GB of memory is enough), and [uv](https://docs.astral.sh/uv/).

```sh
uv run dictate.py
```

The first run downloads the Whisper model (about 1.6 GB); the AI models (about 6 GB) download once you turn on AI features in Settings. uv installs the Python dependencies on its own. Run this way, the Saydown window opens as a page in your browser.

macOS will ask the app you launch it from (Terminal, iTerm, ...) for three permissions under System Settings → Privacy & Security. Grant all three, then start it again:

| Permission | Needed for |
|---|---|
| Microphone | Recording your voice |
| Input Monitoring | Seeing the hotkey |
| Accessibility | Pasting at your cursor |

### Updating the installed app

```sh
./install-app.sh
```

This installs the code in this folder for the `Saydown.app` already in your Applications folder; restart Saydown to use it. Nothing is rebuilt or re-signed, so macOS does not ask for permissions again. It works only with an app you built yourself with `SAYDOWN_LOCAL_UPDATES=1 ./install-app.sh --release`; the download ignores such updates, because they are code that any program on the Mac could replace. With no app installed yet, build one that way first (below) and drag it to Applications. The log is at `~/Library/Logs/saydown.log`.

### Building the download

```sh
./install-app.sh --release
```

This builds `dist/Saydown.app` and `dist/Saydown.dmg` with Python and every package inside, so it runs on a Mac with nothing installed. It is ad-hoc signed unless `SAYDOWN_SIGN="Developer ID Application: Name (TEAMID)"` is set, in which case the app is signed with the hardened runtime and the DMG notarised and stapled, which removes the "cannot verify" step for people who download it. That needs the Apple Developer Program and a one-time `xcrun notarytool store-credentials saydown --apple-id <Apple ID> --team-id <TEAMID> --password <app-specific password>`.

## Usage

Click into any text field, then either:

- hold **right Option (⌥)**, say a sentence, and release,
- double-tap **right Option** to record hands-free, then tap it once more to stop, or
- click the pill at the bottom of the screen, speak for as long as you like, and click it again.

Clicking the pill never takes focus from the app you are typing in. Recordings shorter than 0.3 seconds are ignored as accidental taps.

Spoken lists are formatted by rule, with no model call. "First, review the report. Second, send the invoice" becomes two numbered lines, and "Here is my agenda: cook, work on the app, mow the lawn" becomes three bullets. The exact rules, and what they deliberately leave alone, are in [docs/list-formatting.md](docs/list-formatting.md).

If Accessibility is unavailable at delivery time, Saydown keeps the completed dictation on the clipboard for manual paste and shows a message. History and Copy buttons remain available during permission setup. A local speech-model loading failure is shown in Setup, and the worker remains available for an API provider. Incomplete or corrupt history lines are skipped when displaying history; the original history file is not modified.

## Updating this personal installation

`./install-app.sh` stores the Python and UI files beside your data as a versioned runtime, without rebuilding or re-signing the app, so macOS does not ask for permissions again. Only an app built with `SAYDOWN_LOCAL_UPDATES=1` uses it. How the runtime pointer, manifests and rollback work is in [docs/updating.md](docs/updating.md).

## Rewrite styles

In the window, each dictation has a Rewrite button. Pick a style and the result appears in a card with Copy and Regenerate; nothing is generated until you pick one. Grammar, Shorter, Bullet points and your own styles show first; More shows the rest. Eleven styles are built in (Grammar, Polite, Sugarcoated, Shorter, Bullet points, Summary, Tasks, Email, Daily update, Meeting notes, Thought dump). Add, edit, and delete your own in the Styles tab: a name of up to 30 characters and an instruction of up to 300. Custom styles are saved in `styles.json`.

Every style uses the same prompt, which tells the model to keep your facts, names, and numbers, and to treat the dictation as text to rewrite, never as instructions.

The same panel has a box for a one-off instruction, such as "make it more confident" or "rewrite for my manager", and shows your original words when cleanup changed them.

## Ask

The Ask tab answers questions from your saved dictations, such as "what did I say about pricing?", and lists the dictations it cited. It finds the dictations closest in meaning to your question, so "how much will we charge" finds what you said about pricing. This uses a second, small model (0.2 GB) that Saydown downloads with the first; until it arrives Ask matches keywords instead. Questions about a time, such as "what did I say yesterday", "last week" or "on Monday", look only at the dictations from those days.

## Notes

Record a note, in the window, records without typing anything: stop it from the same button, the pill, or your shortcut, and the transcript is saved in the list. It is a personal voice note for thinking out loud, not a meeting transcript: open Rewrite and pick Meeting notes, Thought dump, Tasks or Summary. A note is one voice with no speaker labels, and the styles see about 30 minutes of speech at most.

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

Pasting uses the clipboard, and whatever you had copied is put back straight afterwards. The dictation is marked as passing through, so clipboard managers that follow the [nspasteboard.org](http://nspasteboard.org) convention leave it out of their history.

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

The terms are given to Whisper as spelling hints and to the cleanup model as corrections for misheard words. A word that is nearly a term of five letters or more is respelled ("Kubernetis" becomes Kubernetes), but an ordinary word one letter away, or the term with an ending such as -s or -ing, is left as you said it. The file is read on every dictation, so edits apply without a restart. It is git-ignored because it is personal.

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

On a 16 GB M5 MacBook Air, scored with `bench_text_model.py` (42 checks of cleanup, rewriting, editing, voice commands and Ask):

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

API use is billed to your OpenAI API account separately from a ChatGPT subscription. OpenAI receives your recording and dictionary hints. Cleanup, Ask, rewriting, and history stay local unless you also [run the AI features on a paid API](#ai-features-on-a-paid-api).

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
the upload exceeds the limit. Cleanup, Ask, rewriting, and history remain local
unless you also [run the AI features on a paid API](#ai-features-on-a-paid-api).

## Paid speech and AI features through DeepInfra

Select **DeepInfra · paid API** under **Settings → Speech-to-text** and save a
DeepInfra API key. It offers one speech model, Whisper Large V3 Turbo. DeepInfra
receives your recording and dictionary hints, usage is billed to your DeepInfra
account, and the key is stored as `saydown-deepinfra` in macOS Keychain. The
same local fallback applies.

## AI features on a paid API

**Settings → Run AI features on** moves cleanup of self-corrections, Rewrite,
Ask and voice commands from the model on your Mac to a larger Gemma model at
OpenRouter or DeepInfra, using the key saved for that provider under
Speech-to-text. This frees about 5 GB of memory. The provider then receives the
text involved: the sentences being cleaned, the dictation being rewritten, and
for Ask the saved dictations closest to your question. History itself stays in
its file on your Mac, and search by meaning still runs locally. If a cleanup
request fails, the text is tidied by rule instead. Choose **This Mac** to keep
all text local.

## Project layout

| File | Role |
|---|---|
| `dictate.py` | The app: hotkey, recording, transcription, cleanup, Rewrite, Ask, settings, and the localhost server behind the window |
| `ui.html` | The window: history, Ask, dictionary, styles, settings |
| `launcher.m` | Native launcher compiled into the `.app`; starts the bundled Python |
| `runtime_bootstrap.py` | Picks the current runtime version at launch and falls back to the bundled one |
| `update_app.py` | Installs a new runtime version into the app (see [docs/updating.md](docs/updating.md)) |
| `install-app.sh` | Builds the `.app` and `.dmg`, or updates an installed app |
| `test_speech.py`, `test_runtime.py`, `test_recorder.py`, `test_ui.js` | Unit tests for cleanup rules and API fallbacks, the runtime updater, the recorder helper, and the window |
| `bench_text_model.py` | Scores a text model (local or cloud) on 42 real dictations |
| `bench_dictation.py` | End-to-end check of local Whisper plus formatting on synthetic speech |
| `docs/` | Longer notes: list formatting rules, the update mechanism |

## How it works

1. A native macOS key monitor watches for the hotkey and `sounddevice` records while it is held.
2. `mlx-whisper` transcribes the audio on the GPU.
3. Rules tidy the transcript; only sentences where you corrected yourself go to the cleanup model, with the prompt in `dictate.py`.
4. The result goes to the clipboard and is pasted with Cmd+V.

The pill is a small AppKit panel drawn with PyObjC. The window is `ui.html` shown in a WebKit view; it talks to the dictation process over a localhost-only server that rejects requests from any other website.

Your history, dictionary and settings are plain files in `~/Library/Application Support/Saydown` (`history.jsonl`, `dictionary.txt`, `settings.json`, `styles.json`).

The logic is in `dictate.py`; the window's layout is in `ui.html`.

## Testing

```sh
uv run dictate.py --test
uv run --with-requirements <(uv export --script dictate.py) bench_text_model.py mlx-community/gemma-4-e4b-it-5bit
```

The first checks the settings and the text rules in about a second, with no model. The second runs raw transcripts, rewrites, edits, voice commands and Ask through the text model and prints what passed. When the tool gets one of your dictations wrong, add it to the cleanup list in `bench_text_model.py` so a later prompt change cannot quietly break it again.

## Known limits

- Auto-detect can mistake a short non-English dictation for English; pick the language in Settings if you mostly dictate in one.
- A microphone plugged in after launch appears in Settings only after a restart.
- Dictating passes through your clipboard: what you had copied is put back 0.4 seconds later, so an app slower than that to paste gets the old contents.
- The first self-correction, Rewrite or Ask after four idle hours is about 3 seconds slower while the text model reloads.
- Text appears when you release the key, not as you speak.
- macOS on Apple Silicon only. For Windows, use [Saydown for Windows](https://github.com/shreylad/saydown-windows).

## Licence

GPL-3.0. See `LICENSE`.
