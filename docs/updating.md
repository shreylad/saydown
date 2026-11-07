# Updating the installed app without re-signing

`./install-app.sh` calls `update_app.py` rather than rebuilding the app. Python/UI updates are stored as versioned files under `~/Library/Application Support/Saydown/Runtime`, selected by an atomic pointer, without changing or re-signing the app bundle. The bundled Python and native launcher remain in the app. Changing either of those, or `runtime_bootstrap.py`, needs a new `--release` build and can require renewed permissions; the updater refuses to run when the bootstrap differs.

Each runtime has a manifest of file hashes and must match the bundled fallback version. An incomplete, corrupt, or incompatible runtime falls back to the bundled implementation. Updates retain the previous runtimes, so pointing `Runtime/current.json` back at one undoes an update; an installation failure restores the previous pointer. Startup never runs `tccutil reset` automatically. Permission grants remain under the user's control; if an old grant is stale, remove and re-add Saydown manually in System Settings.

Regression checks: `python -B -m unittest test_speech test_runtime -q` with the app's bundled Python and site-packages. Keep `-B`: without it Python writes cache files into the app, which invalidates its signature and blocks the next update. `bench_dictation.py --output /path/to/report-directory` exercises local Whisper and formatting on synthetic English TTS fixtures, using the cached model and isolated settings. It does not use the microphone or paid APIs. Synthetic results do not establish accuracy for accents, background noise, or arbitrary real recordings.

The terminal logs every dictation with the raw transcription beside the cleaned text, so you can see which step made a mistake.
