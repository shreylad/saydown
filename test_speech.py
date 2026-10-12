"""Offline regression tests; no real Keychain, microphone, models, or API calls.

Run with the app's Python and site-packages, or `uv run --script dictate.py`
dependencies installed in the current interpreter: python -m unittest test_speech -v.
"""
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
import wave
from pathlib import Path
from unittest.mock import Mock, patch


class SpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        spec = importlib.util.spec_from_file_location("saydown_test", Path(__file__).with_name("dictate.py"))
        cls.app = importlib.util.module_from_spec(spec)
        with patch.object(Path, "home", return_value=Path(cls.temp.name)):
            spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.a = self.app
        self.a.SETTINGS.write_text(json.dumps(self.a.DEFAULTS))
        self.a._cached[:] = [None, None]
        self.a.speech_status.update(engine="", message="")
        self.a._key_cache.clear()
        for cached in self.a._key_status.values():
            cached[:] = [0.0, False]
        self.audio = self.a.np.ones(1600, dtype=self.a.np.float32) * 0.1
        self.local = Mock(return_value={"text": "local words", "segments": []})

    def config(self, **values):
        self.a.SETTINGS.write_text(json.dumps(self.a.DEFAULTS | values))
        self.a._cached[:] = [None, None]

    def test_local_never_reads_key_or_calls_api(self):
        with patch.object(self.a, "cloud_key") as key, patch.object(self.a, "hear") as hear:
            result, engine = self.a.transcribe(self.audio, "en", self.local)
        key.assert_not_called()
        hear.assert_not_called()
        self.assertEqual((result["text"], engine), ("local words", "local"))

    def test_every_model_routes_selected_model(self):
        for model in self.a.SPEECH_MODELS:
            with self.subTest(model=model):
                self.config(speech_provider="openai", speech_model=model)
                with patch.object(self.a, "cloud_key", return_value="test-secret"), patch.object(self.a, "hear", return_value="API words") as hear:
                    result, engine = self.a.transcribe(self.audio, "en", self.local)
                self.assertEqual(engine, model)
                self.assertEqual(result["text"], "API words")
                self.assertEqual(hear.call_args.kwargs["model"], model)
        self.local.assert_not_called()

    def test_missing_key_falls_back(self):
        self.config(speech_provider="openai")
        with patch.object(self.a, "cloud_key", return_value=""), patch.object(self.a, "hear") as hear:
            self.assertEqual(self.a.transcribe(self.audio, None, self.local)[1], "local")
        hear.assert_not_called()
        self.assertIn("No OpenAI key", self.a.speech_status["message"])

    def test_network_auth_and_quota_failures_do_not_leak_errors(self):
        self.config(speech_provider="openai")
        for error in [TimeoutError("secret"), urllib.error.HTTPError("url", 401, "secret", {}, None),
                      urllib.error.HTTPError("url", 429, "secret", {}, None)]:
            with patch.object(self.a, "cloud_key", return_value="secret"), patch.object(self.a, "hear", side_effect=error):
                self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            self.assertNotIn("secret", self.a.speech_status["message"])

    def test_validate_models_and_preserve_other_settings(self):
        with patch.object(self.a, "microphones", return_value=[]):
            for model in self.a.SPEECH_MODELS:
                result = self.a.valid_settings({"speech_provider": "openai", "speech_model": model})
                self.assertEqual(result["hotkey"], self.a.DEFAULT_HOTKEY)
                self.assertEqual(result["speech_model"], model)
            for body in [{"speech_provider": "unknown"}, {"speech_model": "unknown"}, {"speech_model": []}]:
                with self.assertRaises(ValueError):
                    self.a.valid_settings(body)

    def test_existing_key_opt_in_migrates_but_explicit_local_wins(self):
        self.a.SETTINGS.write_text('{"language":"en"}')
        self.a._cached[:] = [None, None]
        with patch.object(self.a, "cloud_key_saved", return_value=True):
            self.assertEqual(self.a.settings()["speech_model"], "gpt-4o-mini-transcribe")
            self.assertEqual(self.a.settings()["speech_provider"], "openai")
            self.config(speech_provider="local")
            self.assertEqual(self.a.settings()["speech_provider"], "local")

    def test_multipart_contains_model_language_dictionary_and_real_wav(self):
        for model in self.a.SPEECH_MODELS:
            response = io.BytesIO(b'{"text":"hello"}')
            with patch.object(self.a.urllib.request, "urlopen", return_value=response) as upload:
                self.assertEqual(self.a.hear(self.audio, "test-key", "en", "Saydown", model), "hello")
            request = upload.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.openai.com/v1/audio/transcriptions")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            body = request.data
            self.assertIn(model.encode(), body)
            self.assertIn(b'Saydown', body)
            self.assertIn(b'name="languages[]"' if model == "gpt-transcribe" else b'name="language"', body)
            wav_bytes = body.split(b'Content-Type: audio/wav\r\n\r\n')[1].split(b'\r\n--')[0]
            with wave.open(io.BytesIO(wav_bytes)) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
                self.assertEqual(wav.getnframes(), len(self.audio))

    def test_bad_response_and_large_upload_fail_before_fallback(self):
        with patch.object(self.a.urllib.request, "urlopen", return_value=io.BytesIO(b'{"text":null}')):
            with self.assertRaises(ValueError):
                self.a.hear(self.audio, "key")
        with patch.object(self.a.urllib.request, "urlopen") as upload:
            with self.assertRaises(ValueError):
                self.a.hear(self.a.np.zeros(12_000_001), "key")
            upload.assert_not_called()

    def handler(self, path, body=None, origin=None, content_type="application/json"):
        cls = self.a.make_handler(Mock(), Mock(), Mock(), lambda: False, Mock(), Mock())
        handler = cls.__new__(cls)
        encoded = json.dumps(body or {}).encode()
        handler.path = path
        handler.headers = {"Host": self.a.HOST, "Content-Type": content_type, "Content-Length": str(len(encoded))}
        if origin:
            handler.headers["Origin"] = origin
        handler.rfile = io.BytesIO(encoded)
        handler.reply = Mock()
        return handler

    def test_key_route_does_not_enable_cloud_or_echo_key(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"})
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_called_once_with("fake-test-key-123456789", "openai")
        self.assertEqual(self.a.settings()["speech_provider"], "local")
        self.assertNotIn("fake-test", str(handler.reply.call_args))

    def test_blank_key_cannot_accidentally_remove_saved_key(self):
        for key in ["", "   ", None]:
            handler = self.handler("/api/speech-key", {"key": key})
            with patch.object(self.a, "save_cloud_key") as save:
                handler.do_POST()
            save.assert_not_called()
            self.assertEqual(handler.reply.call_args.kwargs["status"], 400)

    def test_remove_key_is_explicit(self):
        handler = self.handler("/api/speech-key", {"key": "", "remove": True})
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_called_once_with("", "openai")

    def test_cross_origin_key_write_is_blocked(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"}, "https://example.org")
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_not_called()
        self.assertEqual(handler.reply.call_args.kwargs["status"], 403)

    def test_plain_form_key_write_is_blocked(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"}, content_type="text/plain")
        with patch.object(self.a, "save_cloud_key") as save:
            handler.do_POST()
        save.assert_not_called()
        self.assertEqual(handler.reply.call_args.kwargs["status"], 400)

    def test_keychain_error_is_visible(self):
        handler = self.handler("/api/speech-key", {"key": "fake-test-key-123456789"})
        with patch.object(self.a, "save_cloud_key", side_effect=OSError("Keychain access was denied.")):
            handler.do_POST()
        self.assertEqual(handler.reply.call_args.args[0]["error"], "Keychain access was denied.")

    def test_state_contains_key_presence_never_secret(self):
        handler = self.handler("/api/state")
        with patch.object(self.a, "cloud_key_saved", return_value=True), patch.object(self.a, "cloud_key") as key, patch.object(self.a, "microphones", return_value=[]):
            handler.do_GET()
        key.assert_not_called()
        self.assertTrue(handler.reply.call_args.args[0]["speech"]["key_saved"])
        self.assertNotIn("api_key", handler.reply.call_args.args[0]["settings"])

    def test_openrouter_routes_selected_model_and_correct_key(self):
        for model in self.a.OPENROUTER_MODELS:
            self.config(speech_provider="openrouter", openrouter_model=model)
            with patch.object(self.a, "cloud_key", return_value="router-secret") as key, patch.object(self.a, "hear", return_value="Router words") as hear:
                result, engine = self.a.transcribe(self.audio, "en", self.local)
            key.assert_called_once_with("openrouter")
            self.assertEqual(result["text"], "Router words")
            self.assertEqual(engine, f"openrouter:{model}")
            self.assertEqual(hear.call_args.kwargs, {"model": model, "provider": "openrouter"})
            self.assertIn("via OpenRouter", self.a.speech_status["message"])
        self.local.assert_not_called()

    def test_openrouter_request_uses_router_url_and_namespaced_models(self):
        for model in self.a.OPENROUTER_MODELS:
            with self.subTest(model=model), patch.object(self.a.urllib.request, "urlopen", return_value=io.BytesIO(b'{"text":"Router words"}')) as upload:
                self.assertEqual(self.a.hear(self.audio, "router-secret", "en", "Saydown", model, "openrouter"), "Router words")
                request = upload.call_args.args[0]
                self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/audio/transcriptions")
                self.assertEqual(request.get_header("Authorization"), "Bearer router-secret")
                self.assertIn(model.encode(), request.data)
                self.assertIn(b'name="language"', request.data)
                self.assertNotIn(b'name="languages[]"', request.data)
                self.assertIn(b'filename="speech.wav"', request.data)

    def test_cross_provider_models_are_rejected_before_request(self):
        with patch.object(self.a.urllib.request, "urlopen") as upload:
            for provider, model in [("openai", "openai/whisper-large-v3-turbo"), ("openrouter", "gpt-transcribe"),
                                    ("unknown", "gpt-transcribe")]:
                with self.assertRaises(ValueError):
                    self.a.hear(self.audio, "secret", model=model, provider=provider)
            upload.assert_not_called()

    def test_provider_keys_use_separate_keychain_services(self):
        for provider in ("openai", "openrouter"):
            with patch.object(self.a, "keychain_lookup", return_value="dummy") as lookup:
                self.a.cloud_key(provider)
                lookup.assert_called_once_with(provider, read_secret=True)
        self.assertNotEqual(self.a.KEYCHAINS["openai"], self.a.KEYCHAINS["openrouter"])

    def test_repeated_dictations_read_key_only_once_without_security_command(self):
        with patch.object(self.a, "keychain_lookup", return_value="dummy-secret") as lookup, patch.object(self.a.subprocess, "run") as subprocess_run:
            for _ in range(5):
                self.assertEqual(self.a.cloud_key("openrouter"), "dummy-secret")
            lookup.assert_called_once_with("openrouter", read_secret=True)
            subprocess_run.assert_not_called()

    def test_keychain_denial_is_not_reported_as_missing_or_prompted_repeatedly(self):
        with patch.object(self.a, "keychain_lookup", side_effect=PermissionError("denied")) as lookup:
            for _ in range(3):
                with self.assertRaises(PermissionError):
                    self.a.cloud_key("openrouter")
            lookup.assert_called_once()
        self.config(speech_provider="openrouter")
        with patch.object(self.a, "hear") as hear:
            self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            hear.assert_not_called()
        self.assertIn("Keychain access was denied", self.a.speech_status["message"])
        self.assertNotIn("No OpenRouter key", self.a.speech_status["message"])

    def test_save_replaces_cached_denial_and_clears_stale_failure(self):
        self.a._key_cache["openrouter"] = None
        self.a._key_cache["openai"] = "other-secret"
        self.a.speech_status.update(engine="local", message="No OpenRouter key is saved.")
        with patch.object(self.a, "_write_cloud_key") as write, patch.object(self.a, "keychain_lookup") as lookup:
            self.a.save_cloud_key("new-dummy-secret", "openrouter")
            write.assert_called_once_with("new-dummy-secret", "openrouter")
            self.assertEqual(self.a.cloud_key("openrouter"), "new-dummy-secret")
            self.assertEqual(self.a.cloud_key("openai"), "other-secret")
            lookup.assert_not_called()
        self.assertEqual(self.a.speech_status, {"engine": "", "message": ""})

    def test_removal_clears_cached_secret(self):
        self.a._key_cache["openrouter"] = "old-dummy-secret"
        with patch.object(self.a, "_write_cloud_key"), patch.object(self.a, "keychain_lookup") as lookup:
            self.a.save_cloud_key("", "openrouter")
            self.assertEqual(self.a.cloud_key("openrouter"), "")
            lookup.assert_not_called()

    def test_failed_save_does_not_replace_cached_secret(self):
        self.a._key_cache["openrouter"] = "old-dummy-secret"
        with patch.object(self.a, "_write_cloud_key", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                self.a.save_cloud_key("new-dummy-secret", "openrouter")
        self.assertEqual(self.a.cloud_key("openrouter"), "old-dummy-secret")

    def test_metadata_checks_never_read_secret_or_spawn_security(self):
        with patch.object(self.a, "keychain_lookup", return_value=True) as lookup, patch.object(self.a.subprocess, "run") as run:
            self.assertTrue(self.a.cloud_key_saved("openrouter"))
            self.assertTrue(self.a.cloud_key_saved("openrouter"))
            lookup.assert_called_once_with("openrouter")
            run.assert_not_called()

    def test_native_lookup_frees_secret_buffer_and_matches_correct_service(self):
        secret = self.a.ctypes.create_string_buffer(b"dummy-secret")
        security = Mock()
        def found(*args):
            self.assertEqual(args[2], b"saydown-openrouter")
            args[5]._obj.value = len(b"dummy-secret")
            args[6]._obj.value = self.a.ctypes.addressof(secret)
            return 0
        security.SecKeychainFindGenericPassword.side_effect = found
        with patch.object(self.a.ctypes, "CDLL", return_value=security):
            self.assertEqual(self.a.keychain_lookup("openrouter", True), "dummy-secret")
        security.SecKeychainItemFreeContent.assert_called_once()
        self.assertEqual(security.SecKeychainItemFreeContent.call_args.args[1].value, self.a.ctypes.addressof(secret))

    def test_native_metadata_lookup_requests_no_password_buffer(self):
        security = Mock()
        security.SecKeychainFindGenericPassword.return_value = 0
        with patch.object(self.a.ctypes, "CDLL", return_value=security):
            self.assertTrue(self.a.keychain_lookup("openrouter"))
        self.assertEqual(security.SecKeychainFindGenericPassword.call_args.args[5:], (None, None, None))
        security.SecKeychainItemFreeContent.assert_not_called()

    def test_native_missing_and_denied_results_are_distinct(self):
        security = Mock()
        with patch.object(self.a.ctypes, "CDLL", return_value=security):
            security.SecKeychainFindGenericPassword.return_value = -25300
            self.assertEqual(self.a.keychain_lookup("openrouter", True), "")
            self.assertFalse(self.a.keychain_lookup("openrouter"))
            security.SecKeychainFindGenericPassword.return_value = -25293
            with self.assertRaises(PermissionError):
                self.a.keychain_lookup("openrouter", True)

    def test_complete_courtesy_sentences_skip_model_but_keep_text(self):
        samples = ["Sorry I am late. Please send the report before the meeting.",
                   "Sorry, I'm late. I'll be there in ten minutes.",
                   "Sorry for the delay. Thanks for waiting.",
                   "Thanks for waiting. Sorry I'm late."]
        with patch.object(self.a, "clean") as clean:
            for text in samples:
                self.assertEqual(self.a.polish(text), text)
            clean.assert_not_called()

    def test_corrections_still_use_the_model_including_after_apologies(self):
        samples = ["Send it Tuesday, sorry, Wednesday.",
                   "Sorry I am late. Send it Tuesday, no wait, Wednesday.",
                   "Sorry, please cancel the flight.",
                   "I think we should, actually scratch that, send the invoice.",
                   "Sorry I'm late, no wait, I mean early.",
                   "Sorry I am late"]
        for text in samples:
            with patch.object(self.a, "clean", return_value="Corrected text.") as clean:
                self.assertEqual(self.a.polish(text), "Corrected text.")
                clean.assert_called_once_with(text)

    def test_ordinary_references_do_not_call_command_model(self):
        samples = ["Please send that file to Alex.", "I will send it tomorrow.",
                   "This is the latest report.", "I told them that already.",
                   "That works for me.", "I like this."]
        with patch.object(self.a, "llm") as llm:
            for text in samples:
                self.assertEqual(self.a.command(text, "Previous message."), "TYPE")
            llm.assert_not_called()

    def test_actual_voice_commands_still_reach_classifier(self):
        samples = ["Make that shorter", "Fix the grammar", "Undo that", "Scratch that", "Never mind",
                   "Delete the last sentence", "Make it more confident", "Turn that into bullet points",
                   "Translate that to Spanish", "Delete that", "Take that back", "That is too long"]
        for text in samples:
            with patch.object(self.a, "llm", return_value="EDIT") as llm:
                self.a.command(text, "Previous message.")
                llm.assert_called_once()

    def test_user_agenda_becomes_four_lines_with_introduction(self):
        introduction = "Thank you so much for reaching me out. I am writing to see that like how is everything going."
        raw = (introduction + " First our agenda is doing apple tree, second agenda is doing green tree, "
               "second third agenda is cleaning the grass and fourth agenda is making a barn house.")
        expected = (introduction + "\n\n1. Our agenda is doing apple tree\n2. Agenda is doing green tree\n"
                    "3. Agenda is cleaning the grass\n4. Agenda is making a barn house.")
        with patch.object(self.a, "llm") as llm:
            self.assertEqual(self.a.polish(raw), expected)
            llm.assert_not_called()

    def test_spoken_lists_with_punctuation_or_without_it(self):
        samples = {
            "First, review the report. Second, send the invoice.": "1. Review the report.\n2. Send the invoice.",
            "first buy apples second buy pears third buy milk": "1. Buy apples\n2. Buy pears\n3. Buy milk",
            "First: apples; and second: pears.": "1. Apples\n2. Pears.",
            "1st review the report, 2nd send it, third archive it.": "1. Review the report\n2. Send it\n3. Archive it.",
            "FIRST review this, SECOND send that.": "1. Review this\n2. Send that.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_normal_ordinals_and_incomplete_lists_are_unchanged(self):
        samples = ["This is my first time here and I am having second thoughts.",
                   "First review the first draft. Send the second version tomorrow.",
                   "Our first year went well and our second year was better.",
                   "First place and second place receive prizes.",
                   "First review the report.", "Second review the report. Third send it.",
                   "First review it, third send it.", "First review it, second send it, second archive it.",
                   "First, second, third.", "First review it, second.",
                   "First review it, second send it, fourth archive it."]
        for raw in samples:
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), raw)

    def test_ordinals_used_as_adjectives_or_idioms_stay_prose(self):
        samples = ["At first I was unsure, but on second look it was fine.",
                   "The first quarter was strong and the second quarter was weak.",
                   "My first job was in Austin. My second job was in Dallas.",
                   "Our first customer paid late, and our second customer never paid.",
                   "He finished first, she finished second, and I finished third.",
                   "First we eat, then we talk. Second helpings are allowed.",
                   "First thing tomorrow, call Sam. Second thing, email Bob."]
        for raw in samples:
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), raw)
        self.assertEqual(self.a.polish("First of all, thank you. Second of all, I am sorry."),
                         "1. Thank you.\n2. I am sorry.")

    def test_tidy_keeps_real_words_that_look_like_fillers(self):
        samples = {"I was in the ER.": "I was in the ER.",
                   "Mm-hmm, I agree.": "Mm-hmm, I agree.",
                   'He said "um" twice.': 'He said "um" twice.',
                   "Call 5 5 5 1 2 3 4 now.": "Call 5 5 5 1 2 3 4 now.",
                   "Um, I think the the the plan is fine.": "I think the plan is fine."}
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.tidy(raw), expected)

    def test_punctuation_only_transcripts_are_not_speech(self):
        for raw in ("!", ".", "...", "?!", "", " - "):
            self.assertFalse(self.a.spoken(raw), raw)
        for raw in ("Hi.", "5", "é"):
            self.assertTrue(self.a.spoken(raw), raw)

    def test_ordinals_inside_items_are_preserved(self):
        raw = "First review the first draft on the second floor. Second send the second version."
        self.assertEqual(self.a.polish(raw),
                         "1. Review the first draft on the second floor.\n2. Send the second version.")

    def test_self_correction_is_cleaned_before_list_formatting(self):
        raw = "First send it Tuesday, I mean Wednesday. Second archive it."
        with patch.object(self.a, "clean", return_value="First send it Wednesday. Second archive it.") as clean:
            self.assertEqual(self.a.polish(raw), "1. Send it Wednesday.\n2. Archive it.")
        clean.assert_called_once_with(raw)

    def test_list_formatting_preserves_dictionary_spellings(self):
        with patch.object(self.a, "term_list", return_value=["apple tree -> AppleTree"]):
            self.assertEqual(self.a.polish("First work on apple tree, second plant pears."),
                             "1. Work on AppleTree\n2. Plant pears.")

    def test_all_ten_ordinals_and_repeated_formatting(self):
        words = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
        raw = ", ".join(f"{word} do task {i}" for i, word in enumerate(words, 1))
        expected = "\n".join(f"{i}. Do task {i}" for i in range(1, 11))
        self.assertEqual(self.a.polish(raw), expected)
        self.assertEqual(self.a.spoken_list(expected), expected)

    def test_paste_keeps_numbered_lines_in_a_single_clipboard_paste(self):
        text = self.a.polish("First review it, second send it.")
        board = Mock()
        board.pasteboardItems.return_value = []
        appkit = Mock()
        appkit.NSPasteboard.generalPasteboard.return_value = board
        appkit.NSPasteboardTypeString = self.a.AppKit.NSPasteboardTypeString
        with patch.object(self.a, "AppKit", appkit), \
             patch.object(self.a, "press") as press, patch.object(self.a.threading, "Timer"):
            self.a.paste(text)
        board.setString_forType_.assert_called_once_with("1. Review it\n2. Send it.", self.a.AppKit.NSPasteboardTypeString)
        press.assert_called_once_with(9, self.a.COMMAND)

    def test_user_unnumbered_agenda_becomes_four_bullets(self):
        raw = ("Here is our agenda for the day. Cooking home, working on the app, "
               "freely running on the loan and working crazy on the computer.")
        expected = ("Here is our agenda for the day.\n\n- Cooking home\n- Working on the app\n"
                    "- Freely running on the loan\n- Working crazy on the computer.")
        with patch.object(self.a, "llm") as llm:
            self.assertEqual(self.a.polish(raw), expected)
            llm.assert_not_called()

    def test_bullet_introductions_and_inline_lists(self):
        samples = {
            "Here are my tasks: writing code, testing it, and shipping it.":
                "Here are my tasks:\n\n- Writing code\n- Testing it\n- Shipping it.",
            "Our agenda includes cooking at home, working on the app and running outside.":
                "Our agenda includes\n\n- Cooking at home\n- Working on the app\n- Running outside.",
            "My goals for today are cooking at home and working on the app.":
                "My goals for today are\n\n- Cooking at home\n- Working on the app.",
            "Shopping list: apples, bread, and milk.":
                "Shopping list:\n\n- Apples\n- Bread\n- Milk.",
            "Today's priorities: testing the app; fixing the errors; shipping it.":
                "Today's priorities:\n\n- Testing the app\n- Fixing the errors\n- Shipping it.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_bullets_keep_compound_phrases_and_protected_commas(self):
        samples = {
            "Our agenda: research and development, testing the app and reviewing the budget.":
                "Our agenda:\n\n- Research and development\n- Testing the app\n- Reviewing the budget.",
            'My tasks: reading "Research, Development", writing the report.':
                'My tasks:\n\n- Reading "Research, Development"\n- Writing the report.',
            "My tasks: checking costs (food, transport), paying 1,200 dollars, and filing receipts.":
                "My tasks:\n\n- Checking costs (food, transport)\n- Paying 1,200 dollars\n- Filing receipts.",
            "My tasks: studying cooking and cleaning techniques, testing the app.":
                "My tasks:\n\n- Studying cooking and cleaning techniques\n- Testing the app.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_bullets_keep_greetings_and_closing_prose(self):
        raw = "Hello team. Here is our agenda. Cooking at home, working on the app. Thanks for your help."
        self.assertEqual(self.a.polish(raw),
                         "Hello team. Here is our agenda.\n\n- Cooking at home\n- Working on the app.\n\nThanks for your help.")

    def test_ordinary_comma_prose_and_single_tasks_stay_unchanged(self):
        samples = ["I enjoy cooking at home, working on the app, and running outside.",
                   "We discussed the agenda yesterday, worked on the app, and went home.",
                   "Our agenda was useful, interesting, and practical.",
                   "Here is our agenda. Working on the app.",
                   "The agenda is ready, but we have not agreed on the tasks.",
                   "The agenda is new, exciting, and useful.",
                   "My tasks: cooking at home,, working on the app.",
                   'My tasks: reading "Cooking, writing the report.']
        for raw in samples:
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), raw)

    def test_numbered_agenda_keeps_numbered_format(self):
        self.assertEqual(self.a.polish("Our agenda: first cook dinner, second work on the app."),
                         "Our agenda:\n\n1. Cook dinner\n2. Work on the app.")

    def test_bullet_formatting_preserves_existing_lists(self):
        for text in ["Our agenda:\n\n- Cooking, cleaning\n- Working on the app.",
                     "Our agenda:\n\n1. Cooking, cleaning\n2. Working on the app."]:
            self.assertEqual(self.a.spoken_bullets(text), text)

    def test_bullets_format_after_corrections_and_dictionary(self):
        with patch.object(self.a, "clean", return_value="My tasks: working on say down, cooking at home."), \
             patch.object(self.a, "term_list", return_value=["say down -> Saydown"]):
            self.assertEqual(self.a.polish("My tasks: working on say down, cooking, I mean cooking at home."),
                             "My tasks:\n\n- Working on Saydown\n- Cooking at home.")

    def test_bullet_paste_keeps_all_items_in_one_clipboard_operation(self):
        text = self.a.polish("My tasks: cooking at home, working on the app.")
        board, appkit = Mock(), Mock()
        board.pasteboardItems.return_value = []
        appkit.NSPasteboard.generalPasteboard.return_value = board
        appkit.NSPasteboardTypeString = self.a.AppKit.NSPasteboardTypeString
        with patch.object(self.a, "AppKit", appkit), patch.object(self.a, "press") as press, \
             patch.object(self.a.threading, "Timer"):
            self.a.paste(text)
        board.setString_forType_.assert_called_once_with("My tasks:\n\n- Cooking at home\n- Working on the app.",
                                                        appkit.NSPasteboardTypeString)
        press.assert_called_once_with(9, self.a.COMMAND)

    def test_user_repeated_intent_agenda_without_intro_punctuation(self):
        raw = ("This is the agenda of the day I am going to cook, I am going to drive my car, "
               "I am going to clean the grasses and I am going to make the pesticides to clean "
               "the bad insects from the garden")
        expected = ("This is the agenda of the day\n\n- I am going to cook\n- I am going to drive my car\n"
                    "- I am going to clean the grasses\n- I am going to make the pesticides to clean "
                    "the bad insects from the garden")
        with patch.object(self.a, "llm") as llm:
            self.assertEqual(self.a.polish(raw), expected)
            llm.assert_not_called()

    def test_intent_lists_handle_punctuation_contractions_and_multiple_ands(self):
        samples = {
            "This is our agenda. I am going to cook, I am going to drive and I am going to clean.":
                "This is our agenda.\n\n- I am going to cook\n- I am going to drive\n- I am going to clean.",
            "This is our agenda I'm going to cook and I'm going to drive and I'm going to clean.":
                "This is our agenda\n\n- I'm going to cook\n- I'm going to drive\n- I'm going to clean.",
            "Here are our tasks we will cook we will drive we will clean":
                "Here are our tasks\n\n- We will cook\n- We will drive\n- We will clean",
            "My tasks for today are I need to cook, I need to drive and I need to clean.":
                "My tasks for today are\n\n- I need to cook\n- I need to drive\n- I need to clean.",
            "That's my agenda I’m going to cook, I’m going to drive.":
                "That's my agenda\n\n- I’m going to cook\n- I’m going to drive.",
            "This is the agenda of the day cooking at home, working on the app.":
                "This is the agenda of the day\n\n- Cooking at home\n- Working on the app.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_repeated_action_sentences_are_bullets_and_keep_closing_prose(self):
        samples = {
            "This is our agenda. I will cook. I will drive. I will clean. Thanks for your help.":
                "This is our agenda.\n\n- I will cook.\n- I will drive.\n- I will clean.\n\nThanks for your help.",
            "Our agenda: Cooking at home. Working on the app. Thank you.":
                "Our agenda:\n\n- Cooking at home.\n- Working on the app.\n\nThank you.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_repeated_intent_preserves_compounds_subordinate_clauses_and_quotes(self):
        samples = {
            "My tasks: I will cook fish and chips, I will drive and I will clean.":
                "My tasks:\n\n- I will cook fish and chips\n- I will drive\n- I will clean.",
            "Our agenda: I will decide whether I will drive, I will cook.":
                "Our agenda:\n\n- I will decide whether I will drive\n- I will cook.",
            'Our agenda: I will read "I will cook, I will drive", I will clean.':
                'Our agenda:\n\n- I will read "I will cook, I will drive"\n- I will clean.',
            "My tasks: I will cook (if I will be home), I will work on the app.":
                "My tasks:\n\n- I will cook (if I will be home)\n- I will work on the app.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_intent_prose_and_single_action_are_not_lists(self):
        samples = ["I am going to cook, I am going to drive my car and I am going to clean the garden.",
                   "This is the agenda of the day I am going to cook.",
                   "This is the agenda of the day I will decide whether I will drive.",
                   "This is the agenda of the day we discussed yesterday.",
                   "We changed the agenda because I will cook, I will drive, and I will clean."]
        for raw in samples:
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), raw)

    def test_intent_agenda_is_one_multiline_clipboard_paste(self):
        text = self.a.polish("This is our agenda I will cook, I will drive and I will clean.")
        board, appkit = Mock(), Mock()
        board.pasteboardItems.return_value = []
        appkit.NSPasteboard.generalPasteboard.return_value = board
        appkit.NSPasteboardTypeString = self.a.AppKit.NSPasteboardTypeString
        with patch.object(self.a, "AppKit", appkit), patch.object(self.a, "press") as press, \
             patch.object(self.a.threading, "Timer"):
            self.a.paste(text)
        board.setString_forType_.assert_called_once_with(
            "This is our agenda\n\n- I will cook\n- I will drive\n- I will clean.", appkit.NSPasteboardTypeString)
        press.assert_called_once_with(9, self.a.COMMAND)

    def test_broader_list_wording_and_imperatives(self):
        samples = {
            "Here are the things I need to do: cook dinner, drive my car and clean the garden.":
                "Here are the things I need to do:\n\n- Cook dinner\n- Drive my car\n- Clean the garden.",
            "Here's what I need to do buy apples, cook dinner and review the report.":
                "Here's what I need to do\n\n- Buy apples\n- Cook dinner\n- Review the report.",
            "I have three things to do: cook dinner, drive my car and clean the garden.":
                "I have three things to do:\n\n- Cook dinner\n- Drive my car\n- Clean the garden.",
            "For today, my plans are cooking dinner, driving my car and cleaning the garden.":
                "For today, my plans are\n\n- Cooking dinner\n- Driving my car\n- Cleaning the garden.",
            "My to-do list: cook dinner and drive my car and clean the garden.":
                "My to-do list:\n\n- Cook dinner\n- Drive my car\n- Clean the garden.",
            "Shopping list: apples, milk and bread.":
                "Shopping list:\n\n- Apples\n- Milk\n- Bread.",
            "Shopping list: eggs, salt and pepper.":
                "Shopping list:\n\n- Eggs\n- Salt and pepper.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_abbreviations_and_protected_punctuation_in_lists(self):
        samples = {
            'My tasks: call Dr. Lee, read "Draft. Final, approved", and clean the garden. Thanks.':
                'My tasks:\n\n- Call Dr. Lee\n- Read "Draft. Final, approved"\n- Clean the garden.\n\nThanks.',
            'My tasks: I will read "Cook, drive" and I will clean the garden.':
                'My tasks:\n\n- I will read "Cook, drive"\n- I will clean the garden.',
            "Our agenda: checking costs (food, transport), paying 1,200 dollars and reviewing the budget.":
                "Our agenda:\n\n- Checking costs (food, transport)\n- Paying 1,200 dollars\n- Reviewing the budget.",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.a.polish(raw), expected)

    def test_formatted_lists_and_paragraphs_survive_polish(self):
        samples = ["Our agenda:\n\n- Cooking, cleaning\n- Working on the app.",
                   "Our agenda:\n\n1. Cooking, cleaning\n2. Working on the app.",
                   "Hello team.\n\nHere is the report."]
        for text in samples:
            self.assertEqual(self.a.polish(text), text)

    def test_quoted_ordinals_and_unbalanced_lists_are_not_reformatted(self):
        samples = ['She said "first cook dinner, second drive home" to me.',
                   'My tasks: read "First cook dinner, second drive home", review the report.',
                   'My tasks: buy food (milk, bread], clean the garden.',
                   '"My tasks: cook dinner, clean the garden."']
        for text in samples:
            if text.startswith('My tasks: read'):
                self.assertEqual(self.a.polish(text),
                                 'My tasks:\n\n- Read "First cook dinner, second drive home"\n- Review the report.')
            else:
                self.assertEqual(self.a.polish(text), text)

    def test_permission_requests_never_reset_existing_grants(self):
        for listen, post in [(True, True), (False, True), (True, False), (False, False)]:
            with patch.object(self.a.Quartz, "CGPreflightListenEventAccess", return_value=listen), \
                 patch.object(self.a.Quartz, "CGPreflightPostEventAccess", return_value=post), \
                 patch.object(self.a.Quartz, "CGRequestListenEventAccess") as request_listen, \
                 patch.object(self.a.Quartz, "CGRequestPostEventAccess") as request_post, \
                 patch.object(self.a.subprocess, "run") as run:
                self.assertEqual(self.a.request_keyboard_access(), (listen, post))
                self.assertEqual(request_listen.call_count, int(not listen))
                self.assertEqual(request_post.call_count, int(not post))
                run.assert_not_called()

    def test_settings_state_and_requests_are_safe_during_real_main_startup(self):
        server_class = Mock()
        def server(address, handler_class):
            server_class.handler = handler_class
            return Mock()
        def at_startup():
            handler = self.handler("/api/state")
            handler.__class__ = server_class.handler
            with patch.object(self.a, "cloud_key_saved", return_value=False), \
                 patch.object(self.a, "microphones", return_value=[]):
                handler.do_GET()
            self.assertFalse(handler.reply.call_args.args[0]["note"])
            for route in ("/api/note", "/api/ask-microphone"):
                handler = self.handler(route)
                handler.__class__ = server_class.handler
                handler.do_POST()
                self.assertEqual(handler.reply.call_args.kwargs["status"], 503)
                self.assertIn("still starting", handler.reply.call_args.args[0]["error"])
            raise RuntimeError("test startup boundary")
        with patch.object(self.a, "ThreadingHTTPServer", side_effect=server), \
             patch.object(self.a.threading, "Thread") as thread:
            thread.return_value.start.side_effect = at_startup
            with self.assertRaisesRegex(RuntimeError, "test startup boundary"):
                self.a.main()

    def test_bad_history_lines_do_not_hide_valid_entries_or_modify_the_file(self):
        good = {"ts": 1_700_000_000, "text": "Keep this.", "audio_s": 3.5}
        bad = ["{unfinished", "[]", '{}', json.dumps(good | {"ts": "yesterday"}),
               json.dumps(good | {"audio_s": -1}), json.dumps(good | {"ts": float("inf")})]
        raw = json.dumps(good) + "\n" + "\n".join(bad) + "\n"
        self.a.HISTORY.write_text(raw)
        self.assertEqual(self.a.read_history(), [good])
        self.assertEqual(self.a.HISTORY.read_text(), raw)
        self.a.HISTORY.write_text("")

    def test_completed_dictation_stays_on_clipboard_if_accessibility_is_missing(self):
        board, appkit = Mock(), Mock()
        appkit.NSPasteboard.generalPasteboard.return_value = board
        appkit.NSPasteboardTypeString = self.a.AppKit.NSPasteboardTypeString
        with patch.object(self.a, "AppKit", appkit), patch.object(self.a, "keyboard_access", return_value=(True, False)), \
             patch.object(self.a, "paste") as paste:
            self.assertFalse(self.a.deliver("- Cook dinner\n- Drive home"))
        board.setString_forType_.assert_called_once_with("- Cook dinner\n- Drive home", appkit.NSPasteboardTypeString)
        paste.assert_not_called()
        self.assertIn("clipboard", self.a.status["delivery"])

    def test_granted_accessibility_uses_paste_and_clears_delivery_warning(self):
        with patch.object(self.a, "keyboard_access", return_value=(True, True)), patch.object(self.a, "paste") as paste:
            self.assertTrue(self.a.deliver("Hello"))
        paste.assert_called_once_with("Hello")
        self.assertEqual(self.a.status["delivery"], "")

    def test_failed_local_warmup_returns_with_visible_error_instead_of_killing_worker(self):
        local = Mock(side_effect=OSError("offline"))
        self.a.status["loading"] = True
        self.a.load_local_speech(local)
        self.assertFalse(self.a.status["loading"])
        self.assertIn("could not load", self.a.status["speech_error"])
        local.reset_mock(side_effect=True)
        self.a.load_local_speech(local)
        self.assertEqual(self.a.status["speech_error"], "")

    def test_transcribed_comma_introductions_are_recognized(self):
        samples = {
            "I have three things to do, cook dinner, drive my car and clean the garden.":
                "I have three things to do,\n\n- Cook dinner\n- Drive my car\n- Clean the garden.",
            "My tasks, I will cook fish and chips, I will drive and I will clean.":
                "My tasks,\n\n- I will cook fish and chips\n- I will drive\n- I will clean.",
            "My tasks, call Dr. Lee, review the report and clean the garden.":
                "My tasks,\n\n- Call Dr. Lee\n- Review the report\n- Clean the garden.",
        }
        for raw, expected in samples.items():
            self.assertEqual(self.a.polish(raw), expected)

    def test_local_noun_inference_preserves_verbatim_items_and_compounds(self):
        for raw, reply, expected in [
            ("Shopping list apples milk and bread", '["apples", "milk", "bread"]',
             "Shopping list\n\n- Apples\n- Milk\n- Bread"),
            ("Shopping list eggs salt and pepper", '["eggs", "salt and pepper"]',
             "Shopping list\n\n- Eggs\n- Salt and pepper"),
            ("Shopping list banana bread whole milk and eggs.", '["banana bread", "whole milk", "eggs"]',
             "Shopping list\n\n- Banana bread\n- Whole milk\n- Eggs"),
        ]:
            with patch.object(self.a, "llm", return_value=reply) as llm:
                self.assertEqual(self.a.polish(raw), expected)
                llm.assert_called_once()

    def test_noun_model_cannot_invent_drop_reorder_or_split_compound_items(self):
        raw = "Shopping list eggs salt and pepper"
        for reply in ['["eggs", "insecticide"]', '["eggs", "pepper"]', '["pepper", "eggs"]',
                      '["eggs", "salt", "pepper"]', '["egg", "salt and pepper"]', '{}', 'not json']:
            with patch.object(self.a, "llm", return_value=reply):
                self.assertEqual(self.a.polish(raw), raw)
        with patch.object(self.a, "llm", side_effect=OSError("unavailable")):
            self.assertEqual(self.a.polish(raw), raw)

    def test_clear_lists_and_ordinary_prose_never_call_noun_model(self):
        with patch.object(self.a, "llm") as llm:
            self.a.polish("Shopping list: apples, milk and bread.")
            self.a.polish("My tasks: cook dinner and drive home.")
            self.a.polish("I enjoy apples, milk and bread.")
            llm.assert_not_called()

    def test_openrouter_failure_and_missing_key_use_local(self):
        self.config(speech_provider="openrouter")
        with patch.object(self.a, "cloud_key", return_value=""), patch.object(self.a, "hear") as hear:
            self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            hear.assert_not_called()
            self.assertIn("No OpenRouter key", self.a.speech_status["message"])
        with patch.object(self.a, "cloud_key", return_value="router-secret"), patch.object(self.a, "hear", side_effect=TimeoutError("router-secret")):
            self.assertEqual(self.a.transcribe(self.audio, "en", self.local)[1], "local")
            self.assertNotIn("router-secret", self.a.speech_status["message"])

    def test_openrouter_model_selection_preserves_openai_model(self):
        with patch.object(self.a, "microphones", return_value=[]):
            result = self.a.valid_settings({"speech_provider": "openrouter", "openrouter_model": "openai/gpt-transcribe"})
            self.assertEqual(result["speech_model"], "gpt-transcribe")
            self.assertEqual(result["openrouter_model"], "openai/gpt-transcribe")
            with self.assertRaises(ValueError):
                self.a.valid_settings({"openrouter_model": "gpt-transcribe"})

    def test_openrouter_key_save_and_remove_are_scoped_to_router(self):
        for body, value in [({"key": "fake-router-key-123456789", "provider": "openrouter"}, "fake-router-key-123456789"),
                            ({"key": "", "provider": "openrouter", "remove": True}, "")]:
            handler = self.handler("/api/speech-key", body)
            with patch.object(self.a, "save_cloud_key") as save:
                handler.do_POST()
                save.assert_called_once_with(value, "openrouter")
            self.assertEqual(self.a.settings()["speech_provider"], "local")

    def test_invalid_key_provider_is_rejected(self):
        for provider in ["local", "unknown", []]:
            handler = self.handler("/api/speech-key", {"key": "fake-router-key-123456789", "provider": provider})
            with patch.object(self.a, "save_cloud_key") as save:
                handler.do_POST()
                save.assert_not_called()
            self.assertEqual(handler.reply.call_args.kwargs["status"], 400)

    def test_openrouter_state_reports_only_router_models_and_key_presence(self):
        self.config(speech_provider="openrouter")
        handler = self.handler("/api/state")
        with patch.object(self.a, "cloud_key_saved", return_value=True) as saved, patch.object(self.a, "microphones", return_value=[]):
            handler.do_GET()
        saved.assert_called_once_with("openrouter")
        self.assertEqual(handler.reply.call_args.args[0]["speech"]["models"], self.a.OPENROUTER_MODELS)


if __name__ == "__main__":
    unittest.main()
